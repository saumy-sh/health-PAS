"""
agent1_document_intelligence.py  (IMPROVED + FIXED)
─────────────────────────────────────────────────────
Agent 1 — Document Intelligence

FIXES vs previous improved version:
  ✅ Token overflow fixed — two-pass extraction strategy:
       Pass 1 (main call): core fields + documents_identified  → max_tokens=4096
       Pass 2 (enrichment call, text-only): per_document_extraction → max_tokens=2000
     Pass 2 is lightweight (no images) so it won't overflow.
  ✅ Robust JSON recovery — _safe_parse_json() tries 4 strategies before giving up:
       1. Direct json.loads
       2. Strip markdown fences
       3. Extract largest {...} block via regex
       4. Truncation repair: close open strings/arrays/objects
  ✅ max_tokens raised to 4096 for Pass 1 (Bedrock Nova Pro supports up to 5120)
  ✅ Graceful degradation — if per_document_extraction fails, pipeline continues
     using only the core extraction (same behaviour as old agent1)

KEY IMPROVEMENTS retained from improved version:
  ① Full document content (actual values) passed downstream
  ② completeness_summary per document (score + missing critical fields)
  ③ document_content_map  — flat { doc_type → content } lookup
  ④ field_source_map       — { field → which document it came from }
  ⑤ null_fields_by_doc    — { doc_type → [blank fields] }
"""

import json
import re
import base64
import io
from pathlib import Path
from typing import Union

import fitz          # PyMuPDF — pip install pymupdf
from PIL import Image

from bedrock_client import invoke, PRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Supported formats
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PDF_EXTENSIONS   = {".pdf"}

CRITICAL_FIELDS_BY_DOC = {
    "Doctor Notes": [
        "diagnosis", "icd10_codes", "clinical_findings",
        "requested_procedure", "cpt_codes", "ordering_physician",
        "symptom_duration_weeks",
    ],
    "Insurance Card": [
        "insurer_name", "policy_number", "member_id", "plan_type",
    ],
    "Patient Information Sheet": [
        "patient_name", "dob", "mrn",
    ],
    "Lab Report": [
        "test_name", "results",
    ],
    "Medical Pretreatment Estimate": [
        "total_estimated_cost", "cpt_codes",
    ],
    "Physician Referral": [
        "referring_physician", "referred_to", "reason_for_referral",
    ],
    "Prescription": [
        "medication_name", "dosage", "prescribing_physician", "duration",
    ],
    "Prior Treatment Documentation": [
        "prior_treatments",
    ],
}

FIELD_DESCRIPTIONS = {
    "diagnosis":              "primary diagnosis",
    "icd10_codes":            "ICD-10 code(s)",
    "clinical_findings":      "physical exam findings",
    "requested_procedure":    "procedure name",
    "cpt_codes":              "CPT code(s)",
    "ordering_physician":     "ordering physician name",
    "symptom_duration_weeks": "symptom duration in weeks",
    "insurer_name":           "insurance company name",
    "policy_number":          "policy number",
    "member_id":              "member ID",
    "plan_type":              "plan type (PPO/HMO/etc)",
    "patient_name":           "patient full name",
    "dob":                    "date of birth",
    "mrn":                    "medical record number",
    "test_name":              "lab test name",
    "results":                "lab results",
    "total_estimated_cost":   "total estimated cost",
    "referring_physician":    "referring physician",
    "referred_to":            "referred-to specialist",
    "reason_for_referral":    "reason for referral",
    "medication_name":        "medication name",
    "dosage":                 "dosage",
    "prescribing_physician":  "prescribing physician",
    "duration":               "prescription duration",
    "prior_treatments":       "list of prior treatments",
}


# ─────────────────────────────────────────────────────────────────────────────
# File loaders
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_block(image_path: str, max_size: tuple = (1600, 1600)) -> dict:
    path = Path(image_path)
    fmt  = path.suffix.lstrip(".").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in {"png", "jpeg", "webp", "gif"}:
        fmt = "png"
    img = Image.open(path)
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    save_fmt = "PNG" if fmt in {"png", "gif"} else fmt.upper()
    img.save(buf, format=save_fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    out_fmt = "png" if save_fmt == "PNG" else fmt
    print(f"  [Agent1] Image  : {path.name}  ({img.size[0]}x{img.size[1]}px)  → {out_fmt}")
    return {"image": {"format": out_fmt, "source": {"bytes": b64}}}


def _pdf_to_blocks(pdf_path: str, dpi: int = 150) -> list:
    doc    = fitz.open(pdf_path)
    blocks = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        buf = io.BytesIO(pix.tobytes("png"))
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        blocks.append({"image": {"format": "png", "source": {"bytes": b64}}})
        print(f"  [Agent1] PDF page {i+1}/{len(doc)} : {Path(pdf_path).name}")
    doc.close()
    return blocks


def _file_to_blocks(file_path: str) -> list:
    path   = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return [_image_to_block(file_path)]
    elif suffix in PDF_EXTENSIONS:
        return _pdf_to_blocks(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _normalise_input(documents) -> list:
    if isinstance(documents, str):
        return [documents]
    if isinstance(documents, list):
        return documents
    if isinstance(documents, dict):
        return list(documents.values())
    raise TypeError(f"Unsupported input type: {type(documents)}")


def _load_documents(file_paths: list) -> list:
    all_blocks = []
    for fp in file_paths:
        try:
            all_blocks.extend(_file_to_blocks(fp))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [Agent1] ⚠  Skipping '{fp}': {exc}")
    return all_blocks


# ─────────────────────────────────────────────────────────────────────────────
# Robust JSON parser — 4-strategy recovery
# ─────────────────────────────────────────────────────────────────────────────

def _repair_truncated_json(text: str) -> str:
    """
    Attempt to close an incomplete JSON string by tracking open
    brackets/braces and appending closing tokens.
    Handles the most common truncation pattern (mid-string or mid-array).
    """
    # Close any open string first (odd number of unescaped quotes)
    in_string = False
    escaped   = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        text += '"'   # close the open string

    # Count unclosed braces/brackets
    stack = []
    in_string = False
    escaped   = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in "{[":
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    # Close remaining open structures
    closing = {"[": "]", "{": "}"}
    for opener in reversed(stack):
        text += closing[opener]

    return text


def _safe_parse_json(raw: str, context: str = "") -> dict:
    """
    Try 4 strategies to parse potentially truncated/malformed JSON.
    Returns parsed dict on success, empty dict on total failure.
    """
    text = raw.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 3: extract the largest {...} block
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: repair truncated JSON
    try:
        repaired = _repair_truncated_json(text)
        result   = json.loads(repaired)
        print(f"  [Agent1] ⚠  JSON truncation detected{' in ' + context if context else ''} — repaired successfully")
        return result
    except (json.JSONDecodeError, Exception) as e:
        print(f"  [Agent1] ❌ JSON parse failed{' in ' + context if context else ''}: {e}")
        print(f"  [Agent1] Raw response (first 500 chars): {raw[:500]}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 prompt — core extraction (no per_document_extraction to save tokens)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a highly accurate medical document intelligence agent. "
            "Extract all fields required for insurance prior-authorization from clinical documents. "
            "Documents are NOT labelled — YOU identify each one. "
            "Return ONLY valid JSON — no markdown, no preamble. "
            "Set missing fields to null. Dates in YYYY-MM-DD. Currency as numbers only."
        )
    }
]

CORE_EXTRACTION_PROMPT = """Inspect every document image. Identify each document type and extract fields.

RULES:
1. prior_treatments: each entry = one treatment + duration + outcome on one line.
   CORRECT: ["NSAIDs (Ibuprofen) - 4 weeks - Minimal relief"]
   WRONG:   ["NSAIDs", "4 weeks", "Minimal relief"]
2. member_id vs policy_number: read independently. "Member ID" → member_id. "Policy No" → policy_number.
3. cpt_codes: collect ALL CPT codes from ALL documents, deduplicated.
4. documents_identified: only mark true if you can clearly read that document type.

Return this JSON (keep values concise — avoid long strings to stay within limits):
{
  "documents_identified": {
    "lab_report": bool, "doctor_notes": bool, "patient_info": bool,
    "insurance_card": bool, "pretreatment_estimate": bool,
    "prior_treatment_documentation": bool, "procedure_order": bool,
    "physician_referral": bool, "prescription": bool
  },
  "patient": {
    "name": string, "dob": string, "gender": string, "mrn": string,
    "address": string, "city": string, "state": string, "zip": string,
    "phone": string, "email": string, "emergency_contact_name": string
  },
  "insurance": {
    "insurer_name": string, "plan_type": string, "policy_number": string,
    "member_id": string, "group_number": string, "copay_specialist": number,
    "copay_office_visit": number, "copay_er": number,
    "administered_by": string, "underwritten_by": string
  },
  "clinical": {
    "diagnosis": string, "icd10_codes": [string], "chief_complaint": string,
    "clinical_findings": string, "prior_treatments": [string],
    "requested_procedure": string, "cpt_codes": [string],
    "symptom_duration_weeks": number, "pain_score": number,
    "ordering_physician": string, "physician_specialty": string,
    "physician_phone": string, "physician_npi": string
  },
  "lab_results": {
    "test_name": string, "test_date": string, "results": string,
    "reference_ranges": string, "ordering_physician": string,
    "lab_name": string, "abnormal_flags": [string]
  },
  "cost_estimate": {
    "total_estimated_cost": number,
    "line_items": [{"description": string, "cpt_code": string, "estimated_price": number}]
  },
  "facility": {
    "hospital_name": string, "hospital_address": string,
    "date_of_service": string, "date_of_proposed_treatment": string
  },
  "pharmacy": {
    "pharmacy_name": string, "pharmacy_address": string, "pharmacy_phone": string
  },
  "prescription": {
    "medication_name": string, "dosage": string, "frequency": string,
    "duration": string, "prescribing_physician": string,
    "date_prescribed": string, "refills": number
  },
  "referral": {
    "referring_physician": string, "referred_to": string,
    "reason_for_referral": string, "referral_date": string,
    "referral_validity_days": number
  },
  "prescribed_medications": [string],
  "extraction_confidence": "high" | "medium" | "low",
  "missing_fields": [string]
}"""


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 prompt — per-document enrichment (text-only, no images)
# ─────────────────────────────────────────────────────────────────────────────

def _build_enrichment_prompt(core_data: dict) -> str:
    """
    Pass 2: given the already-extracted core data, ask the model to produce
    per_document_extraction with completeness info. No images — text-only call.
    This keeps Pass 2 fast and within token limits.
    """
    return f"""Based on the following already-extracted medical record data, produce a
per_document_extraction array. For each document type that was identified,
list the fields extracted from it and which critical fields are blank/missing.

EXTRACTED DATA:
{json.dumps(core_data, indent=2, default=str)[:3000]}

Return ONLY this JSON (no markdown):
{{
  "per_document_extraction": [
    {{
      "document_type": string,
      "extracted_fields": {{ field: value }},
      "blank_fields": [string],
      "illegible_fields": [string],
      "notes": string
    }}
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _build_documents_present(data: dict) -> dict:
    nova_id = data.get("documents_identified", {})
    lab     = data.get("lab_results", {})
    clin    = data.get("clinical", {})
    ins     = data.get("insurance", {})
    p       = data.get("patient", {})
    cost    = data.get("cost_estimate", {})
    ref     = data.get("referral", {})
    rx      = data.get("prescription", {})

    def _has_values(d: dict) -> bool:
        return any(v for v in d.values() if v not in (None, [], "", "null"))

    return {
        "lab_report":                   nova_id.get("lab_report", False) and _has_values(lab),
        "doctor_notes":                 nova_id.get("doctor_notes", False) and bool(clin.get("diagnosis") or clin.get("clinical_findings")),
        "patient_info":                 nova_id.get("patient_info", False) and bool(p.get("name") and p.get("dob")),
        "insurance_card":               nova_id.get("insurance_card", False) and bool(ins.get("policy_number") and ins.get("insurer_name")),
        "pretreatment_estimate":        nova_id.get("pretreatment_estimate", False) and bool(cost.get("total_estimated_cost")),
        "prior_treatment_documentation": nova_id.get("prior_treatment_documentation", False) and bool(clin.get("prior_treatments")),
        "procedure_order":              nova_id.get("procedure_order", False) and bool(clin.get("requested_procedure") and clin.get("cpt_codes")),
        "physician_referral":           nova_id.get("physician_referral", False) and bool(ref.get("referring_physician") or ref.get("reason_for_referral")),
        "prescription":                 nova_id.get("prescription", False) and bool(rx.get("medication_name")),
    }


def _score_completeness(content: dict, doc_type: str) -> dict:
    critical = CRITICAL_FIELDS_BY_DOC.get(doc_type, [])
    if not critical:
        return {"completeness_score": 1.0, "missing_critical_fields": [], "present_critical_fields": [], "null_fields": [k for k, v in content.items() if not v]}

    present, missing = [], []
    for field in critical:
        val = content.get(field)
        if val and val not in ([], "null", "") and str(val).strip():
            present.append(field)
        else:
            missing.append(field)

    return {
        "completeness_score":      round(len(present) / len(critical), 2),
        "missing_critical_fields": missing,
        "present_critical_fields": present,
        "null_fields":             [k for k, v in content.items() if not v and v != 0],
    }


def _build_documents_list(data: dict, docs_present: dict) -> list:
    p     = data.get("patient", {})
    ins   = data.get("insurance", {})
    clin  = data.get("clinical", {})
    lab   = data.get("lab_results", {})
    cost  = data.get("cost_estimate", {})
    fac   = data.get("facility", {})
    pharm = data.get("pharmacy", {})
    ref   = data.get("referral", {})
    rx    = data.get("prescription", {})

    documents = []

    def _add(doc_type, source, content):
        documents.append({
            "document_type": doc_type,
            "source":        source,
            "content":       content,
            "completeness":  _score_completeness(content, doc_type),
        })

    if docs_present.get("lab_report"):
        _add("Lab Report", "lab_results", dict(lab))

    if docs_present.get("doctor_notes"):
        _add("Doctor Notes", "clinical", {
            "diagnosis":              clin.get("diagnosis"),
            "icd10_codes":            clin.get("icd10_codes"),
            "chief_complaint":        clin.get("chief_complaint"),
            "clinical_findings":      clin.get("clinical_findings"),
            "prior_treatments":       clin.get("prior_treatments"),
            "requested_procedure":    clin.get("requested_procedure"),
            "cpt_codes":              clin.get("cpt_codes"),
            "symptom_duration_weeks": clin.get("symptom_duration_weeks"),
            "pain_score":             clin.get("pain_score"),
            "ordering_physician":     clin.get("ordering_physician"),
            "physician_specialty":    clin.get("physician_specialty"),
            "physician_phone":        clin.get("physician_phone"),
            "physician_npi":          clin.get("physician_npi"),
        })

    if docs_present.get("patient_info"):
        _add("Patient Information Sheet", "patient", {
            "patient_name": p.get("name"), "dob": p.get("dob"),
            "gender": p.get("gender"), "mrn": p.get("mrn"),
            "address": p.get("address"), "city": p.get("city"),
            "state": p.get("state"), "zip": p.get("zip"),
            "phone": p.get("phone"), "email": p.get("email"),
            "emergency_contact": p.get("emergency_contact_name"),
        })

    if docs_present.get("insurance_card"):
        _add("Insurance Card", "insurance", {
            "insurer_name": ins.get("insurer_name"), "plan_type": ins.get("plan_type"),
            "policy_number": ins.get("policy_number"), "member_id": ins.get("member_id"),
            "group_number": ins.get("group_number"), "copay_specialist": ins.get("copay_specialist"),
            "copay_office_visit": ins.get("copay_office_visit"), "copay_er": ins.get("copay_er"),
            "administered_by": ins.get("administered_by"), "underwritten_by": ins.get("underwritten_by"),
        })

    if docs_present.get("pretreatment_estimate"):
        _add("Medical Pretreatment Estimate", "cost_estimate", {
            "total_estimated_cost":       cost.get("total_estimated_cost"),
            "line_items":                 cost.get("line_items", []),
            "cpt_codes":                  [i.get("cpt_code") for i in cost.get("line_items", []) if i.get("cpt_code")],
            "date_of_proposed_treatment": fac.get("date_of_proposed_treatment"),
            "hospital_name":              fac.get("hospital_name"),
        })

    if docs_present.get("physician_referral"):
        _add("Physician Referral", "referral", {
            "referring_physician":    ref.get("referring_physician"),
            "referred_to":            ref.get("referred_to"),
            "reason_for_referral":    ref.get("reason_for_referral"),
            "referral_date":          ref.get("referral_date"),
            "referral_validity_days": ref.get("referral_validity_days"),
        })

    if docs_present.get("prescription"):
        _add("Prescription", "prescription", {
            "medication_name":       rx.get("medication_name"),
            "dosage":                rx.get("dosage"),
            "frequency":             rx.get("frequency"),
            "duration":              rx.get("duration"),
            "prescribing_physician": rx.get("prescribing_physician"),
            "date_prescribed":       rx.get("date_prescribed"),
            "refills":               rx.get("refills"),
        })

    if docs_present.get("prior_treatment_documentation") and not docs_present.get("doctor_notes"):
        _add("Prior Treatment Documentation", "clinical", {
            "prior_treatments": clin.get("prior_treatments", []),
        })

    if pharm.get("pharmacy_name"):
        content = {k: v for k, v in pharm.items() if v}
        documents.append({
            "document_type": "Pharmacy Information", "source": "pharmacy",
            "content": content,
            "completeness": {"completeness_score": 1.0, "missing_critical_fields": [], "null_fields": []},
        })

    return documents


def _build_enrichment_maps(documents: list) -> tuple:
    content_map   = {doc["document_type"]: doc["content"] for doc in documents}
    field_src_map = {}
    null_by_doc   = {}

    for doc in documents:
        for field, value in doc.get("content", {}).items():
            if value and value not in ([], "null", "") and str(value).strip():
                field_src_map[field] = doc["document_type"]
        nulls = doc.get("completeness", {}).get("null_fields", [])
        if nulls:
            null_by_doc[doc["document_type"]] = nulls

    return content_map, field_src_map, null_by_doc


def _build_handoff(core_data: dict, per_doc_data: dict) -> dict:
    p    = core_data.get("patient", {})
    ins  = core_data.get("insurance", {})
    clin = core_data.get("clinical", {})
    lab  = core_data.get("lab_results", {})
    fac  = core_data.get("facility", {})
    cost = core_data.get("cost_estimate", {})

    docs      = _build_documents_present(core_data)
    documents = _build_documents_list(core_data, docs)
    content_map, field_src_map, null_by_doc = _build_enrichment_maps(documents)

    completeness_summary = {
        doc["document_type"]: {
            "score":            doc["completeness"]["completeness_score"],
            "missing_critical": doc["completeness"]["missing_critical_fields"],
            "status": (
                "complete"   if doc["completeness"]["completeness_score"] == 1.0 else
                "partial"    if doc["completeness"]["completeness_score"] >= 0.5  else
                "incomplete"
            ),
        }
        for doc in documents
    }

    return {
        # ── Flat backward-compatible fields ──────────────────────────────────
        "patient_name":               p.get("name"),
        "patient_dob":                p.get("dob"),
        "patient_mrn":                p.get("mrn"),
        "patient_gender":             p.get("gender"),
        "patient_address":            p.get("address"),
        "patient_phone":              p.get("phone"),
        "patient_email":              p.get("email"),
        "emergency_contact":          p.get("emergency_contact_name"),
        "insurer":                    ins.get("insurer_name"),
        "plan_type":                  ins.get("plan_type"),
        "policy_number":              ins.get("policy_number"),
        "group_number":               ins.get("group_number"),
        "member_id":                  ins.get("member_id"),
        "copay_specialist":           ins.get("copay_specialist"),
        "diagnosis":                  clin.get("diagnosis"),
        "icd10_codes":                clin.get("icd10_codes", []),
        "icd10":                      (clin.get("icd10_codes") or [None])[0],
        "procedure":                  clin.get("requested_procedure"),
        "cpt_codes":                  clin.get("cpt_codes", []),
        "cpt":                        (clin.get("cpt_codes") or [None])[0],
        "ordering_physician":         clin.get("ordering_physician"),
        "physician_specialty":        clin.get("physician_specialty"),
        "physician_phone":            clin.get("physician_phone"),
        "physician_npi":              clin.get("physician_npi"),
        "clinical_findings":          clin.get("clinical_findings"),
        "prior_treatments":           clin.get("prior_treatments", []),
        "symptom_duration_weeks":     clin.get("symptom_duration_weeks"),
        "pain_score":                 clin.get("pain_score"),
        "hospital":                   fac.get("hospital_name"),
        "date_of_service":            fac.get("date_of_service"),
        "date_of_proposed_treatment": fac.get("date_of_proposed_treatment"),
        "total_estimated_cost":       cost.get("total_estimated_cost"),
        "cost_line_items":            cost.get("line_items", []),
        "prescribed_medications":     core_data.get("prescribed_medications", []),
        "lab_results":                core_data.get("lab_results", {}),
        "pharmacy":                   core_data.get("pharmacy", {}),
        "documents_present":          docs,

        # ── Full per-doc list with content + completeness ─────────────────────
        "documents":                  documents,

        # ── NEW enrichment structures ─────────────────────────────────────────
        "document_content_map":       content_map,
        "field_source_map":           field_src_map,
        "null_fields_by_doc":         null_by_doc,
        "completeness_summary":       completeness_summary,

        # ── Pass 2 enrichment (may be empty if pass 2 failed gracefully) ──────
        "per_document_extraction":    per_doc_data.get("per_document_extraction", []),

        "extraction_confidence":      core_data.get("extraction_confidence"),
        "missing_fields":             core_data.get("missing_fields", []),
        "_raw":                       core_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(documents: Union[str, list, dict]) -> dict:
    """
    Two-pass extraction:
      Pass 1 (multimodal) : core fields from images  → max_tokens=4096
      Pass 2 (text-only)  : per_document enrichment  → max_tokens=2000
    """
    print("\n[Agent 1] Document Intelligence — START")

    file_paths   = _normalise_input(documents)
    print(f"[Agent 1] Input files ({len(file_paths)}):")
    for fp in file_paths:
        print(f"  • {fp}")

    image_blocks = _load_documents(file_paths)
    print(f"[Agent 1] Total content blocks: {len(image_blocks)}")

    if not image_blocks:
        raise ValueError("[Agent 1] No valid documents could be loaded. Aborting.")

    # ── PASS 1: core extraction (multimodal, higher token limit) ─────────────
    print("[Agent 1] Pass 1 — Core extraction (Nova Pro, max_tokens=4096)...")
    messages_p1 = [{"role": "user", "content": image_blocks + [{"text": CORE_EXTRACTION_PROMPT}]}]

    raw_p1 = invoke(
        model_id=PRO_MODEL_ID,
        messages=messages_p1,
        system=SYSTEM_PROMPT,
        max_tokens=4096,
        temperature=0.1,
    )

    core_data = _safe_parse_json(raw_p1, context="Pass 1 core extraction")
    if not core_data:
        raise RuntimeError(
            "[Agent 1] ❌ Pass 1 failed — could not parse core extraction response.\n"
            f"Raw (first 500 chars): {raw_p1[:500]}"
        )

    # ── PASS 2: per-document enrichment (text-only, lightweight) ─────────────
    print("[Agent 1] Pass 2 — Per-document enrichment (text-only, max_tokens=2000)...")
    try:
        enrichment_prompt = _build_enrichment_prompt(core_data)
        messages_p2 = [{"role": "user", "content": [{"text": enrichment_prompt}]}]

        raw_p2 = invoke(
            model_id=PRO_MODEL_ID,
            messages=messages_p2,
            system=SYSTEM_PROMPT,
            max_tokens=2000,
            temperature=0.1,
        )
        per_doc_data = _safe_parse_json(raw_p2, context="Pass 2 per-doc enrichment")
        if not per_doc_data:
            print("  [Agent1] ⚠  Pass 2 parse failed — continuing without per_document_extraction")
            per_doc_data = {}
    except Exception as e:
        print(f"  [Agent1] ⚠  Pass 2 failed ({e}) — continuing without per_document_extraction")
        per_doc_data = {}

    # ── Build output ──────────────────────────────────────────────────────────
    output = _build_handoff(core_data, per_doc_data)

    print("\n[Agent 1] Extraction complete.")
    print(f"  Confidence : {output.get('extraction_confidence')}")
    print(f"  Missing    : {output.get('missing_fields')}")

    print("\n  Documents identified:")
    for doc, present in output.get("documents_present", {}).items():
        print(f"    [{'FOUND    ' if present else 'NOT FOUND'}] {doc}")

    print("\n  Document completeness:")
    for doc_type, summary in output.get("completeness_summary", {}).items():
        score  = summary["score"]
        status = summary["status"].upper()
        bar    = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"    {doc_type:<38} [{bar}] {int(score*100):3d}% {status}")
        if summary["missing_critical"]:
            print(f"      ⚠  Missing critical fields: {summary['missing_critical']}")

    print("\n[Agent 1] Document Intelligence — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run([
        "data/patient_data/lab_report_ai.png",
        "data/patient_data/docotr_notes_ai.png",
        "data/patient_data/patient_ai.png",
        "data/patient_data/insurance_card_ai.png",
        "data/patient_data/medical_pretreatment_estimate_ai.pdf",
    ])
    display = {k: v for k, v in result.items() if k not in ("_raw", "per_document_extraction")}
    print(json.dumps(display, indent=2, default=str))
