"""
agent1_document_intelligence.py
────────────────────────────────
Agent 1 — Document Intelligence
• Accepts ANY number of medical documents with NO labelling required
• Supports: PNG, JPG/JPEG, WEBP, GIF images AND multi-page PDFs
• Converts all documents to base64 image blocks automatically
• Sends everything to Nova Pro in ONE multimodal call
• Nova Pro itself classifies what each document is (lab report, insurance
  card, doctor notes, etc.) — the caller does not need to pre-label files
• Returns a rich structured dict of all extracted medical entities
• Includes documents_present field for downstream doc checking

Accepted input formats (all equivalent):
  run(["file1.png", "file2.pdf", "file3.jpg"])          # plain list
  run({"a": "file1.png", "b": "file2.pdf"})             # labelled dict (legacy)
  run("single_file.pdf")                                 # single path string

Output : dict  (AGENT1_OUTPUT)
"""

import json
import base64
import io
from pathlib import Path
from typing import Union

import fitz                         # PyMuPDF — pip install pymupdf
from PIL import Image               # pip install Pillow

from bedrock_client import invoke, PRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Supported formats
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PDF_EXTENSIONS   = {".pdf"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — file → Nova content block(s)
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_block(image_path: str, max_size: tuple = (1600, 1600)) -> dict:
    """
    Load an image file, resize to max_size, base64-encode.
    Returns a single Nova image content block.
    """
    path = Path(image_path)
    fmt = path.suffix.lstrip(".").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in {"png", "jpeg", "webp", "gif"}:
        fmt = "png"  # Re-encode anything exotic as PNG

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
    """
    Rasterise every page of a PDF at `dpi` resolution.
    Returns a list of Nova image content blocks (one per page).
    """
    path = Path(pdf_path)
    doc  = fitz.open(str(path))
    blocks = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        print(f"  [Agent1] PDF    : {path.name}  page {i + 1}/{len(doc)}")
        blocks.append({"image": {"format": "png", "source": {"bytes": b64}}})
    doc.close()
    return blocks


def _file_to_blocks(file_path: str) -> list:
    """
    Dispatch a single file path to the correct encoder based on extension.
    Returns a list of Nova content blocks (images → 1 block, PDFs → N blocks).
    Raises FileNotFoundError or ValueError for bad inputs.
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if ext in IMAGE_EXTENSIONS:
        return [_image_to_block(file_path)]
    elif ext in PDF_EXTENSIONS:
        return _pdf_to_blocks(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}' for '{path.name}'. "
            f"Supported: {sorted(IMAGE_EXTENSIONS | PDF_EXTENSIONS)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Input normalisation — accept str / list / dict
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_input(documents: Union[str, list, dict]) -> list:
    """
    Accept any of:
      • str   — single file path
      • list  — ordered list of file paths (no labels needed)
      • dict  — legacy labelled dict  {"lab_report": "path", ...}

    Returns a flat, ordered list of file path strings.
    Dict keys are intentionally discarded — Nova Pro identifies document types.
    """
    if isinstance(documents, str):
        return [documents]
    elif isinstance(documents, list):
        return list(documents)
    elif isinstance(documents, dict):
        return list(documents.values())   # drop keys, keep values in insertion order
    else:
        raise TypeError(
            f"documents must be str, list, or dict — got {type(documents).__name__}"
        )


def _load_documents(file_paths: list) -> list:
    """
    Convert a list of file paths → flat list of Nova content blocks.
    Files are presented to the model in the order supplied.
    Bad files are skipped with a warning rather than crashing the pipeline.
    """
    all_blocks = []
    for fp in file_paths:
        try:
            blocks = _file_to_blocks(fp)
            all_blocks.extend(blocks)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [Agent1] ⚠  Skipping '{fp}': {exc}")
    return all_blocks


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a highly accurate medical document intelligence agent. "
            "Read the provided clinical documents and extract all fields required "
            "for insurance prior-authorization. "
            "The documents are NOT labelled — YOU must identify what each one is. "
            "Return ONLY a valid JSON object — no markdown, no preamble. "
            "Set missing fields to null. Dates in YYYY-MM-DD. "
            "Currency as numbers only. Collect ALL CPT and ICD-10 codes as arrays."
        )
    }
]

EXTRACTION_PROMPT = """You are given one or more medical document images (unlabelled).
Inspect every image carefully, identify what type of document each one is,
and extract ALL fields below across ALL documents.

IMPORTANT EXTRACTION RULES:
1. prior_treatments — each entry must be ONE complete treatment on a single line,
   combining the treatment name, duration, AND outcome together.
   CORRECT:   ["NSAIDs (Ibuprofen) - 4 weeks - Minimal relief",
               "Physical Therapy - 3 weeks - Symptoms persisted"]
   INCORRECT: ["NSAIDs (Ibuprofen) - 4 weeks", "Minimal relief", "Physical Therapy - 3 weeks"]
   Never split outcome phrases like "minimal relief" into their own list entry.

2. member_id vs policy_number — these are DIFFERENT fields. Read them independently.
   "Policy No" or "Policy Number" → policy_number field.
   "Member ID" or "Member #"      → member_id field.
   Never copy one into the other.

3. cpt_codes — collect ALL CPT codes found across ALL documents (doctor notes,
   pretreatment estimate, cost line items). Merge into one deduplicated array.

4. documents_identified — set a key to true ONLY if you can clearly read that
   document type among the images provided.

Return a single JSON object matching EXACTLY this schema
(null for any field not found):

{
  "patient": {
    "name": string,
    "dob": string,
    "age": integer,
    "gender": string,
    "mrn": string,
    "address": string,
    "city": string,
    "state": string,
    "zip": string,
    "phone": string,
    "email": string,
    "emergency_contact_name": string,
    "emergency_contact_phone": string,
    "emergency_contact_relationship": string
  },
  "insurance": {
    "insurer_name": string,
    "plan_type": string,
    "policy_number": string,
    "group_number": string,
    "member_id": string,
    "policyholder_name": string,
    "relationship_to_patient": string,
    "copay_office_visit": number,
    "copay_er": number,
    "copay_specialist": number,
    "administered_by": string,
    "underwritten_by": string,
    "bin": string,
    "pcn": string,
    "rxgrp": string,
    "customer_service_phone": string
  },
  "clinical": {
    "diagnosis": string,
    "icd10_codes": [string],
    "chief_complaint": string,
    "symptom_duration_weeks": integer,
    "pain_score": integer,
    "clinical_findings": string,
    "prior_treatments": [string],
    "requested_procedure": string,
    "cpt_codes": [string],
    "ordering_physician": string,
    "physician_phone": string,
    "physician_specialty": string
  },
  "facility": {
    "hospital_name": string,
    "hospital_address": string,
    "date_of_service": string,
    "date_of_proposed_treatment": string
  },
  "cost_estimate": {
    "line_items": [
      {"description": string, "cpt_code": string, "estimated_price": number}
    ],
    "total_estimated_cost": number
  },
  "lab_results": {
    "hba1c": string,
    "cholesterol_total": string,
    "ldl": string,
    "hdl": string,
    "triglycerides": string,
    "creatinine": string,
    "egfr": string,
    "alt": string,
    "ast": string
  },
  "pharmacy": {
    "pharmacy_name": string,
    "pharmacy_address": string,
    "pharmacy_phone": string
  },
  "prescribed_medications": [string],
  "documents_identified": {
    "lab_report": true | false,
    "doctor_notes": true | false,
    "patient_info": true | false,
    "insurance_card": true | false,
    "pretreatment_estimate": true | false,
    "prior_treatment_documentation": true | false,
    "procedure_order": true | false,
    "physician_referral": true | false
  },
  "extraction_confidence": "high" | "medium" | "low",
  "missing_fields": [string]
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _build_documents_present(data: dict) -> dict:
    """
    Build documents_present by combining:
    1. Nova Pro's own documents_identified assessment (from the images)
    2. Field-level verification (confirm extracted fields back up the doc claim)
    This dual-check avoids false positives — a doc must be both seen AND yield data.
    """
    nova_identified = data.get("documents_identified", {})
    lab             = data.get("lab_results", {})
    clin            = data.get("clinical", {})
    ins             = data.get("insurance", {})
    p               = data.get("patient", {})
    cost            = data.get("cost_estimate", {})

    return {
        "lab_report": (
            nova_identified.get("lab_report", False)
            and any(v for v in lab.values() if v)
        ),
        "doctor_notes": (
            nova_identified.get("doctor_notes", False)
            and bool(clin.get("diagnosis") or clin.get("clinical_findings"))
        ),
        "patient_info": (
            nova_identified.get("patient_info", False)
            and bool(p.get("name") and p.get("dob"))
        ),
        "insurance_card": (
            nova_identified.get("insurance_card", False)
            and bool(ins.get("policy_number") and ins.get("insurer_name"))
        ),
        "pretreatment_estimate": (
            nova_identified.get("pretreatment_estimate", False)
            and bool(cost.get("total_estimated_cost"))
        ),
        "prior_treatment_documentation": (
            nova_identified.get("prior_treatment_documentation", False)
            and bool(clin.get("prior_treatments") and len(clin.get("prior_treatments", [])) > 0)
        ),
        "procedure_order": (
            nova_identified.get("procedure_order", False)
            and bool(clin.get("requested_procedure") and clin.get("cpt_codes"))
        ),
        "physician_referral": nova_identified.get("physician_referral", False),
    }


def _build_documents_list(data: dict, docs_present: dict) -> list:
    """
    Build the structured `documents` list that Agents 2, 3, and 4 consume.
    Each entry represents one document type that was identified and verified,
    carrying the subset of extracted fields that came from that document.
    """
    p    = data.get("patient", {})
    ins  = data.get("insurance", {})
    clin = data.get("clinical", {})
    lab  = data.get("lab_results", {})
    cost = data.get("cost_estimate", {})
    fac  = data.get("facility", {})
    pharm = data.get("pharmacy", {})

    documents = []

    if docs_present.get("lab_report"):
        documents.append({
            "document_type": "Lab Report",
            "content": {k: v for k, v in lab.items() if v},
        })

    if docs_present.get("doctor_notes"):
        documents.append({
            "document_type": "Doctor Notes",
            "content": {
                "diagnosis":           clin.get("diagnosis"),
                "icd10_codes":         clin.get("icd10_codes"),
                "chief_complaint":     clin.get("chief_complaint"),
                "clinical_findings":   clin.get("clinical_findings"),
                "prior_treatments":    clin.get("prior_treatments"),
                "requested_procedure": clin.get("requested_procedure"),
                "cpt_codes":           clin.get("cpt_codes"),
                "symptom_duration_weeks": clin.get("symptom_duration_weeks"),
                "pain_score":          clin.get("pain_score"),
                "ordering_physician":  clin.get("ordering_physician"),
                "physician_specialty": clin.get("physician_specialty"),
                "physician_phone":     clin.get("physician_phone"),
            },
        })

    if docs_present.get("patient_info"):
        documents.append({
            "document_type": "Patient Information Sheet",
            "content": {
                "patient_name":    p.get("name"),
                "dob":             p.get("dob"),
                "gender":          p.get("gender"),
                "mrn":             p.get("mrn"),
                "address":         p.get("address"),
                "city":            p.get("city"),
                "state":           p.get("state"),
                "zip":             p.get("zip"),
                "phone":           p.get("phone"),
                "email":           p.get("email"),
                "emergency_contact": p.get("emergency_contact_name"),
            },
        })

    if docs_present.get("insurance_card"):
        documents.append({
            "document_type": "Insurance Card",
            "content": {
                "insurer_name":    ins.get("insurer_name"),
                "plan_type":       ins.get("plan_type"),
                "policy_number":   ins.get("policy_number"),
                "member_id":       ins.get("member_id"),
                "group_number":    ins.get("group_number"),
                "copay_specialist": ins.get("copay_specialist"),
                "copay_office_visit": ins.get("copay_office_visit"),
                "copay_er":        ins.get("copay_er"),
                "administered_by": ins.get("administered_by"),
                "underwritten_by": ins.get("underwritten_by"),
            },
        })

    if docs_present.get("pretreatment_estimate"):
        documents.append({
            "document_type": "Medical Pretreatment Estimate",
            "content": {
                "total_estimated_cost": cost.get("total_estimated_cost"),
                "line_items":           cost.get("line_items", []),
                "cpt_codes":            [i.get("cpt_code") for i in cost.get("line_items", []) if i.get("cpt_code")],
                "date_of_proposed_treatment": fac.get("date_of_proposed_treatment"),
                "hospital_name":        fac.get("hospital_name"),
            },
        })

    if docs_present.get("prior_treatment_documentation"):
        # Already covered by doctor_notes; add only if it's a separate doc
        if not docs_present.get("doctor_notes"):
            documents.append({
                "document_type": "Prior Treatment Documentation",
                "content": {
                    "prior_treatments": clin.get("prior_treatments", []),
                },
            })

    if pharm.get("pharmacy_name"):
        documents.append({
            "document_type": "Pharmacy Information",
            "content": {k: v for k, v in pharm.items() if v},
        })

    return documents


def _build_handoff(data: dict) -> dict:
    """Flatten rich extraction → compact downstream format."""
    p    = data.get("patient", {})
    ins  = data.get("insurance", {})
    clin = data.get("clinical", {})
    fac  = data.get("facility", {})
    cost = data.get("cost_estimate", {})
    docs = _build_documents_present(data)

    # Merge CPT codes from cost line items into the clinical cpt_codes list
    # so all CPT codes from all documents are captured in one place
    line_item_cpts = [
        item.get("cpt_code")
        for item in cost.get("line_items", [])
        if item.get("cpt_code") and item.get("cpt_code") != "N/A"
    ]
    clinical_cpts = clin.get("cpt_codes") or []
    all_cpts = list(dict.fromkeys(clinical_cpts + line_item_cpts))  # deduplicate, preserve order

    return {
        "patient_name":               p.get("name"),
        "patient_dob":                p.get("dob"),
        "patient_mrn":                p.get("mrn"),
        "patient_gender":             p.get("gender"),
        "patient_address":            f"{p.get('address')}, {p.get('city')}, {p.get('state')} {p.get('zip')}",
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
        "cpt_codes":                  all_cpts,
        "cpt":                        (all_cpts or [None])[0],
        "ordering_physician":         clin.get("ordering_physician"),
        "physician_specialty":        clin.get("physician_specialty"),
        "physician_phone":            clin.get("physician_phone"),
        "prior_treatments":           clin.get("prior_treatments", []),
        "clinical_findings":          clin.get("clinical_findings"),
        "symptom_duration_weeks":     clin.get("symptom_duration_weeks"),
        "pain_score":                 clin.get("pain_score"),
        "hospital":                   fac.get("hospital_name"),
        "date_of_service":            fac.get("date_of_service"),
        "date_of_proposed_treatment": fac.get("date_of_proposed_treatment"),
        "total_estimated_cost":       cost.get("total_estimated_cost"),
        "cost_line_items":            cost.get("line_items", []),
        "prescribed_medications":     data.get("prescribed_medications", []),
        "lab_results":                data.get("lab_results", {}),
        "pharmacy":                   data.get("pharmacy", {}),
        "documents_present":          docs,
        "documents":                  _build_documents_list(data, docs),
        "extraction_confidence":      data.get("extraction_confidence"),
        "missing_fields":             data.get("missing_fields", []),
        "_raw":                       data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(documents: Union[str, list, dict]) -> dict:
    """
    Agent 1 entry point — Document Intelligence.

    Parameters
    ----------
    documents : str | list | dict
        Any of the following are accepted — NO labels required:

        1. A plain list of file paths (recommended):
               run([
                   "lab_report_ai.png",
                   "docotr_notes_ai.png",
                   "patient_ai.png",
                   "insurance_card_ai.png",
                   "medical_pretreatment_estimate_ai.pdf",
                   "extra_referral.png",   # ← extra docs work automatically
               ])

        2. A single file path (str):
               run("lab_report.png")

        3. A labelled dict (legacy / backward-compatible):
               run({
                   "lab_report":            "lab_report_ai.png",
                   "pretreatment_estimate": "medical_pretreatment_estimate_ai.pdf",
               })
           Dict keys are IGNORED — Nova Pro identifies document types itself.

        Supported file types : PNG, JPG/JPEG, WEBP, GIF (images), PDF.
        Multi-page PDFs are rasterised page-by-page automatically.
        Files that cannot be loaded are skipped with a warning.

    Returns
    -------
    dict — structured extraction + compact handoff fields ready for Agent 2.
           Key fields: patient_name, insurer, diagnosis, icd10_codes, cpt_codes,
                       documents_present, extraction_confidence, missing_fields, _raw.
    """
    print("\n[Agent 1] Document Intelligence — START")

    # ── 1. Normalise input → flat list of file paths ───────────────────────
    file_paths = _normalise_input(documents)
    print(f"[Agent 1] Input files ({len(file_paths)}):")
    for fp in file_paths:
        print(f"  • {fp}")

    # ── 2. Convert every file → Nova image content blocks ─────────────────
    image_blocks = _load_documents(file_paths)
    print(f"[Agent 1] Total content blocks (images/pages): {len(image_blocks)}")

    if not image_blocks:
        raise ValueError("[Agent 1] No valid documents could be loaded. Aborting.")

    # ── 3. Build multimodal message: all images first, then the prompt ─────
    messages = [
        {
            "role": "user",
            "content": image_blocks + [{"text": EXTRACTION_PROMPT}],
        }
    ]

    # ── 4. Single Nova Pro call — classify + extract everything at once ────
    print("[Agent 1] Calling Nova Pro...")
    raw = invoke(
        model_id=PRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.1,
    )

    # ── 5. Parse response and build handoff dict ───────────────────────────
    extracted = _parse_json(raw)
    output    = _build_handoff(extracted)

    print("[Agent 1] Extraction complete.")
    print(f"  Confidence : {output.get('extraction_confidence')}")
    print(f"  Missing    : {output.get('missing_fields')}")

    docs = output.get("documents_present", {})
    print("  Documents identified:")
    for doc, present in docs.items():
        status = "FOUND    " if present else "NOT FOUND"
        print(f"    [{status}] {doc}")

    print("[Agent 1] Document Intelligence — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — three equivalent call styles
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Style 1 — plain list, no labels (recommended)
    result = run([
        "lab_report_ai.png",
        "docotr_notes_ai.png",
        "patient_ai.png",
        "insurance_card_ai.png",
        "medical_pretreatment_estimate_ai.pdf",
    ])

    # Style 2 — single file
    # result = run("lab_report_ai.png")

    # Style 3 — legacy labelled dict (backward-compatible)
    # result = run({
    #     "lab_report":            "lab_report_ai.png",
    #     "doctor_notes":          "docotr_notes_ai.png",
    #     "patient_info":          "patient_ai.png",
    #     "insurance_card":        "insurance_card_ai.png",
    #     "pretreatment_estimate": "medical_pretreatment_estimate_ai.pdf",
    # })

    display = {k: v for k, v in result.items() if k != "_raw"}
    print(json.dumps(display, indent=2))
