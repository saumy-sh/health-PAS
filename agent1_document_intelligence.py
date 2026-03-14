"""
agent1_document_intelligence.py
────────────────────────────────
Agent 1 — Document Intelligence
• Accepts paths to medical documents (PNG images + PDF)
• Converts all documents to base64 image blocks
• Sends everything to Nova Pro in ONE multimodal call
• Returns a rich structured dict of all extracted medical entities

Input  : dict with file paths
Output : dict  (AGENT1_OUTPUT)
"""

import json
import base64
import io
from pathlib import Path

import fitz                         # PyMuPDF — pip install pymupdf
from PIL import Image               # pip install Pillow

from bedrock_client import invoke, PRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_block(image_path: str, max_size: tuple = (1600, 1600)) -> dict:
    """Load an image, resize, base64-encode → Nova image content block."""
    path = Path(image_path)
    fmt = path.suffix.lstrip(".").lower()
    if fmt == "jpg":
        fmt = "jpeg"

    img = Image.open(path)
    img.thumbnail(max_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format=fmt.upper())
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    print(f"  [Agent1] Loaded image  : {path.name}  ({img.size[0]}×{img.size[1]}px)")
    return {"image": {"format": fmt, "source": {"bytes": b64}}}


def _pdf_to_blocks(pdf_path: str, dpi: int = 150) -> list:
    """Convert each PDF page to a PNG → list of Nova image content blocks."""
    doc = fitz.open(pdf_path)
    blocks = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        print(f"  [Agent1] Loaded PDF page: {Path(pdf_path).name} — page {i+1}")
        blocks.append({"image": {"format": "png", "source": {"bytes": b64}}})
    return blocks


def _load_documents(doc_paths: dict) -> list:
    """
    Build the full list of image content blocks from all input documents.
    doc_paths keys: lab_report, doctor_notes, patient_info,
                    insurance_card, pretreatment_estimate
    """
    blocks = []
    for key in ["lab_report", "doctor_notes", "patient_info", "insurance_card"]:
        if key in doc_paths:
            blocks.append(_image_to_block(doc_paths[key]))

    if "pretreatment_estimate" in doc_paths:
        blocks.extend(_pdf_to_blocks(doc_paths["pretreatment_estimate"]))

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a highly accurate medical document intelligence agent. "
            "Read the provided clinical documents and extract all fields required "
            "for insurance prior-authorization. "
            "Return ONLY a valid JSON object — no markdown, no preamble. "
            "Set missing fields to null. Dates in YYYY-MM-DD. "
            "Currency as numbers only. Collect ALL CPT and ICD-10 codes as arrays."
        )
    }
]

EXTRACTION_PROMPT = """You are given multiple medical document images.
Read every document carefully and extract ALL fields below.

Return a single JSON object matching EXACTLY this schema:

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
  "extraction_confidence": "high" | "medium" | "low",
  "missing_fields": [string]
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
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


def _build_handoff(data: dict) -> dict:
    """Flatten rich extraction → compact downstream format."""
    p = data.get("patient", {})
    ins = data.get("insurance", {})
    clin = data.get("clinical", {})
    fac = data.get("facility", {})
    cost = data.get("cost_estimate", {})

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
        "cpt_codes":                  clin.get("cpt_codes", []),
        "cpt":                        (clin.get("cpt_codes") or [None])[0],
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
        "extraction_confidence":      data.get("extraction_confidence"),
        "missing_fields":             data.get("missing_fields", []),
        "_raw":                       data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(doc_paths: dict) -> dict:
    """
    Agent 1 entry point.

    Parameters
    ----------
    doc_paths : dict
        {
            "lab_report":            "path/to/lab_report_ai.png",
            "doctor_notes":          "path/to/docotr_notes_ai.png",
            "patient_info":          "path/to/patient_ai.png",
            "insurance_card":        "path/to/insurance_card_ai.png",
            "pretreatment_estimate": "path/to/medical_pretreatment_estimate_ai.pdf"
        }

    Returns
    -------
    dict  — structured extraction + compact handoff fields
    """
    print("\n[Agent 1] Document Intelligence — START")

    image_blocks = _load_documents(doc_paths)
    print(f"[Agent 1] Total image blocks: {len(image_blocks)}")

    # Build message: all images first, then the extraction prompt
    messages = [
        {
            "role": "user",
            "content": image_blocks + [{"text": EXTRACTION_PROMPT}],
        }
    ]

    print("[Agent 1] Calling Nova Pro...")
    raw = invoke(
        model_id=PRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.1,
    )

    extracted = _parse_json(raw)
    output = _build_handoff(extracted)

    print("[Agent 1] Extraction complete.")
    print(f"  Confidence : {output.get('extraction_confidence')}")
    print(f"  Missing    : {output.get('missing_fields')}")
    print("[Agent 1] Document Intelligence — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DOC_PATHS = {
        "lab_report":            "lab_report_ai.png",
        "doctor_notes":          "docotr_notes_ai.png",
        "patient_info":          "patient_ai.png",
        "insurance_card":        "insurance_card_ai.png",
        "pretreatment_estimate": "medical_pretreatment_estimate_ai.pdf",
    }
    result = run(DOC_PATHS)
    # Print without _raw for readability
    display = {k: v for k, v in result.items() if k != "_raw"}
    print(json.dumps(display, indent=2))
