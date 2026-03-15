"""
agent2_policy_checker.py
─────────────────────────
Agent 2 — Policy Query Requirement Checker
• Validates that all fields required for a prior-auth query are present
• Works from Agent 1's document list + merged fields
• Checks: insurer, ICD-10, CPT, patient ID, physician, facility
• Flags missing or malformed fields and traces which document they'd normally come from
• Uses Nova Micro (text-only, fast, cheap)

Input  : dict from Agent 1
Output : dict  { "ready": bool, "missing": [...], "warnings": [...], "data": {...} }
"""

import json
import re

from bedrock_client import invoke, MICRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Validation rules
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "patient_name",
    "patient_dob",
    "patient_mrn",
    "insurer",
    "policy_number",
    "diagnosis",
    "icd10",
    "procedure",
    "cpt",
    "ordering_physician",
    "hospital",
    "date_of_proposed_treatment",
]

RECOMMENDED_FIELDS = [
    "group_number",
    "member_id",
    "physician_specialty",
    "physician_phone",
    "prior_treatments",
    "symptom_duration_weeks",
    "total_estimated_cost",
]

# Maps a required field to the document type that typically contains it
FIELD_TO_DOCUMENT = {
    "patient_name":               "Patient Information Sheet",
    "patient_dob":                "Patient Information Sheet",
    "patient_mrn":                "Patient Information Sheet / Lab Report",
    "insurer":                    "Insurance Card / Patient Information Sheet",
    "policy_number":              "Insurance Card",
    "diagnosis":                  "Doctor Notes / Pretreatment Estimate",
    "icd10":                      "Doctor Notes / Pretreatment Estimate",
    "procedure":                  "Doctor Notes / Pretreatment Estimate",
    "cpt":                        "Pretreatment Estimate / Doctor Notes",
    "ordering_physician":         "Doctor Notes / Pretreatment Estimate",
    "hospital":                   "Doctor Notes / Pretreatment Estimate",
    "date_of_proposed_treatment": "Pretreatment Estimate / Doctor Notes",
    "group_number":               "Insurance Card",
    "member_id":                  "Insurance Card",
    "physician_specialty":        "Doctor Notes / Pretreatment Estimate",
    "physician_phone":            "Pretreatment Estimate",
    "prior_treatments":           "Doctor Notes / Pretreatment Estimate",
    "symptom_duration_weeks":     "Doctor Notes",
    "total_estimated_cost":       "Pretreatment Estimate",
}

ICD10_PATTERN = re.compile(r"^[A-Z]\d{2}(\.\d+)?$", re.IGNORECASE)
CPT_PATTERN   = re.compile(r"^\d{5}(-\d+)?$")


# ─────────────────────────────────────────────────────────────────────────────
# Local validation
# ─────────────────────────────────────────────────────────────────────────────

def _local_validate(data: dict) -> tuple:
    missing_required    = []
    missing_recommended = []
    format_warnings     = []

    for field in REQUIRED_FIELDS:
        val = data.get(field)
        if not val or (isinstance(val, list) and len(val) == 0):
            missing_required.append({
                "field":             field,
                "suggested_document": FIELD_TO_DOCUMENT.get(field, "Unknown"),
            })

    for field in RECOMMENDED_FIELDS:
        val = data.get(field)
        if not val or (isinstance(val, list) and len(val) == 0):
            missing_recommended.append({
                "field":             field,
                "suggested_document": FIELD_TO_DOCUMENT.get(field, "Unknown"),
            })

    icd10 = data.get("icd10", "")
    if icd10 and not ICD10_PATTERN.match(str(icd10)):
        format_warnings.append(f"ICD-10 code '{icd10}' may be malformed")

    for cpt in data.get("cpt_codes", []):
        if not CPT_PATTERN.match(str(cpt)):
            format_warnings.append(f"CPT code '{cpt}' may be malformed")

    dob = data.get("patient_dob", "")
    if dob and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(dob)):
        format_warnings.append(f"Date of birth '{dob}' is not YYYY-MM-DD")

    return missing_required, missing_recommended, format_warnings


def _check_documents(data: dict) -> list:
    """
    Check documents_present from Agent 1.
    Returns list of warning dicts for missing documents.
    """
    docs     = data.get("documents_present", {})
    warnings = []
    for doc, present in docs.items():
        if not present:
            warnings.append({
                "document": doc,
                "warning":  f"Document not found: {doc}",
            })
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# LLM sanity check — now includes document list context
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a medical prior-authorization intake specialist. "
            "Review the extracted patient data and flag any inconsistencies, "
            "suspicious values, or fields that look incorrect. "
            "Return ONLY a JSON object — no markdown, no preamble."
        )
    }
]


def _llm_sanity_check(data: dict) -> dict:
    # Build a summary of documents found and their key content
    doc_summary = []
    for doc in data.get("documents", []):
        doc_summary.append({
            "document_type": doc.get("document_type"),
            "fields_present": list(doc.get("content", {}).keys()),
        })

    review_subset = {
        "patient_name":        data.get("patient_name"),
        "patient_dob":         data.get("patient_dob"),
        "insurer":             data.get("insurer"),
        "diagnosis":           data.get("diagnosis"),
        "icd10_codes":         data.get("icd10_codes"),
        "procedure":           data.get("procedure"),
        "cpt_codes":           data.get("cpt_codes"),
        "ordering_physician":  data.get("ordering_physician"),
        "physician_specialty": data.get("physician_specialty"),
        "hospital":            data.get("hospital"),
        "prior_treatments":    data.get("prior_treatments"),
        "total_estimated_cost": data.get("total_estimated_cost"),
        "documents_found":     doc_summary,
    }

    prompt = f"""Review the following extracted prior-authorization data for quality.

Documents extracted:
{json.dumps(doc_summary, indent=2)}

Merged data:
{json.dumps({k: v for k, v in review_subset.items() if k != 'documents_found'}, indent=2)}

Check:
1. Do the ICD-10 codes match the stated diagnosis?
2. Do the CPT codes match the requested procedure?
3. Are the physician specialty and procedure consistent?
4. Are there any obviously wrong or suspicious values?
5. Are there critical documents missing that would be expected for a prior-auth?
6. Is the data consistent across the different documents?

Return JSON:
{{
  "issues": [string],
  "icd10_procedure_match": true | false,
  "cpt_procedure_match": true | false,
  "cross_document_consistency": true | false,
  "overall_quality": "good" | "fair" | "poor",
  "recommendation": string
}}"""

    messages = [{"role": "user", "content": [{"text": prompt}]}]
    raw = invoke(
        model_id=MICRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=500,
        temperature=0.1,
    )

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"issues": ["LLM sanity check parse error"], "overall_quality": "fair"}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent1_output: dict) -> dict:
    print("\n[Agent 2] Policy Query Requirement Checker — START")

    missing_req, missing_rec, fmt_warns = _local_validate(agent1_output)
    doc_warnings = _check_documents(agent1_output)

    if missing_req:
        print("  Missing required fields:")
        for m in missing_req:
            print(f"    - {m['field']} → expected in: {m['suggested_document']}")
    else:
        print("  Missing required: None")

    if doc_warnings:
        print(f"  Missing documents: {[w['document'] for w in doc_warnings]}")

    print(f"  Format warnings : {fmt_warns or 'None'}")

    print("[Agent 2] Running LLM sanity check via Nova Micro...")
    llm_review = _llm_sanity_check(agent1_output)
    print(f"  LLM overall quality : {llm_review.get('overall_quality')}")
    print(f"  LLM issues          : {llm_review.get('issues')}")

    ready = (
        len(missing_req) == 0
        and llm_review.get("overall_quality", "fair") != "poor"
    )

    output = {
        "ready":                ready,
        "missing_required":     missing_req,      # list of {field, suggested_document}
        "missing_recommended":  missing_rec,      # list of {field, suggested_document}
        "format_warnings":      fmt_warns,
        "doc_warnings":         doc_warnings,     # list of {document, warning}
        "llm_review":           llm_review,
        "data":                 agent1_output,
    }

    print(f"[Agent 2] Ready for pipeline: {ready}")
    print("[Agent 2] Policy Query Requirement Checker — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "patient_name": "John R. Doe",
        "patient_dob": "1985-05-12",
        "patient_mrn": "PT-88293",
        "insurer": "BlueCross BlueShield",
        "policy_number": "BCBS-99001122",
        "diagnosis": "Lumbar Radiculopathy",
        "icd10": "M54.16",
        "icd10_codes": ["M54.16"],
        "procedure": "MRI Lumbar Spine Without Contrast",
        "cpt": "72148",
        "cpt_codes": ["72148"],
        "ordering_physician": "Dr. Sarah Jenkins",
        "physician_specialty": "Orthopedic Surgery",
        "hospital": "Metropolitan General Hospital",
        "date_of_proposed_treatment": "2024-03-15",
        "total_estimated_cost": 2025,
        "prior_treatments": ["NSAIDs 4 weeks", "PT 3 weeks"],
        "documents_present": {
            "lab_report": True,
            "doctor_notes": True,
            "patient_info": True,
            "insurance_card": True,
            "pretreatment_estimate": True,
            "prior_treatment_documentation": True,
            "procedure_order": True,
            "physician_referral": False,
        },
        "documents": [
            {"document_type": "Lab Report", "content": {"patient_name": "John R. Doe", "hba1c": "5.6%"}},
            {"document_type": "Doctor Notes", "content": {"diagnosis": "Lumbar Radiculopathy", "icd10_codes": ["M54.16"]}},
        ],
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
