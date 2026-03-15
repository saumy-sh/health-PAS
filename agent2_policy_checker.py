"""
agent2_policy_checker.py
─────────────────────────
Agent 2 — Policy Query Requirement Checker
• Validates that all fields required for a prior-auth query are present
• Checks documents_present from Agent 1 and warns early on missing docs
• Checks: insurer, ICD-10, CPT, patient ID, physician, facility
• Flags missing or malformed fields before hitting policy retrieval
• Validates urgency flag extracted by Agent 1 (YAML: urgency_detection stage)
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
    "urgency",
]

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
            missing_required.append(field)

    for field in RECOMMENDED_FIELDS:
        val = data.get(field)
        if not val or (isinstance(val, list) and len(val) == 0):
            missing_recommended.append(field)

    icd10 = data.get("icd10", "")
    if icd10 and not ICD10_PATTERN.match(str(icd10)):
        format_warnings.append(f"ICD-10 code '{icd10}' may be malformed")

    for cpt in data.get("cpt_codes", []):
        if not CPT_PATTERN.match(str(cpt)):
            format_warnings.append(f"CPT code '{cpt}' may be malformed")

    dob = data.get("patient_dob", "")
    if dob and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(dob)):
        format_warnings.append(f"Date of birth '{dob}' is not YYYY-MM-DD")

    # Urgency validation (YAML: urgency_detection stage)
    urgency = data.get("urgency", "routine")
    valid_urgencies = {"routine", "urgent", "emergent"}
    if urgency not in valid_urgencies:
        format_warnings.append(
            f"Urgency value '{urgency}' is not recognized; defaulting to 'routine'"
        )

    return missing_required, missing_recommended, format_warnings


def _check_documents(data: dict) -> list:
    """
    Check documents_present from Agent 1.
    Returns a list of warning strings for any documents not found.
    """
    docs = data.get("documents_present", {})
    if not docs:
        return []

    warnings = []
    for doc, present in docs.items():
        if not present:
            warnings.append(f"document_missing:{doc}")
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# LLM sanity check
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a medical prior-authorization intake specialist. "
            "Review the extracted patient data and flag any inconsistencies, "
            "suspicious values, or fields that look incorrect. "
            "Also validate whether the urgency level is clinically justified. "
            "Return ONLY a JSON object — no markdown, no preamble."
        )
    }
]


def _llm_sanity_check(data: dict) -> dict:
    review_subset = {
        "patient_name":         data.get("patient_name"),
        "patient_dob":          data.get("patient_dob"),
        "insurer":              data.get("insurer"),
        "diagnosis":            data.get("diagnosis"),
        "icd10_codes":          data.get("icd10_codes"),
        "procedure":            data.get("procedure"),
        "cpt_codes":            data.get("cpt_codes"),
        "ordering_physician":   data.get("ordering_physician"),
        "physician_specialty":  data.get("physician_specialty"),
        "hospital":             data.get("hospital"),
        "prior_treatments":     data.get("prior_treatments"),
        "total_estimated_cost": data.get("total_estimated_cost"),
        "documents_present":    data.get("documents_present", {}),
        # YAML urgency_detection output
        "urgency":              data.get("urgency", "routine"),
        "urgency_justification": data.get("urgency_justification"),
    }

    prompt = f"""Review the following extracted prior-authorization data for quality.
Check:
1. Do the ICD-10 codes match the stated diagnosis?
2. Do the CPT codes match the requested procedure?
3. Are the physician specialty and procedure consistent?
4. Are there any obviously wrong or suspicious values?
5. Are there any critical documents missing from documents_present?
6. Is the urgency level ('routine' / 'urgent' / 'emergent') clinically justified
   given the diagnosis, symptom duration, and pain score?

Data:
{json.dumps(review_subset, indent=2)}

Return JSON:
{{
  "issues": [string],
  "icd10_procedure_match": true | false,
  "cpt_procedure_match": true | false,
  "urgency_justified": true | false,
  "urgency_notes": string,
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

    # Check documents_present from Agent 1
    doc_warnings = _check_documents(agent1_output)
    if doc_warnings:
        missing_docs_found = [w.replace("document_missing:", "") for w in doc_warnings]
        print(f"  Documents not found : {missing_docs_found}")
        missing_rec.extend(doc_warnings)

    print(f"  Missing required    : {missing_req or 'None'}")
    print(f"  Missing recommended : {missing_rec or 'None'}")
    print(f"  Format warnings     : {fmt_warns or 'None'}")
    print(f"  Urgency level       : {agent1_output.get('urgency', 'routine')}")

    print("[Agent 2] Running LLM sanity check via Nova Micro...")
    llm_review = _llm_sanity_check(agent1_output)
    print(f"  LLM overall quality : {llm_review.get('overall_quality')}")
    print(f"  LLM issues          : {llm_review.get('issues')}")
    print(f"  Urgency justified   : {llm_review.get('urgency_justified')} — {llm_review.get('urgency_notes')}")

    ready = (
        len(missing_req) == 0
        and llm_review.get("overall_quality", "fair") != "poor"
    )

    output = {
        "ready":                ready,
        "missing_required":     missing_req,
        "missing_recommended":  missing_rec,
        "format_warnings":      fmt_warns,
        "doc_warnings":         doc_warnings,
        "llm_review":           llm_review,
        # Pass urgency forward for Agent 6 form-filling
        "urgency":              agent1_output.get("urgency", "routine"),
        "urgency_justification": agent1_output.get("urgency_justification"),
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
        "group_number": "8800221",
        "member_id": "BCBS-9900122",
        "diagnosis": "Lumbar Radiculopathy",
        "icd10": "M54.16",
        "icd10_codes": ["M54.16"],
        "procedure": "MRI Lumbar Spine Without Contrast",
        "cpt": "72148",
        "cpt_codes": ["72148", "72148-26", "99204", "97161"],
        "ordering_physician": "Dr. Sarah Jenkins",
        "physician_specialty": "Orthopedic Surgery",
        "physician_phone": "555-010-8899",
        "hospital": "Metropolitan General Hospital",
        "date_of_proposed_treatment": "2024-03-15",
        "total_estimated_cost": 2025,
        "prior_treatments": ["NSAIDs - 4 weeks - minimal relief", "Physical Therapy - 3 weeks"],
        "symptom_duration_weeks": 7,
        "urgency": "routine",
        "urgency_justification": "No emergent indicators; standard imaging pre-authorization.",
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
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
