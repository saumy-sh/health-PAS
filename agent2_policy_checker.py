"""
agent2_policy_checker.py
─────────────────────────
Agent 2 — Policy Query Requirement Checker
• Validates that all fields required for a prior-auth query are present
• Checks: insurer, ICD-10, CPT, patient ID, physician, facility
• Flags missing or malformed fields before hitting the policy RAG
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

# Fields that MUST be present for a prior-auth request
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

# Fields that are important but won't block the pipeline
RECOMMENDED_FIELDS = [
    "group_number",
    "member_id",
    "physician_specialty",
    "physician_phone",
    "prior_treatments",
    "symptom_duration_weeks",
    "total_estimated_cost",
]

# ICD-10 pattern: letter + 2 digits + optional dot + optional alphanumeric
ICD10_PATTERN = re.compile(r"^[A-Z]\d{2}(\.\d+)?$", re.IGNORECASE)

# CPT pattern: 5 digits, optionally followed by modifier (e.g. 72148-26)
CPT_PATTERN = re.compile(r"^\d{5}(-\d+)?$")


# ─────────────────────────────────────────────────────────────────────────────
# Local validation (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

def _local_validate(data: dict) -> tuple:
    """
    Pure-Python field presence + format checks.
    Returns (missing_required, missing_recommended, format_warnings).
    """
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

    # Format checks
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


# ─────────────────────────────────────────────────────────────────────────────
# LLM sanity check — Nova Micro reviews the extracted fields for coherence
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
    """
    Ask Nova Micro to review the data for logical inconsistencies.
    Returns dict: { "issues": [...], "overall_quality": "good|fair|poor" }
    """
    # Only send the key clinical + insurance fields (keep token count low)
    review_subset = {
        "patient_name":     data.get("patient_name"),
        "patient_dob":      data.get("patient_dob"),
        "insurer":          data.get("insurer"),
        "diagnosis":        data.get("diagnosis"),
        "icd10_codes":      data.get("icd10_codes"),
        "procedure":        data.get("procedure"),
        "cpt_codes":        data.get("cpt_codes"),
        "ordering_physician": data.get("ordering_physician"),
        "physician_specialty": data.get("physician_specialty"),
        "hospital":         data.get("hospital"),
        "prior_treatments": data.get("prior_treatments"),
        "total_estimated_cost": data.get("total_estimated_cost"),
    }

    prompt = f"""Review the following extracted prior-authorization data for quality.
Check:
1. Do the ICD-10 codes match the stated diagnosis?
2. Do the CPT codes match the requested procedure?
3. Are the physician specialty and procedure consistent?
4. Are there any obviously wrong or suspicious values?

Data:
{json.dumps(review_subset, indent=2)}

Return JSON:
{{
  "issues": [string],
  "icd10_procedure_match": true | false,
  "cpt_procedure_match": true | false,
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
    """
    Agent 2 entry point.

    Parameters
    ----------
    agent1_output : dict — output from Agent 1

    Returns
    -------
    dict {
        "ready"       : bool   — True if all required fields present + quality good,
        "missing_required"    : list of missing required field names,
        "missing_recommended" : list of missing recommended field names,
        "format_warnings"     : list of format issue strings,
        "llm_review"  : dict   — Nova Micro sanity check result,
        "data"        : dict   — original agent1_output passed through
    }
    """
    print("\n[Agent 2] Policy Query Requirement Checker — START")

    missing_req, missing_rec, fmt_warns = _local_validate(agent1_output)

    print(f"  Missing required    : {missing_req or 'None'}")
    print(f"  Missing recommended : {missing_rec or 'None'}")
    print(f"  Format warnings     : {fmt_warns or 'None'}")

    print("[Agent 2] Running LLM sanity check via Nova Micro...")
    llm_review = _llm_sanity_check(agent1_output)
    print(f"  LLM overall quality : {llm_review.get('overall_quality')}")
    print(f"  LLM issues          : {llm_review.get('issues')}")

    # Ready if: no missing required fields AND quality is not 'poor'
    ready = (
        len(missing_req) == 0
        and llm_review.get("overall_quality", "fair") != "poor"
    )

    output = {
        "ready":                ready,
        "missing_required":     missing_req,
        "missing_recommended":  missing_rec,
        "format_warnings":      fmt_warns,
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
    # Mock Agent 1 output for testing
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
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
