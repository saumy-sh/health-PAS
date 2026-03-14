"""
agent6_form_generator.py
─────────────────────────
Agent 6 — Prior Authorization Form Generator
• Takes eligibility determination + patient data
• Generates a complete, ready-to-submit prior authorization form
• Outputs both structured JSON and a formatted text document
• Uses Nova Lite

Input  : dict from Agent 5
Output : dict  { "form_json": {...}, "form_text": str, "form_file": str }
"""

import json
from datetime import datetime
from pathlib import Path

from bedrock_client import invoke, LITE_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a medical prior-authorization form specialist. "
            "Generate complete, accurate prior-authorization request forms "
            "using all provided patient, clinical, and insurance data. "
            "Forms must be thorough, clinically precise, and persuasive. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_prompt(data: dict, eligibility: dict) -> str:
    return f"""Generate a complete prior authorization request form using the data below.
The form should be filled in completely — do not leave any field blank if the data is available.

PATIENT DATA:
{json.dumps(data, indent=2, default=str)}

ELIGIBILITY DETERMINATION:
- Decision     : {eligibility.get('determination')}
- Criteria Met : {eligibility.get('criteria_met_count')}/{eligibility.get('criteria_total')}
- Clinical Summary: {eligibility.get('clinical_summary')}
- Recommendation: {eligibility.get('recommendation')}

Return a JSON object representing the completed prior authorization form:
{{
  "form_metadata": {{
    "form_type": "Prior Authorization Request",
    "submission_date": string,
    "reference_number": string,
    "urgency": "routine" | "urgent"
  }},
  "patient_information": {{
    "name": string,
    "date_of_birth": string,
    "gender": string,
    "member_id": string,
    "mrn": string,
    "address": string,
    "phone": string,
    "email": string
  }},
  "insurance_information": {{
    "insurer_name": string,
    "plan_type": string,
    "policy_number": string,
    "group_number": string,
    "policyholder_name": string
  }},
  "requesting_provider": {{
    "name": string,
    "specialty": string,
    "phone": string,
    "facility": string,
    "facility_address": string,
    "npi": string
  }},
  "clinical_information": {{
    "primary_diagnosis": string,
    "icd10_codes": [string],
    "requested_procedure": string,
    "cpt_codes": [string],
    "date_of_service": string,
    "place_of_service": string,
    "clinical_indication": string,
    "symptom_duration": string,
    "clinical_findings": string
  }},
  "medical_necessity_justification": {{
    "statement": string,
    "prior_treatments_failed": [string],
    "conservative_therapy_duration": string,
    "why_procedure_necessary": string,
    "supporting_clinical_evidence": string
  }},
  "cost_information": {{
    "estimated_total_cost": number,
    "line_items": [
      {{"description": string, "cpt_code": string, "estimated_cost": number}}
    ]
  }},
  "attestation": {{
    "statement": string,
    "physician_name": string,
    "physician_signature_placeholder": "[ Physician Signature Required ]",
    "date": string
  }}
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Form renderer — converts JSON form to readable text document
# ─────────────────────────────────────────────────────────────────────────────

def _render_form_text(form: dict) -> str:
    """Render the JSON form as a formatted plain-text document."""
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("            PRIOR AUTHORIZATION REQUEST FORM")
    lines.append(sep)

    meta = form.get("form_metadata", {})
    lines.append(f"  Form Type        : {meta.get('form_type')}")
    lines.append(f"  Reference Number : {meta.get('reference_number')}")
    lines.append(f"  Submission Date  : {meta.get('submission_date')}")
    lines.append(f"  Urgency          : {meta.get('urgency', 'routine').upper()}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("SECTION 1 — PATIENT INFORMATION")
    lines.append("-" * 70)
    pt = form.get("patient_information", {})
    for k, v in pt.items():
        lines.append(f"  {k.replace('_', ' ').title():<25}: {v}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("SECTION 2 — INSURANCE INFORMATION")
    lines.append("-" * 70)
    ins = form.get("insurance_information", {})
    for k, v in ins.items():
        lines.append(f"  {k.replace('_', ' ').title():<25}: {v}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("SECTION 3 — REQUESTING PROVIDER")
    lines.append("-" * 70)
    prov = form.get("requesting_provider", {})
    for k, v in prov.items():
        lines.append(f"  {k.replace('_', ' ').title():<25}: {v}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("SECTION 4 — CLINICAL INFORMATION")
    lines.append("-" * 70)
    clin = form.get("clinical_information", {})
    for k, v in clin.items():
        if isinstance(v, list):
            lines.append(f"  {k.replace('_', ' ').title():<25}: {', '.join(v)}")
        else:
            lines.append(f"  {k.replace('_', ' ').title():<25}: {v}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("SECTION 5 — MEDICAL NECESSITY JUSTIFICATION")
    lines.append("-" * 70)
    mnj = form.get("medical_necessity_justification", {})
    lines.append(f"  {mnj.get('statement', '')}")
    lines.append("")
    lines.append("  Prior Treatments Failed:")
    for tx in mnj.get("prior_treatments_failed", []):
        lines.append(f"    • {tx}")
    lines.append(f"\n  Conservative Therapy Duration: {mnj.get('conservative_therapy_duration')}")
    lines.append(f"\n  Why Procedure is Necessary:\n  {mnj.get('why_procedure_necessary')}")
    lines.append(f"\n  Supporting Clinical Evidence:\n  {mnj.get('supporting_clinical_evidence')}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("SECTION 6 — COST ESTIMATE")
    lines.append("-" * 70)
    cost = form.get("cost_information", {})
    for item in cost.get("line_items", []):
        lines.append(f"  {item.get('description'):<45} CPT {item.get('cpt_code')}  ${item.get('estimated_cost')}")
    lines.append(f"  {'TOTAL ESTIMATED COST':<45}      ${cost.get('estimated_total_cost')}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("SECTION 7 — PHYSICIAN ATTESTATION")
    lines.append("-" * 70)
    att = form.get("attestation", {})
    lines.append(f"  {att.get('statement')}")
    lines.append(f"\n  Physician: {att.get('physician_name')}")
    lines.append(f"  Signature: {att.get('physician_signature_placeholder')}")
    lines.append(f"  Date     : {att.get('date')}")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent5_output: dict, output_dir: str = ".") -> dict:
    """
    Agent 6 entry point.

    Parameters
    ----------
    agent5_output : dict — output from Agent 5
    output_dir    : str  — directory to save form files

    Returns
    -------
    dict with form_json, form_text, and saved file paths
    """
    print("\n[Agent 6] Prior Authorization Form Generator — START")

    data       = agent5_output.get("data", {})
    eligibility = {
        "determination":     agent5_output.get("determination"),
        "criteria_met_count": agent5_output.get("criteria_met_count"),
        "criteria_total":     agent5_output.get("criteria_total"),
        "clinical_summary":   agent5_output.get("clinical_summary"),
        "recommendation":     agent5_output.get("recommendation"),
    }

    prompt   = _build_prompt(data, eligibility)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    print("[Agent 6] Calling Nova Lite to generate form...")
    raw = invoke(
        model_id=LITE_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.1,
    )

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        form_json = json.loads(text.strip())
    except json.JSONDecodeError:
        print("[Agent 6] ⚠ JSON parse error — returning raw text")
        form_json = {"raw_form": raw}

    form_text = _render_form_text(form_json)

    # Save files
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mrn = data.get("patient_mrn", "unknown").replace("-", "")
    json_path = str(Path(output_dir) / f"prior_auth_form_{mrn}_{ts}.json")
    txt_path  = str(Path(output_dir) / f"prior_auth_form_{mrn}_{ts}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(form_json, f, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(form_text)

    print(f"  Form saved → {json_path}")
    print(f"  Form saved → {txt_path}")
    print("[Agent 6] Prior Authorization Form Generator — DONE\n")

    return {
        "form_json":      form_json,
        "form_text":      form_text,
        "form_json_path": json_path,
        "form_txt_path":  txt_path,
        "data":           data,
        "eligibility":    eligibility,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "eligible": True,
        "determination": "APPROVED",
        "confidence": "high",
        "criteria_met_count": 4,
        "criteria_total": 4,
        "clinical_summary": "Patient meets all BCBS criteria for MRI Lumbar Spine authorization.",
        "recommendation": "Approve MRI Lumbar Spine Without Contrast (CPT 72148).",
        "data": {
            "patient_name": "John R. Doe",
            "patient_dob": "1985-05-12",
            "patient_mrn": "PT-88293",
            "patient_gender": "Male",
            "patient_address": "12 Maple St, Boston, MA 02118",
            "patient_phone": "(555) 123-4567",
            "patient_email": "j.doe85@example.com",
            "insurer": "BlueCross BlueShield",
            "plan_type": "PPO",
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
            "prior_treatments": ["NSAIDs 4 weeks minimal relief", "Physical Therapy 3 weeks"],
            "symptom_duration_weeks": 7,
            "pain_score": 7,
            "clinical_findings": "Positive SLR at 30 degrees. Antalgic gait. Severe radicular pain.",
            "prescribed_medications": ["Lyrica 75mg bid", "Ibuprofen"],
            "cost_line_items": [
                {"description": "MRI Lumbar Spine Without Contrast", "cpt_code": "72148", "estimated_price": 1350},
                {"description": "Radiologist Interpretation", "cpt_code": "72148-26", "estimated_price": 250},
                {"description": "Orthopedic Consultation", "cpt_code": "99204", "estimated_price": 220},
                {"description": "Physical Therapy Evaluation", "cpt_code": "97161", "estimated_price": 160},
            ],
        },
    }
    result = run(mock_input)
    print(result["form_text"])
