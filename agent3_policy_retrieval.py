"""
agent3_policy_retrieval.py
───────────────────────────
Agent 3 — Policy Retrieval Agent
• Uses Nova Lite's medical/insurance knowledge to determine
  prior-authorization requirements for a given insurer + ICD-10 + CPT
• Now includes the full document list from Agent 1 for richer context
• Returns required documents WITH the suggested document type that should contain each

Input  : dict from Agent 2
Output : dict  { "authorization_required": bool, "policy_analysis": {...}, ... }
"""

import json

from bedrock_client import invoke, LITE_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a senior prior-authorization policy specialist with 20+ years "
            "of experience across all major US health insurers including BlueCross BlueShield, "
            "Aetna, Cigna, UnitedHealth, Humana, and Medicare/Medicaid. "
            "You know exact clinical criteria, documentation requirements, step-therapy protocols, "
            "specialist referral rules, imaging guidelines, and appeals processes for every "
            "common procedure and diagnosis combination. "
            "Be specific, thorough, and clinically precise. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_doc_summary(documents: list) -> str:
    """Build a readable summary of documents available."""
    if not documents:
        return "No documents available."
    lines = []
    for doc in documents:
        dtype   = doc.get("document_type", "Unknown")
        content = doc.get("content", {})
        keys    = list(content.keys())
        lines.append(f"  - {dtype}: contains {keys}")
    return "\n".join(lines)


def _build_prompt(data: dict) -> str:
    doc_summary = _build_doc_summary(data.get("documents", []))

    # Build a clear summary of which document types are confirmed present
    docs_present = data.get("documents_present", {})
    present_list  = [k for k, v in docs_present.items() if v]
    absent_list   = [k for k, v in docs_present.items() if not v]

    return f"""You are reviewing a prior-authorization request. Perform a THOROUGH policy analysis.

CLAIM DETAILS:
- Insurer              : {data.get('insurer')}
- Plan Type            : {data.get('plan_type')}
- Patient Name         : {data.get('patient_name')}
- Patient DOB          : {data.get('patient_dob')}
- Diagnosis            : {data.get('diagnosis')}
- ICD-10 Codes         : {data.get('icd10_codes')}
- Requested Procedure  : {data.get('procedure')}
- CPT Codes            : {data.get('cpt_codes')}
- Ordering Physician   : {data.get('ordering_physician')}
- Physician Specialty  : {data.get('physician_specialty')}
- Facility             : {data.get('hospital')}
- Symptom Duration     : {data.get('symptom_duration_weeks')} weeks
- Pain Score           : {data.get('pain_score')}/10
- Clinical Findings    : {data.get('clinical_findings')}
- Prior Treatments     : {json.dumps(data.get('prior_treatments', []))}
- Prescribed Meds      : {json.dumps(data.get('prescribed_medications', []))}
- Lab Results          : {json.dumps(data.get('lab_results', {}))}
- Estimated Cost       : ${data.get('total_estimated_cost')}

DOCUMENTS CONFIRMED PRESENT (already in hand — do NOT list these as missing):
{json.dumps(present_list, indent=2)}

DOCUMENTS NOT YET PROVIDED:
{json.dumps(absent_list, indent=2)}

DOCUMENT CONTENT DETAIL:
{doc_summary}

CRITICAL INSTRUCTION FOR required_documents AND missing_documents:
Before marking any document as missing, check the DOCUMENTS CONFIRMED PRESENT list above.
- "doctor_notes" / "Doctor Notes" covers: Clinical Notes, Office Visit Notes, Physician Notes,
  Procedure Template, Clinical Findings, Treatment History.
- "pretreatment_estimate" covers: Cost Estimate, Service Estimate, Billing Estimate,
  CPT Code List, Procedure Order.
- "insurance_card" covers: Insurance Information, Member Card, Policy Details.
- "patient_info" covers: Patient Demographics, Patient Registration.
- "lab_report" covers: Laboratory Results, Diagnostic Results, Blood Work.
- "prior_treatment_documentation" is confirmed present if "doctor_notes" is present
  and prior treatments are documented in it.
Only mark a document as missing if NO confirmed-present document can cover it.

Analyze this claim against {data.get('insurer')} standard policies for CPT {data.get('cpt')}.

For the required_documents field, each entry must include:
- "document_name": the name of the document required
- "document_type": what type of document this is
- "info_to_include": what specific information must be present
- "currently_available": true if a confirmed-present document covers this, false otherwise

Return a JSON object:
{{
  "authorization_required": true | false,
  "primary_cpt_requiring_auth": string,
  "cpt_auth_status": {{
    "<cpt_code>": "auth_required" | "no_auth_needed" | "conditional"
  }},
  "standard_clinical_criteria": {{
    "minimum_symptom_duration_weeks": integer,
    "minimum_symptom_duration_met": true | false,
    "actual_symptom_duration_weeks": integer,
    "step_therapy_required": true | false,
    "step_therapy_steps_required": [string],
    "step_therapy_steps_completed": [string],
    "step_therapy_met": true | false,
    "step_therapy_gaps": [string],
    "clinical_exam_required": true | false,
    "clinical_exam_findings_required": [string],
    "clinical_exam_met": true | false,
    "specialist_referral_required": true | false,
    "specialist_referral_met": true | false,
    "prior_imaging_required": true | false,
    "prior_imaging_met": true | false,
    "medication_trial_required": true | false,
    "medication_trial_details": string,
    "medication_trial_met": true | false
  }},
  "criteria_met": [string],
  "criteria_not_met": [string],
  "required_documents": [
    {{
      "document_name": string,
      "document_type": string,
      "info_to_include": string,
      "currently_available": true | false
    }}
  ],
  "missing_documents": [
    {{
      "document_name": string,
      "document_type": string,
      "info_required": string,
      "why_needed": string
    }}
  ],
  "missing_information": [
    {{
      "info": string,
      "should_be_in_document": string
    }}
  ],
  "unmet_requirements": [
    {{
      "requirement": string,
      "why_not_met": string,
      "what_is_needed": string,
      "document_needed": string
    }}
  ],
  "approval_timeline_days": integer,
  "validity_period_days": integer,
  "applicable_copay": number,
  "submission_portal": string,
  "policy_notes": string,
  "likelihood_of_approval": "high" | "medium" | "low",
  "reasoning": string
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent2_output: dict) -> dict:
    print("\n[Agent 3] Policy Retrieval Agent — START")

    data = agent2_output.get("data", agent2_output)

    print(f"  Insurer   : {data.get('insurer')}")
    print(f"  CPT codes : {data.get('cpt_codes')}")
    print(f"  ICD-10    : {data.get('icd10_codes')}")
    print(f"  Docs available: {[d.get('document_type') for d in data.get('documents', [])]}")
    print("[Agent 3] Querying Nova Lite for policy requirements...")

    prompt   = _build_prompt(data)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    raw = invoke(
        model_id=LITE_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=1500,
        temperature=0.1,
    )

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        policy_analysis = json.loads(text.strip())
    except json.JSONDecodeError:
        print("[Agent 3] Warning: JSON parse error, using fallback")
        policy_analysis = {
            "authorization_required": True,
            "required_documents": [
                {
                    "document_name": "Completed prior authorization request form",
                    "document_type": "Authorization Form",
                    "info_to_include": "Patient ID, insurer, CPT, ICD-10, physician NPI",
                    "currently_available": False,
                },
                {
                    "document_name": "Office visit notes within 60 days",
                    "document_type": "Clinical Notes",
                    "info_to_include": "Diagnosis, clinical findings, treatment plan",
                    "currently_available": True,
                },
            ],
            "missing_documents": [],
            "missing_information": [],
            "unmet_requirements": [],
            "likelihood_of_approval": "medium",
            "policy_notes": "Parse error — standard auth criteria applied.",
            "reasoning": raw,
        }

    # Normalize missing_documents — support both old (list of strings) and new (list of dicts)
    raw_missing = policy_analysis.get("missing_documents", [])
    missing_docs_normalized = []
    for item in raw_missing:
        if isinstance(item, str):
            missing_docs_normalized.append({
                "document_name": item,
                "document_type": "Unknown",
                "info_required": "",
                "why_needed": "",
            })
        else:
            missing_docs_normalized.append(item)

    # Normalize missing_information — support both string list and dict list
    raw_missing_info = policy_analysis.get("missing_information", [])
    missing_info_normalized = []
    for item in raw_missing_info:
        if isinstance(item, str):
            missing_info_normalized.append({
                "info": item,
                "should_be_in_document": "Unknown",
            })
        else:
            missing_info_normalized.append(item)

    output = {
        "authorization_required": policy_analysis.get("authorization_required", True),
        "policy_analysis":        policy_analysis,
        "missing_documents":      missing_docs_normalized,
        "missing_information":    missing_info_normalized,
        "unmet_requirements":     policy_analysis.get("unmet_requirements", []),
        "criteria_not_met":       policy_analysis.get("criteria_not_met", []),
        "data":                   data,
    }

    print(f"  Auth required       : {output['authorization_required']}")
    print(f"  Approval likelihood : {policy_analysis.get('likelihood_of_approval')}")
    print(f"  Criteria NOT met    : {policy_analysis.get('criteria_not_met')}")
    if missing_docs_normalized:
        print("  Missing documents:")
        for d in missing_docs_normalized:
            print(f"    - [{d.get('document_type')}] {d.get('document_name')} — needs: {d.get('info_required')}")
    if missing_info_normalized:
        print("  Missing information:")
        for m in missing_info_normalized:
            print(f"    - {m.get('info')} → expected in: {m.get('should_be_in_document')}")
    print("[Agent 3] Policy Retrieval Agent — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "data": {
            "patient_name": "John R. Doe",
            "insurer": "BlueCross BlueShield",
            "plan_type": "PPO",
            "diagnosis": "Lumbar Radiculopathy",
            "icd10_codes": ["M54.16"],
            "icd10": "M54.16",
            "procedure": "MRI Lumbar Spine Without Contrast",
            "cpt_codes": ["72148"],
            "cpt": "72148",
            "ordering_physician": "Dr. Sarah Jenkins",
            "physician_specialty": "Orthopedic Surgery",
            "hospital": "Metropolitan General Hospital",
            "symptom_duration_weeks": 7,
            "pain_score": 7,
            "clinical_findings": "Positive SLR at 30 degrees right side. Antalgic gait.",
            "prior_treatments": ["NSAIDs 4 weeks minimal relief", "Physical Therapy 3 weeks"],
            "total_estimated_cost": 2025,
            "documents": [
                {"document_type": "Lab Report", "content": {"hba1c": "5.6%"}},
                {"document_type": "Doctor Notes", "content": {"diagnosis": "Lumbar Radiculopathy"}},
                {"document_type": "Patient Information Sheet", "content": {"patient_name": "John R. Doe"}},
                {"document_type": "Insurance Card", "content": {"policy_number": "BCBS-99001122"}},
                {"document_type": "Pretreatment Estimate", "content": {"total_estimated_cost": 2025}},
            ],
        }
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
