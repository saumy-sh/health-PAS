"""
agent3_policy_retrieval.py
───────────────────────────
Agent 3 — Policy Retrieval Agent
• Uses Nova Lite's built-in medical/insurance knowledge to determine
  prior-authorization requirements for a given insurer + ICD-10 + CPT
• Factors in urgency level (routine / urgent / emergent) when determining
  approval timelines and submission paths (YAML: policy_requirement_agent)
• No RAG / vector store needed — Nova Lite reasons directly from its
  training knowledge of standard insurer policies

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
            "Never guess — if a criterion is typically required by this insurer for this procedure, state it. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_prompt(data: dict) -> str:
    urgency = data.get("urgency", "routine")
    urgency_note = (
        "This is an EXPEDITED / URGENT review request — apply accelerated timelines."
        if urgency in ("urgent", "emergent")
        else "This is a routine prior authorization request."
    )

    return f"""You are reviewing a prior-authorization request. Perform a THOROUGH policy analysis.

URGENCY: {urgency.upper()} — {urgency_note}

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

Analyze this claim against {data.get('insurer')} standard policies for
CPT {data.get('cpt')}.

For EACH of the following areas, reason carefully using what you know about
this insurer's policies for this specific procedure + diagnosis combination:

1. STEP THERAPY — Has the patient gone through required conservative treatment steps
   in the correct order and for sufficient duration? What exactly is required?

2. SYMPTOM DURATION — Is the documented duration sufficient? What is the minimum
   this insurer typically requires for this procedure?

3. CLINICAL EXAM FINDINGS — Are the documented findings sufficient to support
   medical necessity? What specific findings does this insurer require?

4. SPECIALIST INVOLVEMENT — Is the ordering physician's specialty appropriate?
   Is a referral from a PCP required first?

5. DOCUMENTATION — List EVERY specific document this insurer requires for this
   CPT code. Be exhaustive — include forms, notes, test results, letters.

6. IMAGING / PRIOR TESTING — Has any required prior imaging or testing been done
   before this procedure can be approved?

7. MEDICATION TRIAL — Has the patient trialed required medications (type, dose,
   duration) before this procedure is approved?

8. UNMET REQUIREMENTS — List EVERY requirement that is NOT yet met based on
   the claim details provided above.

9. MISSING INFORMATION — List every piece of information that was not provided
   but is needed to make a complete determination.

10. URGENCY — If urgency is 'urgent' or 'emergent', specify the expedited review
    timeline this insurer offers, the special submission path, and any additional
    clinical documentation required to justify the expedited request.

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
  "required_documents": [string],
  "missing_documents": [string],
  "missing_information": [string],
  "unmet_requirements": [
    {{
      "requirement": string,
      "why_not_met": string,
      "what_is_needed": string
    }}
  ],
  "approval_timeline_days": integer,
  "expedited_review_available": true | false,
  "expedited_timeline_days": integer | null,
  "expedited_submission_path": string | null,
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
    """
    Agent 3 entry point.

    Parameters
    ----------
    agent2_output : dict — output from Agent 2

    Returns
    -------
    dict with policy requirements and approval likelihood
    """
    print("\n[Agent 3] Policy Retrieval Agent — START")

    data = agent2_output.get("data", agent2_output)

    print(f"  Insurer   : {data.get('insurer')}")
    print(f"  CPT codes : {data.get('cpt_codes')}")
    print(f"  ICD-10    : {data.get('icd10_codes')}")
    print(f"  Urgency   : {data.get('urgency', 'routine')}")
    print("[Agent 3] Querying Nova Lite for policy requirements...")

    prompt   = _build_prompt(data)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    raw = invoke(
        model_id=LITE_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=1200,
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
                "Completed prior authorization request form",
                "Office visit notes within 60 days",
                "Documentation of conservative treatment failure",
                "Referring physician NPI and specialty",
                "ICD-10 and CPT codes",
                "Estimated date of service",
            ],
            "likelihood_of_approval": "medium",
            "policy_notes": "Parse error — standard auth criteria applied.",
            "reasoning": raw,
        }

    output = {
        "authorization_required":       policy_analysis.get("authorization_required", True),
        "policy_analysis":              policy_analysis,
        "missing_documents":            policy_analysis.get("missing_documents", []),
        "missing_information":          policy_analysis.get("missing_information", []),
        "unmet_requirements":           policy_analysis.get("unmet_requirements", []),
        "criteria_not_met":             policy_analysis.get("criteria_not_met", []),
        "expedited_review_available":   policy_analysis.get("expedited_review_available", False),
        "expedited_timeline_days":      policy_analysis.get("expedited_timeline_days"),
        "data":                         data,
    }

    print(f"  Auth required           : {output['authorization_required']}")
    print(f"  Approval likelihood     : {policy_analysis.get('likelihood_of_approval')}")
    print(f"  Required docs           : {policy_analysis.get('required_documents')}")
    print(f"  Criteria NOT met        : {policy_analysis.get('criteria_not_met')}")
    print(f"  Expedited available     : {output['expedited_review_available']}")
    print(f"  Missing documents       : {policy_analysis.get('missing_documents')}")
    print(f"  Missing information     : {policy_analysis.get('missing_information')}")
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
            "cpt_codes": ["72148", "72148-26", "99204", "97161"],
            "cpt": "72148",
            "ordering_physician": "Dr. Sarah Jenkins",
            "physician_specialty": "Orthopedic Surgery",
            "hospital": "Metropolitan General Hospital",
            "symptom_duration_weeks": 7,
            "pain_score": 7,
            "clinical_findings": "Positive SLR at 30 degrees right side. Antalgic gait.",
            "prior_treatments": ["NSAIDs 4 weeks minimal relief", "Physical Therapy 3 weeks"],
            "total_estimated_cost": 2025,
            "urgency": "routine",
            "urgency_justification": "No emergent indicators.",
        }
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
