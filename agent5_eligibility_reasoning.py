"""
agent5_eligibility_reasoning.py
────────────────────────────────
Agent 5 — Eligibility / Policy Reasoning Agent
• Synthesizes all prior agent outputs
• Makes a comprehensive eligibility determination
• Produces a structured reasoning chain and approval recommendation
• Passes urgency forward to Agent 6 for form rendering
• Uses Nova Pro (complex multi-document reasoning)

Input  : dict from Agent 4
Output : dict  { "eligible": bool, "confidence": str, "reasoning": {...}, "recommendation": str }
"""

import json

from bedrock_client import invoke, PRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a senior medical prior-authorization reviewer at an insurance company. "
            "Your job is to make accurate, fair eligibility determinations based on clinical "
            "evidence, policy criteria, and documentation completeness. "
            "You reason step by step and document your clinical and policy logic clearly. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_prompt(data: dict, policy_analysis: dict, doc_status: dict) -> str:
    criteria = policy_analysis.get("clinical_criteria_met", {})

    return f"""You must make an eligibility determination for the following prior-authorization request.
Reason through each criterion carefully.

═══ PATIENT & CLAIM SUMMARY ═══
Patient            : {data.get('patient_name')}, DOB {data.get('patient_dob')}
Insurer            : {data.get('insurer')} | Policy: {data.get('policy_number')}
Diagnosis          : {data.get('diagnosis')} ({data.get('icd10')})
Requested Procedure: {data.get('procedure')} (CPT: {data.get('cpt')})
Ordering Physician : {data.get('ordering_physician')} — {data.get('physician_specialty')}
Facility           : {data.get('hospital')}
Date of Service    : {data.get('date_of_proposed_treatment')}
Estimated Cost     : ${data.get('total_estimated_cost')}
Urgency            : {data.get('urgency', 'routine').upper()}

═══ CLINICAL EVIDENCE ═══
Chief Complaint    : {data.get('diagnosis')}
Symptom Duration   : {data.get('symptom_duration_weeks')} weeks
Pain Score         : {data.get('pain_score')}/10
Clinical Findings  : {data.get('clinical_findings')}
Prior Treatments   : {json.dumps(data.get('prior_treatments', []))}
Prescribed Meds    : {json.dumps(data.get('prescribed_medications', []))}

═══ POLICY CRITERIA ASSESSMENT (from Agent 3) ═══
{json.dumps(criteria, indent=2)}

Policy Notes       : {policy_analysis.get('policy_notes')}
Approval Likelihood: {policy_analysis.get('likelihood_of_approval')}
Policy Reasoning   : {policy_analysis.get('reasoning')}

═══ DOCUMENTATION STATUS (from Agent 4) ═══
All Docs Present   : {doc_status.get('all_docs_present')}
Documents on File  : {json.dumps(data.get('documents_present', {}))}
Missing Docs       : {json.dumps(doc_status.get('missing_docs', []))}
Blockers           : {json.dumps(doc_status.get('blockers', []))}
Missing Information: {json.dumps(doc_status.get('missing_information', []))}

Now perform a full eligibility determination. Return JSON:
{{
  "eligible": true | false,
  "determination": "APPROVED" | "DENIED" | "PENDING_DOCS" | "PENDING_REVIEW",
  "confidence": "high" | "medium" | "low",
  "criteria_evaluation": {{
    "symptom_duration": {{
      "required": "6+ weeks",
      "actual": string,
      "met": true | false,
      "notes": string
    }},
    "conservative_treatment": {{
      "required": "at least 1 failed treatment",
      "actual": string,
      "met": true | false,
      "notes": string
    }},
    "clinical_examination": {{
      "required": "documented clinical findings supporting diagnosis",
      "actual": string,
      "met": true | false,
      "notes": string
    }},
    "documentation_complete": {{
      "required": "all required docs submitted",
      "actual": string,
      "met": true | false,
      "notes": string
    }}
  }},
  "criteria_met_count": integer,
  "criteria_total": integer,
  "denial_reasons": [string],
  "approval_conditions": [string],
  "clinical_summary": string,
  "recommendation": string,
  "urgency": "routine" | "urgent" | "emergent",
  "expedited_review_requested": true | false,
  "suggested_approval_validity_days": integer,
  "reviewer_notes": string
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent4_output: dict) -> dict:
    """
    Agent 5 entry point.

    Parameters
    ----------
    agent4_output : dict — output from Agent 4

    Returns
    -------
    dict with full eligibility determination
    """
    print("\n[Agent 5] Eligibility / Policy Reasoning Agent — START")

    data            = agent4_output.get("data", {})
    policy_analysis = data.get("_raw", {})

    if "policy_analysis" not in data:
        policy_analysis = agent4_output.get("policy_analysis", {})
    else:
        policy_analysis = data.get("policy_analysis", {})

    doc_status = {
        "all_docs_present": agent4_output.get("all_docs_present"),
        "missing_docs":     agent4_output.get("missing_docs", []),
        "blockers":         agent4_output.get("blockers", []),
    }

    prompt = _build_prompt(data, policy_analysis, doc_status)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    print("[Agent 5] Calling Nova Pro for eligibility reasoning...")
    raw = invoke(
        model_id=PRO_MODEL_ID,
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
        determination = json.loads(text.strip())
    except json.JSONDecodeError:
        determination = {
            "eligible": False,
            "determination": "PENDING_REVIEW",
            "confidence": "low",
            "recommendation": "Manual review required due to parse error.",
        }

    output = {
        "eligible":                determination.get("eligible"),
        "determination":           determination.get("determination"),
        "confidence":              determination.get("confidence"),
        "criteria_evaluation":     determination.get("criteria_evaluation", {}),
        "criteria_met_count":      determination.get("criteria_met_count"),
        "criteria_total":          determination.get("criteria_total"),
        "denial_reasons":          determination.get("denial_reasons", []),
        "approval_conditions":     determination.get("approval_conditions", []),
        "recommendation":          determination.get("recommendation"),
        # Urgency passed through for Agent 6 form-filling
        "urgency":                 determination.get("urgency", data.get("urgency", "routine")),
        "expedited_review_requested": determination.get("expedited_review_requested", False),
        "clinical_summary":        determination.get("clinical_summary"),
        "reviewer_notes":          determination.get("reviewer_notes"),
        "suggested_validity_days": determination.get("suggested_approval_validity_days"),
        "data":                    data,
        "policy_analysis":         policy_analysis,
    }

    print(f"  Determination          : {output['determination']}")
    print(f"  Eligible               : {output['eligible']}")
    print(f"  Confidence             : {output['confidence']}")
    print(f"  Criteria met           : {output['criteria_met_count']}/{output['criteria_total']}")
    print(f"  Urgency                : {output['urgency']}")
    print(f"  Expedited review       : {output['expedited_review_requested']}")
    print("[Agent 5] Eligibility / Policy Reasoning Agent — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "all_docs_present": True,
        "missing_docs": [],
        "blockers": [],
        "data": {
            "patient_name": "John R. Doe",
            "patient_dob": "1985-05-12",
            "insurer": "BlueCross BlueShield",
            "policy_number": "BCBS-99001122",
            "diagnosis": "Lumbar Radiculopathy",
            "icd10": "M54.16",
            "procedure": "MRI Lumbar Spine Without Contrast",
            "cpt": "72148",
            "ordering_physician": "Dr. Sarah Jenkins",
            "physician_specialty": "Orthopedic Surgery",
            "hospital": "Metropolitan General Hospital",
            "date_of_proposed_treatment": "2024-03-15",
            "total_estimated_cost": 2025,
            "symptom_duration_weeks": 7,
            "pain_score": 7,
            "urgency": "routine",
            "urgency_justification": "No emergent indicators.",
            "clinical_findings": "Positive SLR at 30 degrees right side. Antalgic gait.",
            "prior_treatments": ["NSAIDs 4 weeks minimal relief", "PT 3 weeks no improvement"],
            "prescribed_medications": ["Lyrica 75mg bid", "Ibuprofen"],
        },
        "policy_analysis": {
            "clinical_criteria_met": {
                "symptom_duration_met": True,
                "conservative_treatment_met": True,
                "clinical_exam_met": True,
                "no_recent_imaging": True,
            },
            "likelihood_of_approval": "high",
            "policy_notes": "All BCBS criteria appear to be met.",
            "reasoning": "Patient meets duration and treatment failure criteria.",
        },
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k not in ("data", "policy_analysis")}, indent=2))
