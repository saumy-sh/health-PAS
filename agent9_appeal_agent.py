"""
agent9_appeal_agent.py
───────────────────────
Agent 9 — Appeal Agent
• Triggered only if prior-auth was DENIED or MORE_INFO_NEEDED
• Analyzes denial reasons
• Generates a formal appeal letter with clinical justification
• Uses Nova Pro (persuasive clinical writing)

Input  : dict from Agent 8
Output : dict  { "appeal_letter": str, "appeal_json": {...}, "appeal_file": str }
"""

import json
from datetime import datetime
from pathlib import Path

from bedrock_client import invoke, PRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a senior medical prior-authorization appeal specialist. "
            "Write formal, persuasive, and clinically precise appeal letters "
            "that cite relevant medical evidence and policy criteria. "
            "Appeals should be professional, factual, and compelling. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_appeal_prompt(data: dict, denial_info: dict, eligibility: dict) -> str:
    return f"""Write a formal prior-authorization appeal letter for the following denied/flagged request.

DENIAL/FLAG INFORMATION:
Decision         : {denial_info.get('decision')}
Action Type      : {denial_info.get('action_type')}
Portal Message   : {denial_info.get('portal_message')}
Additional Docs  : {json.dumps(denial_info.get('additional_docs_requested', []))}
Escalate Appeal  : {denial_info.get('escalate_to_appeal')}

PATIENT & CLAIM DATA:
Patient Name     : {data.get('patient_name')}
DOB              : {data.get('patient_dob')}
Member ID        : {data.get('member_id')}
Policy Number    : {data.get('policy_number')}
Insurer          : {data.get('insurer')}
Diagnosis        : {data.get('diagnosis')} ({data.get('icd10')})
Procedure        : {data.get('procedure')} (CPT {data.get('cpt')})
Physician        : {data.get('ordering_physician')} — {data.get('physician_specialty')}
Hospital         : {data.get('hospital')}
Symptom Duration : {data.get('symptom_duration_weeks')} weeks
Pain Score       : {data.get('pain_score')}/10
Clinical Findings: {data.get('clinical_findings')}
Prior Treatments : {json.dumps(data.get('prior_treatments', []))}
Medications      : {json.dumps(data.get('prescribed_medications', []))}

ELIGIBILITY ASSESSMENT:
Criteria Met     : {eligibility.get('criteria_met_count')}/{eligibility.get('criteria_total')}
Clinical Summary : {eligibility.get('clinical_summary')}
Approval Likelihood Before Denial: {eligibility.get('confidence')}

Return a JSON object:
{{
  "appeal_metadata": {{
    "date": string,
    "re_line": string,
    "priority": "standard" | "urgent" | "expedited"
  }},
  "to": {{
    "name": "Medical Director",
    "organization": string,
    "address": string
  }},
  "from": {{
    "physician_name": string,
    "specialty": string,
    "facility": string,
    "phone": string,
    "address": string
  }},
  "subject": string,
  "opening_paragraph": string,
  "denial_response": string,
  "clinical_justification": string,
  "medical_necessity_argument": string,
  "policy_compliance_argument": string,
  "supporting_evidence": [string],
  "additional_docs_provided": [string],
  "closing_paragraph": string,
  "requested_action": string,
  "urgency_statement": string | null,
  "signature_block": string
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Letter renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_appeal_letter(appeal: dict) -> str:
    lines = []
    sep = "=" * 70

    meta = appeal.get("appeal_metadata", {})
    frm  = appeal.get("from", {})
    to   = appeal.get("to", {})

    lines.append(f"{frm.get('physician_name', '')}")
    lines.append(f"{frm.get('specialty', '')}")
    lines.append(f"{frm.get('facility', '')}")
    lines.append(f"{frm.get('address', '')}")
    lines.append(f"Phone: {frm.get('phone', '')}")
    lines.append(f"\nDate: {meta.get('date', datetime.now().strftime('%Y-%m-%d'))}")
    lines.append("")
    lines.append(f"RE: {meta.get('re_line', 'Prior Authorization Appeal')}")
    lines.append(f"Priority: {meta.get('priority', 'standard').upper()}")
    lines.append("")
    lines.append(sep)
    lines.append(f"To: {to.get('name', 'Medical Director')}")
    lines.append(f"    {to.get('organization', '')}")
    lines.append(f"    {to.get('address', '')}")
    lines.append(sep)
    lines.append("")
    lines.append(f"Subject: {appeal.get('subject', 'Appeal of Prior Authorization Denial')}")
    lines.append("")

    for section_key in [
        "opening_paragraph",
        "denial_response",
        "clinical_justification",
        "medical_necessity_argument",
        "policy_compliance_argument",
    ]:
        content = appeal.get(section_key, "")
        if content:
            lines.append(content)
            lines.append("")

    evidence = appeal.get("supporting_evidence", [])
    if evidence:
        lines.append("Supporting Evidence:")
        for e in evidence:
            lines.append(f"  • {e}")
        lines.append("")

    docs = appeal.get("additional_docs_provided", [])
    if docs:
        lines.append("Documents Attached:")
        for d in docs:
            lines.append(f"  • {d}")
        lines.append("")

    urgency = appeal.get("urgency_statement")
    if urgency:
        lines.append(f"URGENCY NOTE: {urgency}")
        lines.append("")

    lines.append(appeal.get("closing_paragraph", ""))
    lines.append("")
    lines.append(f"Requested Action: {appeal.get('requested_action', '')}")
    lines.append("")
    lines.append(appeal.get("signature_block", ""))
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent8_output: dict, eligibility: dict = None, output_dir: str = ".") -> dict:
    """
    Agent 9 entry point.
    Only triggered when decision is DENIED or MORE_INFO_NEEDED.

    Parameters
    ----------
    agent8_output : dict — output from Agent 8
    eligibility   : dict — output from Agent 5 (passed through orchestrator)
    output_dir    : str  — directory to save appeal files

    Returns
    -------
    dict with appeal letter text, JSON, and saved file path
    """
    print("\n[Agent 9] Appeal Agent — START")

    decision = agent8_output.get("decision", "")
    data     = agent8_output.get("data", {})

    if decision == "APPROVED":
        print("[Agent 9] Decision is APPROVED — no appeal needed.")
        print("[Agent 9] Appeal Agent — SKIPPED\n")
        return {
            "appeal_needed": False,
            "decision":      decision,
            "message":       "Prior authorization approved — no appeal required.",
            "data":          data,
        }

    print(f"  Decision: {decision} — generating appeal...")

    denial_info = {
        "decision":                  decision,
        "action_type":               agent8_output.get("action_type"),
        "portal_message":            agent8_output.get("portal_response", {}).get("message"),
        "additional_docs_requested": agent8_output.get("additional_docs_requested", []),
        "escalate_to_appeal":        agent8_output.get("escalate_to_appeal", False),
    }

    eligibility = eligibility or {
        "criteria_met_count": None,
        "criteria_total":     None,
        "clinical_summary":   data.get("clinical_findings"),
        "confidence":         "medium",
    }

    prompt   = _build_appeal_prompt(data, denial_info, eligibility)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    print("[Agent 9] Calling Nova Pro to write appeal letter...")
    raw = invoke(
        model_id=PRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.2,  # slight creativity for persuasive writing
    )

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        appeal_json = json.loads(text.strip())
    except json.JSONDecodeError:
        print("[Agent 9] ⚠ JSON parse error — using raw text")
        appeal_json = {"raw_letter": raw}

    appeal_letter = _render_appeal_letter(appeal_json)

    # Save
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    mrn = data.get("patient_mrn", "unknown").replace("-", "")
    json_path   = str(Path(output_dir) / f"appeal_letter_{mrn}_{ts}.json")
    letter_path = str(Path(output_dir) / f"appeal_letter_{mrn}_{ts}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(appeal_json, f, indent=2)
    with open(letter_path, "w", encoding="utf-8") as f:
        f.write(appeal_letter)

    print(f"  Appeal saved → {json_path}")
    print(f"  Appeal saved → {letter_path}")
    print("[Agent 9] Appeal Agent — DONE\n")

    return {
        "appeal_needed":    True,
        "decision":         decision,
        "appeal_json":      appeal_json,
        "appeal_letter":    appeal_letter,
        "appeal_json_path": json_path,
        "appeal_txt_path":  letter_path,
        "data":             data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "decision": "DENIED",
        "action_type": "appeal",
        "escalate_to_appeal": True,
        "additional_docs_requested": [],
        "portal_response": {
            "message": "Denied: insufficient documentation of conservative treatment failure."
        },
        "data": {
            "patient_name": "John R. Doe",
            "patient_dob": "1985-05-12",
            "patient_mrn": "PT-88293",
            "member_id": "BCBS-9900122",
            "policy_number": "BCBS-99001122",
            "insurer": "BlueCross BlueShield",
            "diagnosis": "Lumbar Radiculopathy",
            "icd10": "M54.16",
            "procedure": "MRI Lumbar Spine Without Contrast",
            "cpt": "72148",
            "ordering_physician": "Dr. Sarah Jenkins",
            "physician_specialty": "Orthopedic Surgery",
            "physician_phone": "555-010-8899",
            "hospital": "Metropolitan General Hospital",
            "symptom_duration_weeks": 7,
            "pain_score": 7,
            "clinical_findings": "Positive SLR at 30 degrees. Antalgic gait.",
            "prior_treatments": ["NSAIDs 4 weeks minimal relief", "PT 3 weeks"],
            "prescribed_medications": ["Lyrica 75mg bid"],
        },
    }
    result = run(mock_input)
    if result.get("appeal_needed"):
        print(result["appeal_letter"])
