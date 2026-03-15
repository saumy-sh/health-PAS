"""
agent8_status_tracker.py
─────────────────────────
Agent 8 — Claim Status Tracker
• Polls insurer portal for prior-auth decision status
• Parses status responses (approved / denied / pending / more-info-needed)
• Uses Nova Micro for status text classification
• In production: replace _poll_portal_status() with real API polling

Input  : dict from Agent 7
Output : dict  { "status": str, "decision": str, "action_required": bool, "details": {...} }
"""

import json
import random
from datetime import datetime

from bedrock_client import invoke, MICRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Simulated portal status polling
# In production: replace with real API call to insurer portal
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# LLM status classification + action extraction
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a prior-authorization status analyst. "
            "Parse insurer portal status messages and extract structured information. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _classify_status(portal_response: dict, data: dict) -> dict:
    prompt = f"""Parse the following prior-authorization portal status response.

Tracking Number : {portal_response.get('tracking_number')}
Insurer         : {portal_response.get('insurer')}
Raw Status Code : {portal_response.get('raw_status')}
Portal Message  : {portal_response.get('message')}
Decision Code   : {portal_response.get('decision_code')}

Patient Name    : {data.get('patient_name')}
Requested CPT   : {data.get('cpt')}
Diagnosis ICD10 : {data.get('icd10')}

Return JSON:
{{
  "decision": "APPROVED" | "DENIED" | "PENDING" | "MORE_INFO_NEEDED" | "EXPIRED" | "CANCELLED",
  "action_required": true | false,
  "action_type": "none" | "submit_additional_docs" | "appeal" | "resubmit" | "contact_insurer",
  "auth_number": string | null,
  "auth_valid_days": integer | null,
  "additional_docs_requested": [string],
  "next_steps": [string],
  "escalate_to_appeal": true | false,
  "summary": string
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
        return {
            "decision": "PENDING",
            "action_required": True,
            "action_type": "contact_insurer",
            "summary": "Status parse error — manual review required.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent7_output: dict) -> dict:
    """
    Agent 8 entry point.

    Parameters
    ----------
    agent7_output : dict — output from Agent 7

    Returns
    -------
    dict with decision status and next action
    """
    print("\n[Agent 8] Claim Status Tracker — START")
    print("  Submission was not automated — status tracking requires a real tracking number.")

    output = {
        "tracking_number":  agent7_output.get("tracking_number"),  # will be None
        "decision":         "PENDING_SUBMISSION",
        "action_required":  True,
        "action_type":      "manual_submission",
        "next_steps":       [
            f"Submit the generated form to: {agent7_output.get('portal_url')}",
            "Obtain tracking number after submission",
            "Re-run status tracker with real tracking number to check decision",
        ],
        "summary": "Prior auth form generated successfully. Awaiting manual portal submission.",
        "data":    agent7_output.get("data", {}),
    }

    print(f"  Status : {output['decision']}")
    for step in output["next_steps"]:
        print(f"  -> {step}")
    print("[Agent 8] Claim Status Tracker — DONE\n")
    return output

# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "tracking_number": "PA-ABC12345",
        "data": {
            "patient_name": "John R. Doe",
            "insurer": "BlueCross BlueShield",
            "cpt": "72148",
            "icd10": "M54.16",
        },
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k not in ("data", "portal_response")}, indent=2))
