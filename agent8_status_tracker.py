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

# Simulated status responses the portal might return
_SIMULATED_RESPONSES = [
    {
        "raw_status": "APPROVED",
        "message": "Prior authorization PA-{tracking} has been approved for CPT 72148. "
                   "Authorization valid 90 days from approval. Auth number: AUTH-{auth}.",
        "decision_code": "A1",
    },
    {
        "raw_status": "PENDING",
        "message": "Prior authorization PA-{tracking} is under clinical review. "
                   "Expected decision within 3 business days.",
        "decision_code": "P1",
    },
    {
        "raw_status": "MORE_INFO_NEEDED",
        "message": "Additional information required for PA-{tracking}. "
                   "Please submit: (1) Clinical notes from last 60 days, "
                   "(2) Documentation of failed conservative treatment.",
        "decision_code": "I1",
    },
]


def _poll_portal_status(tracking_number: str, insurer: str) -> dict:
    """
    SIMULATED portal status poll.

    Production upgrade path:
      - Availity: GET /v1/prior-auth/{tracking_number}/status
      - Parse HL7 278 response messages
      - Handle OAuth2 token refresh
    """
    # For demo: always return APPROVED (change index to test other flows)
    template = _SIMULATED_RESPONSES[0]

    import uuid
    auth_num = uuid.uuid4().hex[:6].upper()

    return {
        "tracking_number": tracking_number,
        "insurer":         insurer,
        "polled_at":       datetime.now().isoformat(),
        "raw_status":      template["raw_status"],
        "message":         template["message"].format(tracking=tracking_number, auth=auth_num),
        "decision_code":   template["decision_code"],
        "auth_number":     auth_num if template["raw_status"] == "APPROVED" else None,
    }


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

    tracking_number = agent7_output.get("tracking_number")
    data            = agent7_output.get("data", {})
    insurer         = data.get("insurer", "")

    print(f"  Tracking Number : {tracking_number}")
    print(f"  Insurer         : {insurer}")

    print("[Agent 8] Polling portal status (simulated)...")
    portal_response = _poll_portal_status(tracking_number, insurer)
    print(f"  Raw Status      : {portal_response.get('raw_status')}")

    print("[Agent 8] Classifying status via Nova Micro...")
    classification = _classify_status(portal_response, data)

    output = {
        "tracking_number":           tracking_number,
        "decision":                  classification.get("decision"),
        "action_required":           classification.get("action_required"),
        "action_type":               classification.get("action_type"),
        "auth_number":               classification.get("auth_number") or portal_response.get("auth_number"),
        "auth_valid_days":           classification.get("auth_valid_days"),
        "additional_docs_requested": classification.get("additional_docs_requested", []),
        "next_steps":                classification.get("next_steps", []),
        "escalate_to_appeal":        classification.get("escalate_to_appeal", False),
        "summary":                   classification.get("summary"),
        "portal_response":           portal_response,
        "polled_at":                 portal_response.get("polled_at"),
        "data":                      data,
    }

    print(f"  Decision        : {output['decision']}")
    print(f"  Action Required : {output['action_required']}")
    print(f"  Auth Number     : {output['auth_number']}")
    print(f"  Escalate Appeal : {output['escalate_to_appeal']}")
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
