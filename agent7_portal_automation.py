"""
agent7_portal_automation.py
─────────────────────────────
Agent 7 — Portal Automation Agent
• Simulates submission to insurer prior-auth portal
• In production: replace _submit_to_portal() with real HTTP/Selenium/API calls
• Uses Nova Lite for submission planning + response parsing
• Generates a submission record and tracking number

Input  : dict from Agent 6
Output : dict  { "submitted": bool, "tracking_number": str, "submission_record": {...} }
"""

import json
import uuid
from datetime import datetime

from bedrock_client import invoke, LITE_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Simulated portal API
# In production: replace with actual insurer portal API calls or
# browser automation (Playwright / Selenium)
# ─────────────────────────────────────────────────────────────────────────────

PORTAL_REGISTRY = {
    "bluecross blueshield": {
        "portal_name": "Availity",
        "portal_url":  "https://apps.availity.com/availity/web/prior-auth",
        "api_endpoint": "https://api.availity.com/v1/prior-auth/submit",
        "submission_method": "API",
        "auth_type": "OAuth2",
        "avg_response_hours": 72,
    },
    "default": {
        "portal_name": "Generic Insurer Portal",
        "portal_url":  "https://portal.insurer.com/prior-auth",
        "api_endpoint": None,
        "submission_method": "manual",
        "avg_response_hours": 96,
    },
}


def _get_portal_info(insurer: str) -> dict:
    key = insurer.lower().strip()
    for k, v in PORTAL_REGISTRY.items():
        if k in key or key in k:
            return v
    return PORTAL_REGISTRY["default"]


def _submit_to_portal(payload: dict, portal_info: dict) -> dict:
    """
    SIMULATED portal submission.

    Production upgrade path:
      - For Availity: use requests + OAuth2 token
      - For non-API portals: use Playwright browser automation
      - For fax: use Sfax or similar API

    Returns a simulated submission receipt.
    """
    tracking_number = f"PA-{uuid.uuid4().hex[:8].upper()}"
    submitted_at    = datetime.now().isoformat()

    return {
        "success":          True,
        "tracking_number":  tracking_number,
        "submitted_at":     submitted_at,
        "portal":           portal_info.get("portal_name"),
        "portal_url":       portal_info.get("portal_url"),
        "submission_method": portal_info.get("submission_method"),
        "expected_response_by": _expected_response_date(portal_info.get("avg_response_hours", 96)),
        "confirmation_message": f"Prior authorization request {tracking_number} received successfully.",
        "status": "SUBMITTED",
    }


def _expected_response_date(hours: int) -> str:
    from datetime import timedelta
    return (datetime.now() + timedelta(hours=hours)).strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# LLM — submission plan + payload assembly
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a prior-authorization portal submission specialist. "
            "Prepare a complete, correctly formatted submission payload "
            "for electronic portal submission. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_submission_payload(form_json: dict, portal_info: dict) -> dict:
    """
    Use LLM to map the generated form into the portal's expected format.
    For production, this would be schema-matched to the real portal API spec.
    """
    prompt = f"""Prepare a prior-authorization submission payload for the following portal:
Portal: {portal_info.get('portal_name')}
Submission Method: {portal_info.get('submission_method')}

Source Form Data:
{json.dumps(form_json, indent=2, default=str)}

Return a clean submission payload JSON:
{{
  "submission_type": "PRIOR_AUTH_REQUEST",
  "portal": string,
  "patient": {{
    "member_id": string,
    "date_of_birth": string,
    "last_name": string,
    "first_name": string
  }},
  "provider": {{
    "npi": string,
    "name": string,
    "tax_id": string
  }},
  "service_request": {{
    "primary_icd10": string,
    "all_icd10_codes": [string],
    "primary_cpt": string,
    "all_cpt_codes": [string],
    "place_of_service": string,
    "requested_service_date": string,
    "facility": string
  }},
  "clinical_notes": string,
  "urgency": string,
  "attachments_included": [string]
}}"""

    messages = [{"role": "user", "content": [{"text": prompt}]}]
    raw = invoke(
        model_id=LITE_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=800,
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
        return {"error": "payload_parse_error", "raw": raw}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent6_output: dict) -> dict:
    """
    Agent 7 entry point.

    Parameters
    ----------
    agent6_output : dict — output from Agent 6

    Returns
    -------
    dict with submission status and tracking number
    """
    print("\n[Agent 7] Portal Automation Agent — START")

    form_json  = agent6_output.get("form_json", {})
    data       = agent6_output.get("data", {})
    insurer    = data.get("insurer", "")

    portal_info = _get_portal_info(insurer)
    print(f"  Portal : {portal_info['portal_name']}")
    print(f"  Method : {portal_info['submission_method']}")
    print(f"  URL    : {portal_info['portal_url']}")

    print("[Agent 7] Building submission payload via Nova Lite...")
    submission_payload = _build_submission_payload(form_json, portal_info)

    print("[Agent 7] Submitting to portal (simulated)...")
    receipt = _submit_to_portal(submission_payload, portal_info)

    output = {
        "submitted":           receipt.get("success"),
        "tracking_number":     receipt.get("tracking_number"),
        "submitted_at":        receipt.get("submitted_at"),
        "portal":              receipt.get("portal"),
        "status":              receipt.get("status"),
        "expected_response_by": receipt.get("expected_response_by"),
        "confirmation_message": receipt.get("confirmation_message"),
        "submission_payload":  submission_payload,
        "portal_info":         portal_info,
        "data":                data,
        "form_json":           form_json,
    }

    print(f"  Submitted        : {output['submitted']}")
    print(f"  Tracking Number  : {output['tracking_number']}")
    print(f"  Expected Response: {output['expected_response_by']}")
    print("[Agent 7] Portal Automation Agent — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "data": {"insurer": "BlueCross BlueShield", "patient_mrn": "PT-88293"},
        "form_json": {
            "form_metadata": {"form_type": "Prior Authorization Request", "urgency": "routine"},
            "patient_information": {"name": "John R. Doe", "member_id": "BCBS-9900122", "date_of_birth": "1985-05-12"},
            "requesting_provider": {"name": "Dr. Sarah Jenkins", "specialty": "Orthopedic Surgery", "npi": "N/A"},
            "clinical_information": {
                "primary_diagnosis": "Lumbar Radiculopathy",
                "icd10_codes": ["M54.16"],
                "requested_procedure": "MRI Lumbar Spine Without Contrast",
                "cpt_codes": ["72148"],
                "date_of_service": "2024-03-15",
            },
        },
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k not in ("data", "form_json")}, indent=2))
