"""
agent2_policy_checker.py
─────────────────────────
Agent 2 — Policy Search Information Extractor

Role:
  Given the list of document descriptions produced by Agent 1, extract the
  structured fields that are needed to search for the correct insurance policy
  in downstream processing:
    • Patient identity   : name, date of birth, MRN
    • Insurance identity : insurer name, plan type, policy number, member ID,
                           group number
    • Clinical summary   : diagnosis, procedure, ordering physician

  Also performs a readiness check: at minimum, an insurer name and at least one
  of (member_id, policy_number) must be present to proceed.

Input  : dict from Agent 1  →  { "documents": [{document_type, content}, ...] }
Output : {
    "ready":               bool,
    "policy_search_fields": { patient_name, patient_dob, ... },
    "missing_critical":    [field names that could not be found],
    "documents":           [agent1 documents — passed through for downstream agents]
  }
"""

import json
import re

from bedrock_client import invoke, MICRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a medical prior-authorization intake specialist. "
            "You receive natural-language summaries of submitted patient documents "
            "and extract the exact information needed to locate the correct "
            "insurance policy and prepare a pre-authorization request. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# LLM extraction
# ─────────────────────────────────────────────────────────────────────────────

def _build_extraction_prompt(documents: list) -> str:
    doc_block = ... # same as before
    return f"""The following are natural-language summaries of medical documents...

Extract and return ONLY this JSON:
{{
  "patient_name": "...",
  ...
  "procedure": "name of the procedure or treatment for which pre-auth is requested",
  "cpt_codes": ["list", "of", "CPT", "codes"],
  ...
}}

Examples:
- If a clinical note says "Requesting MRI Lumbar Spine Without Contrast (CPT 72148)", set procedure="MRI Lumbar Spine Without Contrast", cpt_codes=["72148"].
- If the note says "Patient needs colonoscopy (CPT 45378)", set procedure="Colonoscopy", cpt_codes=["45378"].
"""


def _extract_fields(documents: list) -> dict:
    prompt   = _build_extraction_prompt(documents)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    raw = invoke(
        model_id=MICRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=1000,
        temperature=0.1,
    )

    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Extract largest {...} block
        import re as _re
        m = _re.search(r'\{.*\}', text, _re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        print(f"  [Agent2] ❌ JSON parse error. Raw: {raw[:300]}")
        return {}


def _check_readiness(fields: dict) -> tuple:
    """
    Minimum requirements to search for a policy:
      - Insurance Provider (insurer_name)
      - Policy Number
      - Patient Name
      - Member ID or Patient DOB (as additional identification)

    For now, enforcing exactly Insurance Provider and Policy Number as requested.
    Returns (ready: bool, missing_info: list)
    """
    missing = []

    if not fields.get("insurer_name"):
        missing.append({
            "document_type": "Insurance Card / Member ID Information",
            "info_needed": "Insurance Provider (Insurer Name)",
            "reason": "The carrier name is required to determine which policy database to search.",
            "priority": "HIGH"
        })

    if not fields.get("policy_number"):
        missing.append({
            "document_type": "Insurance Card / Member ID Information",
            "info_needed": "Policy Number",
            "reason": "The policy number is required to uniquely identify the patient's coverage plan.",
            "priority": "HIGH"
        })

    if not fields.get("member_id"):
        missing.append({
            "document_type": "Insurance Card / Member ID Information",
            "info_needed": "Member ID",
            "reason": "The member ID is required for verification of unique coverage within the insurer's records.",
            "priority": "HIGH"
        })

    # Adding Patient Name as it is usually required for any search
    if not fields.get("patient_name"):
        missing.append({
            "document_type": "Patient Identification / Clinical Notes",
            "info_needed": "Patient Name",
            "reason": "The patient's name is required to verify the policy holder.",
            "priority": "HIGH"
        })

    ready = len(missing) == 0
    return ready, missing


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent1_output: dict) -> dict:
    print("\n[Agent 2] Policy Search Info Extractor — START")

    documents = agent1_output.get("documents", [])
    if not documents:
        print("  [Agent2] ⚠  No documents received from Agent 1")
        return {
            "ready": False,
            "policy_search_fields": {},
            "missing_critical": ["no documents provided"],
            "documents": [],
        }

    print(f"  Received {len(documents)} document(s) from Agent 1:")
    for d in documents:
        print(f"    • {d['document_type']}")

    print("[Agent 2] Extracting policy search fields via Nova Micro...")
    fields = _extract_fields(documents)

    ready, missing = _check_readiness(fields)

    print(f"  Insurer       : {fields.get('insurer_name')}")
    print(f"  Policy number : {fields.get('policy_number')}")
    print(f"  Member ID     : {fields.get('member_id')}")
    print(f"  Patient       : {fields.get('patient_name')}  DOB: {fields.get('patient_dob')}")
    print(f"  Diagnosis     : {fields.get('diagnosis')}")
    print(f"  Procedure     : {fields.get('procedure')}")
    print(f"  Ready         : {ready}")
    if missing:
        print(f"  Missing       : {missing}")

    output = {
        "ready":                ready,
        "policy_search_fields": fields,
        "missing_critical":     missing,
        # Pass agent1 documents through so downstream agents can use them
        "documents":            documents,
    }

    print("[Agent 2] Policy Search Info Extractor — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "documents": [
            {
                "document_type": "Insurance Card",
                "content": (
                    "This is an insurance card for BlueCross BlueShield PPO plan. "
                    "The member's name is John R. Doe, member ID BCBS-9900122, "
                    "policy number BCBS-99001122, group number 8800221. "
                    "Specialist copay $40, office visit copay $20."
                ),
            },
            {
                "document_type": "Doctor's Clinical Notes",
                "content": (
                    "Clinical notes from Dr. Sarah Jenkins (Orthopedic Surgery) at "
                    "Metropolitan General Hospital. Patient John R. Doe, DOB 1985-05-12, "
                    "MRN PT-88293. Diagnosis: Lumbar Radiculopathy (ICD-10 M54.16). "
                    "Clinical findings: positive SLR at 30 degrees, antalgic gait. "
                    "Requesting MRI Lumbar Spine Without Contrast (CPT 72148)."
                ),
            },
        ]
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k != "documents"}, indent=2))
