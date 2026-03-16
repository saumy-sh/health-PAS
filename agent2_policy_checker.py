"""
agent2_policy_checker.py
─────────────────────────
Agent 2 — Policy Search Information Extractor

Role:
  Given the list of document descriptions produced by Agent 1, extract the
  structured fields needed to search for the correct insurance policy.

  BLOCKING fields (pipeline cannot proceed without ALL of these):
    • insurer_name   — which insurance company to query
    • policy_number  — unique policy identifier
    • plan_type      — PPO / HMO / EPO / POS
    • member_id      — member identifier within the insurer's system

  If ANY of these four are missing, Agent 2 halts the pipeline and asks the
  user to upload a document (e.g. insurance card) that contains them.
  A clear, human-readable reason is given for each missing field.

  Non-blocking fields (extracted but do not halt pipeline):
    patient_name, patient_dob, patient_mrn, patient_gender,
    group_number, diagnosis, icd10_codes, procedure, cpt_codes,
    ordering_physician, physician_specialty, facility_name

Input  : { "documents": [{document_type, content}, ...] }
Output : {
    "ready":                bool,
    "policy_search_fields": { ... },
    "missing_critical":     [ {field, document_type, info_needed, reason, priority} ],
    "documents":            [ agent1 docs — passed through ]
  }
"""

import json
import re
from typing import Tuple, List

from bedrock_client import invoke, MICRO_MODEL_ID, LITE_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# The four fields that MUST be present to proceed.
# Format: (field_key, human_label, why_it_is_needed)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_POLICY_FIELDS = [
    (
        "insurer_name",
        "Insurance Company Name",
        (
            "We need to know which insurance company holds this policy so we can look up "
            "the correct pre-authorization rules and coverage requirements. Without the "
            "insurer name we cannot determine which policy database to query or which "
            "criteria apply to this patient's plan."
        ),
    ),
    (
        "policy_number",
        "Policy Number",
        (
            "The policy number uniquely identifies the patient's specific coverage plan "
            "within the insurance company's system. It is required to retrieve the exact "
            "benefits, authorisation criteria, and coverage limits that apply to this "
            "patient. Without it we cannot distinguish between multiple plans offered by "
            "the same insurer."
        ),
    ),
    (
        "plan_type",
        "Plan Type (e.g. PPO, HMO, EPO, POS)",
        (
            "The plan type determines the network rules and referral requirements for "
            "pre-authorization. For example, HMO plans require a primary care physician "
            "referral while PPO plans generally do not. Without this information we cannot "
            "apply the correct authorisation workflow or determine whether a specialist "
            "referral is needed."
        ),
    ),
    (
        "member_id",
        "Member ID",
        (
            "The member ID is the insurer's unique identifier for this individual within "
            "the plan. It is required to verify that the patient has active coverage, "
            "confirm the correct member record is being accessed, and correctly link the "
            "pre-authorization request to the right policy holder in the insurer's system."
        ),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a medical prior-authorization intake specialist. "
            "You receive natural-language summaries of submitted patient documents "
            "and extract the exact information needed to locate the correct insurance policy. "
            "Be thorough — search ALL provided documents carefully for every field. "
            "Insurance details may appear in insurance cards, patient info sheets, or clinical notes. "
            "Return ONLY a valid JSON object — no markdown, no preamble, no explanation."
        )
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Extraction prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_extraction_prompt(documents: list) -> str:
    doc_block = "\n\n".join(
        f"=== Document {i+1}: {d.get('document_type', 'Unknown')} ===\n{d.get('content', '').strip()}"
        for i, d in enumerate(documents)
    )

    return f"""You are given the full text content of multiple medical/insurance documents.
Read EVERY document carefully and extract the fields listed below.

DOCUMENTS:
{doc_block}

EXTRACTION RULES:
1. Search ALL documents — insurance details may be in an insurance card, patient info sheet, or clinical notes.
2. For insurer_name: look for company names like "BlueCross BlueShield", "UnitedHealthcare", "Aetna", "Cigna", "NovaCare", etc.
3. For policy_number: look near labels like "Policy #", "Policy No", "Policy Number", "Pol No".
4. For plan_type: look for "PPO", "HMO", "EPO", "POS" near the insurer name or on an insurance card.
5. For member_id: look near labels like "Member ID", "ID#", "Member No", "Subscriber ID", "Member Number".
6. For group_number: look near labels like "Group #", "Group No", "Group Number".
7. For CPT codes: 5-digit numeric codes. For ICD-10: letter + digits (e.g. M54.16).
8. If a value is truly not present in ANY document, use null — do NOT guess or fabricate.

Return ONLY this exact JSON (no markdown, no extra text):
{{
  "patient_name":        "Full patient name or null",
  "patient_dob":         "YYYY-MM-DD or null",
  "patient_mrn":         "Medical Record Number or null",
  "patient_gender":      "Male/Female/Other or null",
  "insurer_name":        "Insurance company name or null",
  "plan_type":           "PPO/HMO/EPO/POS or null",
  "policy_number":       "Policy number string or null",
  "member_id":           "Member ID string or null",
  "group_number":        "Group number string or null",
  "diagnosis":           "Human-readable diagnosis or null",
  "icd10_codes":         ["list of ICD-10 codes"],
  "procedure":           "Name of procedure/treatment being requested or null",
  "cpt_codes":           ["list of CPT codes"],
  "ordering_physician":  "Full name of ordering physician or null",
  "physician_specialty": "Specialty or null",
  "facility_name":       "Hospital or clinic name or null"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON parser
# ─────────────────────────────────────────────────────────────────────────────

def _safe_parse_json(raw: str) -> dict:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for part in text.split("```")[1:]:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    print(f"  [Agent2] ❌ JSON parse failed. Raw (first 400): {raw[:400]}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Field extraction with model fallback
# ─────────────────────────────────────────────────────────────────────────────

def _extract_fields(documents: list) -> dict:
    prompt   = _build_extraction_prompt(documents)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    print("  [Agent2] Calling Nova Micro for field extraction...")
    raw = invoke(
        model_id=MICRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=1500,
        temperature=0.0,
    )

    if raw.startswith("ERROR:") or not raw.strip():
        print("  [Agent2] ⚠ Nova Micro failed — falling back to Nova Lite...")
        raw = invoke(
            model_id=LITE_MODEL_ID,
            messages=messages,
            system=SYSTEM_PROMPT,
            max_tokens=1500,
            temperature=0.0,
        )

    fields = _safe_parse_json(raw)

    if not fields:
        print("  [Agent2] ⚠ Extraction returned empty dict.")
        return {}

    # Normalise list fields
    for list_field in ("icd10_codes", "cpt_codes"):
        val = fields.get(list_field)
        if val is None:
            fields[list_field] = []
        elif isinstance(val, str):
            fields[list_field] = [val.strip()] if val.strip() else []
        elif not isinstance(val, list):
            fields[list_field] = []

    # Strip empty strings → None
    for key, val in fields.items():
        if isinstance(val, str):
            fields[key] = val.strip() or None

    return fields


# ─────────────────────────────────────────────────────────────────────────────
# Readiness check
# ─────────────────────────────────────────────────────────────────────────────

def _check_readiness(fields: dict) -> Tuple[bool, List[dict]]:
    """
    Checks only the four required policy fields.
    Returns (ready, missing_critical).
    """
    missing = []

    for field_key, human_label, reason in REQUIRED_POLICY_FIELDS:
        if not fields.get(field_key):
            missing.append({
                "field":         field_key,
                "document_type": "Insurance Card / Member ID Card",
                "info_needed":   human_label,
                "reason":        reason,
                "priority":      "HIGH",
            })

    return len(missing) == 0, missing


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent1_output: dict) -> dict:
    print("\n[Agent 2] Policy Search Info Extractor — START")

    documents = agent1_output.get("documents", [])
    if not documents:
        print("  [Agent2] ⚠  No documents received from Agent 1")
        missing = [
            {
                "field":         fk,
                "document_type": "Insurance Card / Member ID Card",
                "info_needed":   hl,
                "reason":        r,
                "priority":      "HIGH",
            }
            for fk, hl, r in REQUIRED_POLICY_FIELDS
        ]
        return {
            "ready":                False,
            "policy_search_fields": {},
            "missing_critical":     missing,
            "documents":            [],
        }

    print(f"  Received {len(documents)} document(s) from Agent 1:")
    for d in documents:
        print(f"    • [{d.get('document_type', 'Unknown')}]")

    print("[Agent 2] Extracting policy search fields...")
    fields = _extract_fields(documents)
    ready, missing = _check_readiness(fields)

    # ── Summary print ─────────────────────────────────────────────────────────
    print(f"\n  ── Extraction Results ──────────────────────────────────────")
    print(f"  insurer_name   : {fields.get('insurer_name')}")
    print(f"  policy_number  : {fields.get('policy_number')}")
    print(f"  plan_type      : {fields.get('plan_type')}")
    print(f"  member_id      : {fields.get('member_id')}")
    print(f"  group_number   : {fields.get('group_number')}")
    print(f"  patient_name   : {fields.get('patient_name')}  DOB: {fields.get('patient_dob')}")
    print(f"  diagnosis      : {fields.get('diagnosis')}  {fields.get('icd10_codes')}")
    print(f"  procedure      : {fields.get('procedure')}  CPT: {fields.get('cpt_codes')}")
    print(f"  physician      : {fields.get('ordering_physician')} ({fields.get('physician_specialty')})")
    print(f"  Ready          : {'✅' if ready else '❌'}  {ready}")
    if missing:
        print(f"\n  ── Missing Required Fields ─────────────────────────────────")
        for m in missing:
            print(f"    🔴 {m['info_needed']}")
            print(f"       Reason: {m['reason'][:100]}...")
    print(f"  ────────────────────────────────────────────────────────────\n")

    print("[Agent 2] Policy Search Info Extractor — DONE\n")
    return {
        "ready":                ready,
        "policy_search_fields": fields,
        "missing_critical":     missing,
        "documents":            documents,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: All four required fields present → should be ready")
    print("=" * 60)
    r1 = run({
        "documents": [
            {
                "document_type": "Insurance Card",
                "content": (
                    "BlueCross BlueShield PPO. Member: John R. Doe. "
                    "Member ID: BCBS-9900122. Policy Number: BCBS-99001122. Group: 8800221."
                ),
            },
        ]
    })
    print(f"  ready={r1['ready']}  missing={[m['info_needed'] for m in r1['missing_critical']]}\n")

    print("=" * 60)
    print("TEST 2: Only clinical note — all four fields missing → should block")
    print("=" * 60)
    r2 = run({
        "documents": [
            {
                "document_type": "Clinical Note",
                "content": (
                    "Patient John R. Doe. DOB 1985-05-12. MRN PT-88293. "
                    "Lumbar Radiculopathy M54.16. Requesting MRI Lumbar Spine (CPT 72148). "
                    "Dr. Sarah Jenkins, Orthopedic Surgery."
                ),
            }
        ]
    })
    print(f"  ready={r2['ready']}")
    for m in r2["missing_critical"]:
        print(f"\n  🔴 {m['info_needed']}")
        print(f"     {m['reason']}")

    print("\n" + "=" * 60)
    print("TEST 3: Insurance card present but plan_type missing")
    print("=" * 60)
    r3 = run({
        "documents": [
            {
                "document_type": "Insurance Card",
                "content": (
                    "BlueCross BlueShield. Member: John R. Doe. "
                    "Member ID: BCBS-9900122. Policy Number: BCBS-99001122."
                    # No "PPO" / "HMO" mentioned
                ),
            },
        ]
    })
    print(f"  ready={r3['ready']}  missing={[m['info_needed'] for m in r3['missing_critical']]}")
