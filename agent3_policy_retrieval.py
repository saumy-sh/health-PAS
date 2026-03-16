"""
agent3_policy_retrieval.py
───────────────────────────
Agent 3 — Policy Retrieval + Requirement Generator

Role:
  Given:
    • policy_search_fields  — structured fields from Agent 2 (insurer, procedure, etc.)
    • documents             — prose summaries from Agent 1

  This agent:
    1. Identifies the specific treatment / procedure / medication for which
       pre-authorization is required.
    2. Retrieves the general pre-authorization policy criteria for that procedure.
    3. Outputs TWO separate requirement lists:

       document_requirements  — documents (or document content) that must be
                                submitted. Each entry describes what document
                                is needed and what information it must contain.

       medical_requirements   — clinical / medical conditions that must be met
                                (e.g. minimum symptom duration, step therapy
                                trials, lab result thresholds). These do NOT
                                refer to documents — they are facts about the
                                patient's medical situation.

  This agent does NOT check whether those requirements are fulfilled.
  That is the job of Agent 4 (documents) and Agent 5 (medical).

Input  : dict from Agent 2
Output : {
    "authorization_required": bool,
    "procedure_identified":   str,
    "document_requirements":  [{document_type, purpose, info_needed}],
    "medical_requirements":   [{requirement, description, threshold, importance}],
    "policy_notes":           str,
    "data":                   agent2 output (passed through)
  }
"""

import json
import re

from bedrock_client import invoke, LITE_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a senior prior-authorization policy specialist at an insurance company. "
            "Your job is to identify what procedure/treatment/medication needs pre-authorization "
            "and enumerate the requirements for that pre-authorization — both what documents must "
            "be submitted and what medical/clinical conditions must be met. "
            "Do NOT evaluate whether those requirements are currently fulfilled. "
            "Just list what is required. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_policy_prompt(fields: dict, documents: list) -> str:
    doc_summaries = "\n\n".join(
        f"[{d['document_type']}]\n{d['content']}"
        for d in documents
    )

    return f"""A patient is requesting prior authorization for a medical procedure/treatment/medication.

POLICY SEARCH FIELDS (from submitted documents):
  Insurer           : {fields.get('insurer_name')}
  Plan type         : {fields.get('plan_type')}
  Policy number     : {fields.get('policy_number')}
  Member ID         : {fields.get('member_id')}
  Group number      : {fields.get('group_number')}
  Diagnosis         : {fields.get('diagnosis')}
  Procedure / Tx    : {fields.get('procedure')}
  ICD-10 codes      : {fields.get('icd10_codes')}
  CPT codes         : {fields.get('cpt_codes')}
  Ordering physician: {fields.get('ordering_physician')} ({fields.get('physician_specialty')})

DOCUMENT SUMMARIES:
{doc_summaries}

Based on the above, determine:
1. Does this procedure/treatment require prior authorization? (Most imaging, surgeries,
   specialist procedures, and expensive medications do.)
2. What is the specific procedure/treatment/medication for which pre-auth is being requested?

Then enumerate ALL pre-authorization requirements in two categories:

DOCUMENT REQUIREMENTS — what documents (or document content) must be provided:
  For each, specify: what type of document, why it is needed, and exactly what
  information that document must contain.

MEDICAL REQUIREMENTS — clinical conditions that must be met (not document-related):
  Examples: minimum weeks of conservative treatment tried, step therapy medications
  tried first, specialist consultation required, minimum pain score, lab result
  thresholds, prior imaging required, age/weight criteria, etc.
  For each, specify the exact requirement, any numeric threshold, and why it matters.

Return JSON:
{{
  "authorization_required": true | false,
  "procedure_identified": "name of the procedure/treatment/medication requiring pre-auth",
  "document_requirements": [
    {{
      "document_type": "e.g. Doctor's Clinical Notes",
      "purpose": "why this document is required",
      "info_needed": "specific information this document must contain"
    }}
  ],
  "medical_requirements": [
    {{
      "requirement": "short name for this requirement",
      "description": "full description of what must be true",
      "threshold": "numeric or categorical threshold if applicable, else null",
      "importance": "required | recommended"
    }}
  ],
  "policy_notes": "any general notes about this insurer's pre-auth policy for this procedure"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON parser
# ─────────────────────────────────────────────────────────────────────────────

def _safe_parse(raw: str) -> dict:
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

    print(f"  [Agent3] ❌ JSON parse failed. Raw: {raw[:300]}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent2_output: dict) -> dict:
    print("\n[Agent 3] Policy Retrieval + Requirement Generator — START")

    fields    = agent2_output.get("policy_search_fields", {})
    documents = agent2_output.get("documents", [])

    print(f"  Insurer    : {fields.get('insurer_name')}")
    print(f"  Procedure  : {fields.get('procedure')}")
    print(f"  ICD-10     : {fields.get('icd10_codes')}")
    print(f"  Docs from Agent 1: {len(documents)}")

    print("[Agent 3] Calling Nova Lite — policy retrieval and requirement enumeration...")
    prompt   = _build_policy_prompt(fields, documents)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    raw = invoke(
        model_id=LITE_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.0,  # maximum determinism
    )

    result = _safe_parse(raw)

    if not result:
        print("[Agent 3] ⚠  Parse failed — using minimal fallback")
        result = {
            "authorization_required": True,
            "procedure_identified":   fields.get("procedure", "Unknown procedure"),
            "document_requirements":  [],
            "medical_requirements":   [],
            "policy_notes":           "Policy retrieval failed — manual review required.",
        }

    doc_reqs     = result.get("document_requirements", [])
    med_reqs     = result.get("medical_requirements", [])
    auth_required = result.get("authorization_required", True)

    print(f"  Auth required        : {auth_required}")
    print(f"  Procedure identified : {result.get('procedure_identified')}")
    print(f"  Document requirements: {len(doc_reqs)}")
    for r in doc_reqs:
        print(f"    📄 [{r.get('document_type')}] — {r.get('purpose', '')[:60]}")
    print(f"  Medical requirements : {len(med_reqs)}")
    for r in med_reqs:
        print(f"    🏥 [{r.get('importance', 'required')}] {r.get('requirement')} — {str(r.get('description', ''))[:60]}")

    output = {
        "authorization_required": auth_required,
        "procedure_identified":   result.get("procedure_identified"),
        "document_requirements":  doc_reqs,
        "medical_requirements":   med_reqs,
        "policy_notes":           result.get("policy_notes", ""),
        # Pass everything through for downstream agents
        "policy_search_fields":   fields,
        "documents":              documents,  # agent1 docs
    }

    print("[Agent 3] Policy Retrieval + Requirement Generator — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "ready": True,
        "policy_search_fields": {
            "patient_name":      "John R. Doe",
            "patient_dob":       "1985-05-12",
            "patient_mrn":       "PT-88293",
            "insurer_name":      "BlueCross BlueShield",
            "plan_type":         "PPO",
            "policy_number":     "BCBS-99001122",
            "member_id":         "BCBS-9900122",
            "group_number":      "8800221",
            "diagnosis":         "Lumbar Radiculopathy",
            "procedure":         "MRI Lumbar Spine Without Contrast",
            "icd10_codes":       ["M54.16"],
            "cpt_codes":         ["72148"],
            "ordering_physician": "Dr. Sarah Jenkins",
            "physician_specialty": "Orthopedic Surgery",
        },
        "documents": [
            {
                "document_type": "Doctor's Clinical Notes",
                "content": (
                    "Clinical notes from Dr. Sarah Jenkins at Metropolitan General Hospital. "
                    "Patient John R. Doe has been experiencing lower back pain radiating down "
                    "the right leg for 7 weeks. Diagnosis: Lumbar Radiculopathy (M54.16). "
                    "Positive SLR at 30 degrees. Prior treatments: NSAIDs for 4 weeks with "
                    "minimal relief, physical therapy for 3 weeks. Requesting MRI Lumbar "
                    "Spine Without Contrast (CPT 72148)."
                ),
            },
            {
                "document_type": "Insurance Card",
                "content": (
                    "BlueCross BlueShield PPO insurance card. Member: John R. Doe. "
                    "Member ID: BCBS-9900122. Policy: BCBS-99001122. Group: 8800221."
                ),
            },
        ],
    }
    result = run(mock_input)
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("documents", "policy_search_fields")},
        indent=2
    ))
