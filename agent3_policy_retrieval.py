"""
agent3_policy_retrieval.py
───────────────────────────
Agent 3 — Policy Retrieval + Requirement Generator

Role:
  Given:
    • policy_search_fields  — structured fields from Agent 2 (insurer, procedure, etc.)
    • documents             — prose summaries from Agent 1

  This agent:
    1. First looks up the requirements from a local knowledge base (policy_requirements.json).
    2. If not found in the KB, falls back to the LLM to identify the specific treatment/procedure
       and retrieve pre-authorization policy criteria.
    3. Outputs TWO separate requirement lists:
       document_requirements  — documents (or document content) that must be submitted.
       medical_requirements   — clinical / medical conditions that must be met.

Input  : dict from Agent 2
Output : {
    "authorization_required": bool,
    "procedure_identified":   str,
    "document_requirements":  [{document_type, purpose, info_needed}],
    "medical_requirements":   [{requirement, description, threshold, importance}],
    "policy_notes":           str,
    "policy_search_fields":   dict (passed through),
    "documents":              list (passed through)
  }
"""

import json
import re
import os
from bedrock_client import invoke, LITE_MODEL_ID

# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base Initialization
# ─────────────────────────────────────────────────────────────────────────────

# Load knowledge base (from a local file or potentially S3)
KB_FILE = "policy_requirements.json"
POLICY_KB = {}

try:
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r") as f:
            POLICY_KB = json.load(f)
        print(f"[Agent 3] Loaded knowledge base from {KB_FILE} ({len(POLICY_KB)} insurers)")
    else:
        print(f"[Agent 3] ⚠  {KB_FILE} not found. Knowledge base lookup will be skipped.")
except Exception as e:
    print(f"[Agent 3] ❌ Error loading knowledge base: {e}")

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

def _build_policy_prompt(fields: dict) -> str:
    return f"""A patient is requesting prior authorization for a medical procedure/treatment/medication.

POLICY SEARCH FIELDS (extracted from submitted documents):
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

Based on the above structured data, determine:
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
# Helper Functions
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

    print(f"  [Agent 3] ❌ JSON parse failed. Raw: {raw[:300]}")
    return {}

def llm_fallback(fields: dict) -> dict:
    """Fallback logic to retrieve policy requirements using LLM."""
    print("[Agent 3] Calling Nova Lite — policy retrieval using Agent 2 fields...")
    prompt = _build_policy_prompt(fields)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    try:
        raw = invoke(
            model_id=LITE_MODEL_ID,
            messages=messages,
            system=SYSTEM_PROMPT,
            max_tokens=2000,
            temperature=0.0,
        )
        result = _safe_parse(raw)
    except Exception as e:
        print(f"  [Agent 3] ❌ Bedrock invocation failed: {e}")
        result = {}

    if not result:
        print("[Agent 3] ⚠  Parse failed — using minimal fallback")
        result = {
            "authorization_required": True,
            "procedure_identified": fields.get("procedure", "Unknown procedure"),
            "document_requirements": [],
            "medical_requirements": [],
            "policy_notes": "Policy retrieval failed — manual review required.",
        }
    else:
        if "policy_notes" not in result:
             result["policy_notes"] = "Retrieved via LLM analysis of known policy trends."

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent2_output: dict) -> dict:
    print("\n[Agent 3] Policy Retrieval + Requirement Generator — START")

    fields = agent2_output.get("policy_search_fields", {})
    insurer = fields.get("insurer_name")
    procedure = fields.get("procedure")
    cpt_codes = fields.get("cpt_codes", [])

    print(f"  Insurer    : {insurer}")
    print(f"  Procedure  : {procedure}")
    print(f"  CPT Codes  : {cpt_codes}")

    # 1. Try to match by insurer + procedure name
    proc_data = None
    if insurer in POLICY_KB:
        proc_data = POLICY_KB[insurer]["procedures"].get(procedure)
        
        # 2. Optionally also match by CPT code if procedure name not found
        if not proc_data and cpt_codes:
            for proc_name, data in POLICY_KB[insurer]["procedures"].items():
                kb_cpt_codes = data.get("cpt_codes", [])
                if any(code in kb_cpt_codes for code in cpt_codes):
                    proc_data = data
                    procedure = proc_name
                    print(f"  [Agent 3] Match found by CPT code in KB: {procedure}")
                    break

    if proc_data:
        print(f"  [Agent 3] ✅ Policy found in knowledge base for insurer '{insurer}'")
        result = {
            "authorization_required": proc_data.get("requires_auth", True),
            "procedure_identified": procedure,
            "document_requirements": proc_data.get("document_requirements", []),
            "medical_requirements": proc_data.get("medical_requirements", []),
            "policy_notes": f"Retrieved from {insurer} knowledge base"
        }
    else:
        # 3. Fallback to LLM
        print(f"  [Agent 3] ℹ  No exact match in KB for {insurer}/{procedure}. Falling back to LLM...")
        result = llm_fallback(fields)

    # Add pass-through data and final logging
    result.update({
        "policy_search_fields": fields,
        "documents": agent2_output.get("documents", [])
    })

    doc_reqs = result.get("document_requirements", [])
    med_reqs = result.get("medical_requirements", [])
    print(f"  Auth required        : {result.get('authorization_required')}")
    print(f"  Procedure identified : {result.get('procedure_identified')}")
    print(f"  Document requirements: {len(doc_reqs)}")
    print(f"  Medical requirements : {len(med_reqs)}")
    print("[Agent 3] Policy Retrieval + Requirement Generator — DONE\n")
    
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "ready": True,
        "policy_search_fields": {
            "patient_name":      "John R. Doe",
            "patient_dob":       "1985-05-12",
            "insurer_name":      "BlueCross BlueShield",
            "procedure":         "MRI Lumbar Spine Without Contrast",
            "cpt_codes":         ["72148"],
        },
        "documents": [
            {"document_type": "Test Doc", "content": "Test content"}
        ],
    }
    # Test KB match
    print("--- Testing KB Match ---")
    res1 = run(mock_input)
    
    # Test Fallback
    print("\n--- Testing Fallback (Unknown Insurer) ---")
    mock_input["policy_search_fields"]["insurer_name"] = "UnknownHealth"
    res2 = run(mock_input)
