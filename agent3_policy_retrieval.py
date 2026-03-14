"""
agent3_policy_retrieval.py
───────────────────────────
Agent 3 — Policy Retrieval Agent (RAG)
• Queries insurer policy rules based on ICD-10 + CPT codes
• Simulates RAG over insurer policy documents (replace _retrieve_policy_chunks
  with a real vector-store / Bedrock Knowledge Base call in production)
• Uses Nova Lite to reason over retrieved policy text
• Returns prior-auth requirements specific to this claim

Input  : dict from Agent 2
Output : dict  { "policy_rules": {...}, "auth_required": bool, "requirements": [...] }
"""

import json

from bedrock_client import invoke, LITE_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Policy Knowledge Base (simulated)
# In production: replace with Bedrock Knowledge Base / OpenSearch / FAISS call
# ─────────────────────────────────────────────────────────────────────────────

# Simulated policy chunks keyed by (insurer_prefix, cpt_code)
POLICY_CHUNKS = {
    ("bluecross", "72148"): """
BLUECROSS BLUESHIELD PRIOR AUTHORIZATION POLICY — MRI LUMBAR SPINE (CPT 72148)
Effective: 2024-01-01

AUTHORIZATION REQUIRED: YES
Clinical Criteria:
1. Patient must have documented low back pain or radicular symptoms for ≥ 6 weeks.
2. Conservative treatment failure required: patient must have tried and failed at least ONE of:
   - NSAIDs for minimum 4 weeks, OR
   - Physical therapy for minimum 4 weeks, OR
   - Chiropractic care for minimum 6 weeks.
3. Clinical examination findings supporting radiculopathy (e.g., positive SLR, neurological deficit).
4. No prior lumbar MRI within 12 months unless significant clinical change.

Required Documentation:
- Completed prior authorization request form
- Office visit notes (within 60 days)
- Documentation of conservative treatment failure
- Referring physician NPI and specialty
- ICD-10 code(s) and CPT code(s)
- Estimated date of service

Approval Timeline: 3–5 business days (standard), 24 hours (urgent).
Valid Period: 90 days from approval date.
Co-pay: $50 specialist co-pay applies at time of service.
    """,
    ("bluecross", "99204"): """
BLUECROSS BLUESHIELD — OUTPATIENT SPECIALIST CONSULTATION (CPT 99204)
No prior authorization required for in-network specialist consultations.
In-network co-pay: $50.
Referral from PCP may be required depending on plan type.
    """,
    ("bluecross", "97161"): """
BLUECROSS BLUESHIELD — PHYSICAL THERAPY EVALUATION (CPT 97161)
Prior authorization required after initial evaluation.
Up to 12 PT visits per calendar year without authorization.
Beyond 12 visits: prior authorization required with functional progress notes.
    """,
    ("default", "72148"): """
STANDARD POLICY — MRI LUMBAR SPINE (CPT 72148)
Prior authorization typically required.
Standard clinical criteria: 6+ weeks of symptoms, conservative treatment failure.
Documentation: physician notes, treatment history, clinical exam findings.
    """,
}


def _retrieve_policy_chunks(insurer: str, cpt_codes: list, icd10_codes: list) -> str:
    """
    Retrieve relevant policy text for the given insurer + procedure codes.
    
    Production upgrade path:
      - Replace with boto3 call to Bedrock Knowledge Base:
        bedrock_agent_runtime.retrieve(knowledgeBaseId=KB_ID, retrievalQuery=...)
      - Or embed query + cosine search over policy PDF embeddings
    """
    insurer_key = insurer.lower().replace(" ", "").replace("-", "")[:10]

    # Normalise to 'bluecross' for BCBS variants
    if "bluecross" in insurer_key or "bcbs" in insurer_key:
        insurer_key = "bluecross"

    chunks = []
    for cpt in cpt_codes:
        key = (insurer_key, cpt)
        fallback = ("default", cpt)
        chunk = POLICY_CHUNKS.get(key) or POLICY_CHUNKS.get(fallback)
        if chunk:
            chunks.append(chunk.strip())

    if not chunks:
        chunks.append(
            f"No specific policy found for insurer '{insurer}' "
            f"and CPT codes {cpt_codes}. "
            "Standard prior authorization process applies. "
            "Contact insurer directly for criteria."
        )

    return "\n\n---\n\n".join(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a prior-authorization specialist with deep knowledge of "
            "insurance policy requirements. Analyze the retrieved policy documents "
            "and patient data to determine exact prior-authorization requirements. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_prompt(policy_text: str, patient_data: dict) -> str:
    return f"""You are reviewing a prior-authorization request. 
Analyze the policy documents and patient data below.

PATIENT DATA:
- Insurer       : {patient_data.get('insurer')}
- Diagnosis     : {patient_data.get('diagnosis')}
- ICD-10 codes  : {patient_data.get('icd10_codes')}
- Procedure     : {patient_data.get('procedure')}
- CPT codes     : {patient_data.get('cpt_codes')}
- Physician     : {patient_data.get('ordering_physician')} ({patient_data.get('physician_specialty')})
- Prior treatments tried: {patient_data.get('prior_treatments')}
- Symptom duration: {patient_data.get('symptom_duration_weeks')} weeks
- Clinical findings: {patient_data.get('clinical_findings')}

RETRIEVED POLICY DOCUMENTS:
{policy_text}

Based on the above, return a JSON object:
{{
  "authorization_required": true | false,
  "primary_cpt_requiring_auth": string,
  "cpt_auth_status": {{
    "<cpt_code>": "auth_required" | "no_auth_needed" | "conditional"
  }},
  "clinical_criteria_met": {{
    "symptom_duration_met": true | false,
    "conservative_treatment_met": true | false,
    "clinical_exam_met": true | false,
    "no_recent_imaging": true | false
  }},
  "required_documents": [string],
  "approval_timeline_days": integer,
  "validity_period_days": integer,
  "applicable_copay": number,
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
    dict with policy rules, auth requirements, and approval likelihood
    """
    print("\n[Agent 3] Policy Retrieval Agent — START")

    data = agent2_output.get("data", agent2_output)
    insurer    = data.get("insurer", "")
    cpt_codes  = data.get("cpt_codes", [])
    icd10_codes = data.get("icd10_codes", [])

    print(f"  Insurer   : {insurer}")
    print(f"  CPT codes : {cpt_codes}")
    print(f"  ICD-10    : {icd10_codes}")

    print("[Agent 3] Retrieving policy chunks...")
    policy_text = _retrieve_policy_chunks(insurer, cpt_codes, icd10_codes)
    print(f"  Policy text length: {len(policy_text)} chars")

    prompt = _build_prompt(policy_text, data)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    print("[Agent 3] Calling Nova Lite for policy reasoning...")
    raw = invoke(
        model_id=LITE_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=1000,
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
        policy_analysis = {"error": "Parse error", "raw": raw}

    output = {
        "authorization_required": policy_analysis.get("authorization_required", True),
        "policy_analysis":        policy_analysis,
        "retrieved_policy_text":  policy_text,
        "data":                   data,
    }

    print(f"  Auth required       : {output['authorization_required']}")
    print(f"  Approval likelihood : {policy_analysis.get('likelihood_of_approval')}")
    print(f"  Required docs       : {policy_analysis.get('required_documents')}")
    print("[Agent 3] Policy Retrieval Agent — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "data": {
            "insurer": "BlueCross BlueShield",
            "diagnosis": "Lumbar Radiculopathy",
            "icd10_codes": ["M54.16"],
            "procedure": "MRI Lumbar Spine Without Contrast",
            "cpt_codes": ["72148", "72148-26", "99204", "97161"],
            "ordering_physician": "Dr. Sarah Jenkins",
            "physician_specialty": "Orthopedic Surgery",
            "prior_treatments": ["NSAIDs 4 weeks minimal relief", "Physical Therapy 3 weeks"],
            "symptom_duration_weeks": 7,
            "clinical_findings": "Positive SLR at 30 degrees right side. Antalgic gait.",
        }
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
