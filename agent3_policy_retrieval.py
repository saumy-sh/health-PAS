"""
agent3_policy_retrieval.py  (RELIABILITY FIX)
───────────────────────────────────────────────
Agent 3 — Policy Retrieval + Document Requirements

ROOT CAUSE OF PREVIOUS UNRELIABILITY:
  The old Agent 3 asked an LLM to freely generate a "required_documents" list.
  With no grounding, it hallucinated vague requirements like "Patient's medical
  history" or "Imaging results" even when those weren't needed — and it did so
  DIFFERENTLY on each run, causing non-deterministic pipeline halts.

THE FIX — Two-layer approach:
  ① DETERMINISTIC LAYER (always runs first):
     A hardcoded `PREAUTH_CHECKLIST` encodes the 7 pre-authorization steps
     from the clinical guidelines. Each checklist item maps to:
       - which document type satisfies it
       - which data fields prove it's satisfied
       - what to tell the user if it's missing
     This layer never changes between runs for the same input documents.

  ② LLM LAYER (used only for policy criteria assessment):
     The LLM is now given a STRICT prompt that says:
       "Only mark a document as missing if it is NOT covered by any
        of the confirmed-present document types listed below."
     The LLM's job is narrowed to: assess clinical criteria met/not met.
     It can NO LONGER invent new document requirements.

  ③ Agent 3 output now includes `canonical_requirements` — the deterministic
     checklist result — which Agent 4 uses as its authoritative source of
     truth instead of the LLM's `required_documents`.
"""

import json
from bedrock_client import invoke, LITE_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC PRE-AUTH CHECKLIST
# Based on the 7 standard pre-authorization steps:
#   1. Insurance information
#   2. Procedure / treatment codes
#   3. Clinical rationale (ICD-10, diagnosis, findings)
#   4. Past treatments and outcomes
#   5. Expedited review evidence (if needed)
#   6. Supporting documentation
#   7. Completeness check
#
# Each entry defines:
#   step          — human-readable step name
#   required      — is this always required for pre-auth?
#   satisfied_by  — list of document_types that can satisfy this requirement
#   proof_fields  — fields in those documents that must have real values
#   missing_msg   — what to tell the user if not satisfied
#   document_type — the canonical document type to request if missing
# ─────────────────────────────────────────────────────────────────────────────

PREAUTH_CHECKLIST = [
    # ── Step 1: Patient Insurance Information ─────────────────────────────────
    {
        "step":         "1_insurance_info",
        "label":        "Patient Insurance Information",
        "description":  "Patient demographics, member ID, policy number, plan type",
        "required":     True,
        "satisfied_by": ["Insurance Card", "Patient Information Sheet"],
        "proof_fields": {
            "Insurance Card":            ["insurer_name", "policy_number", "member_id"],
            "Patient Information Sheet": ["patient_name", "dob"],
        },
        "missing_msg": (
            "Insurance card is missing or incomplete. "
            "Please provide a copy of the patient's insurance card showing: "
            "insurer name, policy number, and member ID."
        ),
        "document_type": "Insurance Card",
    },
    {
        "step":         "1_patient_demographics",
        "label":        "Patient Demographics",
        "description":  "Full name, date of birth, address, MRN",
        "required":     True,
        "satisfied_by": ["Patient Information Sheet", "Doctor Notes", "Insurance Card"],
        "proof_fields": {
            "Patient Information Sheet": ["patient_name", "dob"],
            "Doctor Notes":             ["ordering_physician"],
            "Insurance Card":           ["insurer_name"],
        },
        "missing_msg": (
            "Patient information sheet is missing. "
            "Please provide a form with patient name, date of birth, and address."
        ),
        "document_type": "Patient Information Sheet",
    },

    # ── Step 2: Procedure / Treatment Codes ──────────────────────────────────
    {
        "step":         "2_procedure_codes",
        "label":        "Procedure Codes (CPT/HCPCS)",
        "description":  "CPT or HCPCS code for the requested procedure",
        "required":     True,
        "satisfied_by": ["Doctor Notes", "Medical Pretreatment Estimate"],
        "proof_fields": {
            "Doctor Notes":                  ["requested_procedure", "cpt_codes"],
            "Medical Pretreatment Estimate": ["total_estimated_cost", "cpt_codes"],
        },
        "missing_msg": (
            "No CPT procedure code found. "
            "The doctor notes or pretreatment estimate must include the CPT code "
            "for the requested procedure."
        ),
        "document_type": "Doctor Notes",
    },

    # ── Step 3: Clinical Rationale ────────────────────────────────────────────
    {
        "step":         "3_diagnosis_icd10",
        "label":        "Primary Diagnosis with ICD-10 Code",
        "description":  "ICD-10 diagnosis code and primary diagnosis description",
        "required":     True,
        "satisfied_by": ["Doctor Notes", "Physician Referral"],
        "proof_fields": {
            "Doctor Notes":      ["diagnosis", "icd10_codes"],
            "Physician Referral": ["reason_for_referral"],
        },
        "missing_msg": (
            "ICD-10 diagnosis code is missing. "
            "Please ask the ordering physician to include the ICD-10 code "
            "in the clinical notes."
        ),
        "document_type": "Doctor Notes",
    },
    {
        "step":         "3_clinical_findings",
        "label":        "Clinical Examination Findings",
        "description":  "Documented physical examination findings supporting the diagnosis",
        "required":     True,
        "satisfied_by": ["Doctor Notes"],
        "proof_fields": {
            "Doctor Notes": ["clinical_findings"],
        },
        "missing_msg": (
            "Clinical examination findings are not documented. "
            "Please ask the physician to include examination findings "
            "(e.g. SLR test results, neurological exam) in the clinical notes."
        ),
        "document_type": "Doctor Notes",
    },

    # ── Step 4: Past Treatments and Outcomes ──────────────────────────────────
    {
        "step":         "4_prior_treatments",
        "label":        "Prior Treatments / Step Therapy Documentation",
        "description":  "List of prior treatments tried, duration, and outcomes",
        "required":     True,
        "satisfied_by": ["Doctor Notes", "Physician Referral", "Prior Treatment Documentation"],
        "proof_fields": {
            "Doctor Notes":                   ["prior_treatments"],
            "Physician Referral":             ["reason_for_referral"],
            "Prior Treatment Documentation":  ["prior_treatments"],
        },
        "missing_msg": (
            "Prior treatment history is not documented. "
            "Clinical notes or a referral letter must list the treatments tried "
            "(e.g. NSAIDs, physical therapy), how long, and why they were insufficient."
        ),
        "document_type": "Doctor Notes",
    },

    # ── Step 5: Expedited Review (conditional) ────────────────────────────────
    {
        "step":         "5_expedited_review",
        "label":        "Expedited Review Justification (if urgent)",
        "description":  "Documentation of urgency if expedited review is requested",
        "required":     False,   # only required if urgency flag is set
        "satisfied_by": ["Doctor Notes"],
        "proof_fields": {
            "Doctor Notes": ["clinical_findings"],
        },
        "missing_msg": (
            "Expedited review was requested but no urgency justification was found. "
            "Please include a statement explaining why delayed treatment would cause harm."
        ),
        "document_type": "Doctor Notes",
    },

    # ── Step 6: Supporting Documentation ─────────────────────────────────────
    {
        "step":         "6_cost_estimate",
        "label":        "Pretreatment Cost Estimate",
        "description":  "Estimated cost with CPT codes and line items",
        "required":     True,
        "satisfied_by": ["Medical Pretreatment Estimate"],
        "proof_fields": {
            "Medical Pretreatment Estimate": ["total_estimated_cost"],
        },
        "missing_msg": (
            "A pretreatment cost estimate is required. "
            "Please obtain an itemized estimate from the treating facility "
            "that includes CPT codes and estimated costs."
        ),
        "document_type": "Medical Pretreatment Estimate",
    },
    {
        "step":         "6_ordering_physician",
        "label":        "Ordering Physician Credentials",
        "description":  "Physician name, NPI, specialty, and contact",
        "required":     True,
        "satisfied_by": ["Doctor Notes", "Physician Referral", "Prescription"],
        "proof_fields": {
            "Doctor Notes":      ["ordering_physician", "physician_specialty"],
            "Physician Referral": ["referring_physician"],
            "Prescription":      ["prescribing_physician"],
        },
        "missing_msg": (
            "Ordering physician information is missing. "
            "Clinical notes must include the physician's name, NPI, and specialty."
        ),
        "document_type": "Doctor Notes",
    },

    # ── Step 7: Additional clinical support (procedure-specific) ─────────────
    {
        "step":         "6_lab_results",
        "label":        "Lab Results (if clinically relevant)",
        "description":  "Recent lab work supporting the clinical picture",
        "required":     False,   # nice-to-have, not always mandatory
        "satisfied_by": ["Lab Report"],
        "proof_fields": {
            "Lab Report": ["test_name", "results"],
        },
        "missing_msg": (
            "Lab results referenced in clinical notes are not attached. "
            "If labs were ordered, please include the lab report."
        ),
        "document_type": "Lab Report",
    },
    {
        "step":         "6_prescription",
        "label":        "Prescription Record (for medication trials)",
        "description":  "Prescription evidence of medication trial",
        "required":     False,   # required only if step therapy includes medications
        "satisfied_by": ["Prescription", "Doctor Notes"],
        "proof_fields": {
            "Prescription": ["medication_name", "duration"],
            "Doctor Notes": ["prior_treatments"],
        },
        "missing_msg": (
            "Prescription record for the medication trial is missing. "
            "Please attach the prescription for NSAIDs or other medications "
            "that were part of the required step therapy."
        ),
        "document_type": "Prescription",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic checklist evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_checklist(data: dict) -> dict:
    """
    Evaluate every checklist item against the ACTUAL extracted document content.
    This is 100% deterministic — same input always gives same output.

    Returns:
      satisfied       : list of satisfied steps
      missing_required: list of required steps not satisfied (pipeline blockers)
      missing_optional: list of optional steps not satisfied (warnings only)
      all_required_met: bool
    """
    document_content_map = data.get("document_content_map", {})
    documents_present    = data.get("documents_present", {})

    # Build a flat lookup of what document types are present and their content
    # Use document_content_map if available (new Agent 1), else fall back
    available_doc_types = set(document_content_map.keys())
    if not available_doc_types:
        # Fallback from documents list
        available_doc_types = {
            doc.get("document_type")
            for doc in data.get("documents", [])
        }

    satisfied        = []
    missing_required = []
    missing_optional = []

    for item in PREAUTH_CHECKLIST:
        step   = item["step"]
        label  = item["label"]
        req    = item["required"]

        # Check if any satisfying document type is present AND has the proof fields
        step_satisfied = False
        satisfied_by_doc = None

        for doc_type in item["satisfied_by"]:
            if doc_type not in available_doc_types:
                continue

            # Document is present — now check if proof fields have real values
            content        = document_content_map.get(doc_type, {})
            required_fields = item["proof_fields"].get(doc_type, [])

            if not required_fields:
                # No field requirements — presence of the doc is enough
                step_satisfied   = True
                satisfied_by_doc = doc_type
                break

            # Check that at least the first critical field has a real value
            # (We don't require ALL fields — just that the key content is there)
            fields_present = [
                f for f in required_fields
                if content.get(f) and content.get(f) != []
                   and str(content.get(f)).strip() not in ("", "null", "None")
            ]

            # Satisfied if majority of proof fields are present
            if len(fields_present) >= max(1, len(required_fields) // 2):
                step_satisfied   = True
                satisfied_by_doc = doc_type
                break

        if step_satisfied:
            satisfied.append({
                "step":         step,
                "label":        label,
                "satisfied_by": satisfied_by_doc,
                "required":     req,
            })
        elif req:
            missing_required.append({
                "step":          step,
                "label":         label,
                "required":      True,
                "satisfying_docs": item["satisfied_by"],
                "document_type": item["document_type"],
                "description":   item["description"],
                "missing_msg":   item["missing_msg"],
            })
        else:
            missing_optional.append({
                "step":          step,
                "label":         label,
                "required":      False,
                "satisfying_docs": item["satisfied_by"],
                "document_type": item["document_type"],
                "description":   item["description"],
                "missing_msg":   item["missing_msg"],
            })

    return {
        "satisfied":          satisfied,
        "missing_required":   missing_required,
        "missing_optional":   missing_optional,
        "all_required_met":   len(missing_required) == 0,
        "satisfied_count":    len(satisfied),
        "total_required":     sum(1 for i in PREAUTH_CHECKLIST if i["required"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt — narrowed to clinical criteria ONLY, cannot invent doc requirements
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a senior prior-authorization policy specialist. "
            "Your ONLY job in this call is to assess whether the clinical criteria "
            "for the requested procedure are met. "
            "Do NOT generate new document requirements — document checking is handled separately. "
            "Be specific and clinically precise. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_criteria_prompt(data: dict, checklist_result: dict) -> str:
    """
    Narrowed LLM prompt: assess clinical criteria only.
    The list of confirmed-present documents is injected so the LLM
    cannot hallucinate requirements for documents already provided.
    """
    confirmed_present = [s["satisfied_by"] for s in checklist_result["satisfied"]]
    confirmed_missing = [m["label"] for m in checklist_result["missing_required"]]

    return f"""Assess the clinical criteria for this prior-authorization request.

CLAIM DETAILS:
  Insurer       : {data.get('insurer')}
  Diagnosis     : {data.get('diagnosis')} ({data.get('icd10')})
  Procedure     : {data.get('procedure')} (CPT: {data.get('cpt')})
  Physician     : {data.get('ordering_physician')} — {data.get('physician_specialty')}
  Symptom weeks : {data.get('symptom_duration_weeks')}
  Pain score    : {data.get('pain_score')}/10
  Findings      : {data.get('clinical_findings')}
  Prior Tx      : {json.dumps(data.get('prior_treatments', []))}
  Medications   : {json.dumps(data.get('prescribed_medications', []))}

DOCUMENTS CONFIRMED PRESENT (do NOT request these — they are already submitted):
  {json.dumps(confirmed_present)}

DOCUMENT STEPS NOT YET SATISFIED (already identified by document checker):
  {json.dumps(confirmed_missing)}

YOUR TASK — assess ONLY clinical compliance criteria:
1. Is the minimum symptom duration met for this procedure?
2. Is step therapy / conservative treatment documented and sufficient?
3. Are clinical examination findings adequate?
4. Is there a medication trial documented?
5. Is specialist referral required and present?

IMPORTANT: Do NOT add any new document requirements. Document requirements are
already handled. Only assess whether clinical evidence in the submitted documents
meets the medical necessity criteria for this procedure.

Return JSON:
{{
  "authorization_required": true,
  "primary_cpt_requiring_auth": string,
  "standard_clinical_criteria": {{
    "minimum_symptom_duration_weeks": integer,
    "minimum_symptom_duration_met": true | false,
    "actual_symptom_duration_weeks": integer,
    "step_therapy_required": true | false,
    "step_therapy_steps_required": [string],
    "step_therapy_steps_completed": [string],
    "step_therapy_met": true | false,
    "step_therapy_gaps": [string],
    "clinical_exam_required": true | false,
    "clinical_exam_met": true | false,
    "specialist_referral_required": true | false,
    "specialist_referral_met": true | false,
    "medication_trial_required": true | false,
    "medication_trial_details": string,
    "medication_trial_met": true | false
  }},
  "criteria_met": [string],
  "criteria_not_met": [string],
  "likelihood_of_approval": "high" | "medium" | "low",
  "policy_notes": string,
  "reasoning": string
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent2_output: dict) -> dict:
    print("\n[Agent 3] Policy Retrieval Agent — START")

    data = agent2_output.get("data", agent2_output)

    print(f"  Insurer   : {data.get('insurer')}")
    print(f"  CPT codes : {data.get('cpt_codes')}")
    print(f"  ICD-10    : {data.get('icd10_codes')}")

    # ── Step 1: DETERMINISTIC checklist evaluation ────────────────────────────
    print("[Agent 3] Running deterministic pre-auth checklist...")
    checklist_result = _evaluate_checklist(data)

    print(f"  Checklist: {checklist_result['satisfied_count']}/{checklist_result['total_required']} required steps satisfied")
    for s in checklist_result["satisfied"]:
        print(f"    ✅ {s['label']}  (via {s['satisfied_by']})")
    for m in checklist_result["missing_required"]:
        print(f"    ❌ {m['label']}  — {m['missing_msg'][:60]}...")
    for m in checklist_result["missing_optional"]:
        print(f"    ⚠️  {m['label']}  (optional — not blocking)")

    # ── Step 2: LLM clinical criteria assessment (narrowed scope) ────────────
    print("[Agent 3] Calling Nova Lite for clinical criteria assessment...")
    prompt   = _build_criteria_prompt(data, checklist_result)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    raw = invoke(
        model_id=LITE_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=1000,
        temperature=0.0,   # zero temp = maximum determinism
    )

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        policy_analysis = json.loads(text.strip())
    except json.JSONDecodeError:
        print("[Agent 3] ⚠  JSON parse error — using checklist-only result")
        policy_analysis = {
            "authorization_required":   True,
            "primary_cpt_requiring_auth": data.get("cpt"),
            "standard_clinical_criteria": {},
            "criteria_met":             [s["label"] for s in checklist_result["satisfied"]],
            "criteria_not_met":         [m["label"] for m in checklist_result["missing_required"]],
            "likelihood_of_approval":   "medium",
            "policy_notes":             "LLM criteria assessment failed — checklist used.",
            "reasoning":                raw,
        }

    # ── Step 3: Build required_documents from DETERMINISTIC checklist ─────────
    # This replaces the LLM-generated required_documents list entirely.
    # It is grounded in actual submitted documents and never hallucinates.
    required_documents = []
    for item in PREAUTH_CHECKLIST:
        if not item["required"]:
            continue
        step_label = item["label"]
        # Check if this step was satisfied
        satisfied = any(
            s["step"] == item["step"]
            for s in checklist_result["satisfied"]
        )
        required_documents.append({
            "document_name":      step_label,
            "document_type":      item["document_type"],
            "info_to_include":    item["description"],
            "satisfying_docs":    item["satisfied_by"],
            "currently_available": satisfied,
        })

    # Missing docs = required checklist items not yet satisfied
    missing_docs = [
        {
            "document_name":   m["label"],
            "document_type":   m["document_type"],
            "info_required":   m["description"],
            "why_needed":      m["missing_msg"],
            "satisfying_docs": m["satisfying_docs"],
        }
        for m in checklist_result["missing_required"]
    ]

    # Optional items that are missing = soft warnings only
    optional_missing = [
        {
            "document_name":   m["label"],
            "document_type":   m["document_type"],
            "info_required":   m["description"],
            "why_needed":      m["missing_msg"],
        }
        for m in checklist_result["missing_optional"]
    ]

    policy_analysis["required_documents"] = required_documents

    output = {
        "authorization_required": policy_analysis.get("authorization_required", True),
        "policy_analysis":        policy_analysis,

        # ── DETERMINISTIC results (authoritative for Agent 4) ─────────────
        "canonical_requirements": required_documents,   # grounded, not hallucinated
        "checklist_result":       checklist_result,

        # ── Missing docs from checklist (not from LLM) ────────────────────
        "missing_documents":      missing_docs,
        "optional_missing":       optional_missing,
        "missing_information":    [],
        "unmet_requirements":     [],

        "criteria_not_met":       policy_analysis.get("criteria_not_met", []),
        "data":                   data,
    }

    print(f"  Auth required       : {output['authorization_required']}")
    print(f"  Approval likelihood : {policy_analysis.get('likelihood_of_approval')}")
    print(f"  All required docs   : {checklist_result['all_required_met']}")
    if missing_docs:
        print("  Missing required docs:")
        for d in missing_docs:
            print(f"    ❌ [{d['document_type']}] {d['document_name']}")
    if optional_missing:
        print("  Optional missing:")
        for d in optional_missing:
            print(f"    ⚠️  {d['document_name']}")

    print("[Agent 3] Policy Retrieval Agent — DONE\n")
    return output
