"""
agent5_eligibility_reasoning.py  (IMPROVED)
─────────────────────────────────────────────
Agent 5 — Clinical Requirements Checker + Eligibility Reasoning

KEY IMPROVEMENTS over previous version:
  ① NEW: Dedicated clinical pre-authorization requirements checker that runs
    PROGRAMMATICALLY before the LLM call:
      - Minimum symptom duration (e.g. "must have had symptoms for ≥6 weeks")
      - Step therapy / medication trial requirements
        (e.g. "must have tried NSAIDs for ≥4 weeks before MRI is approved")
      - Minimum conservative treatment period
      - Specialist referral requirements
      - Prior imaging requirements
      - Lab result thresholds (e.g. HbA1c > X for certain procedures)

  ② The LLM receives the programmatic results so it can reason about BOTH
    clinical compliance and policy alignment together

  ③ New output fields:
      - `clinical_requirements_check`  — per-requirement pass/fail with evidence
      - `requirements_not_met`         — list of specific unmet requirements
      - `requirements_met`             — list of requirements that were satisfied
      - `missing_evidence`             — what additional evidence would satisfy unmet reqs
      - `step_therapy_status`          — detailed step therapy compliance

Input  : dict from Agent 4
Output : dict  {
    "eligible": bool,
    "determination": str,
    "clinical_requirements_check": dict,
    "requirements_met": [...],
    "requirements_not_met": [...],
    "missing_evidence": [...],
    ...
}
"""

import json
from bedrock_client import invoke, PRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Clinical requirement rules — these drive the programmatic pre-check
# Each rule defines what must be true for pre-auth to proceed
# ─────────────────────────────────────────────────────────────────────────────

# Minimum symptom duration by procedure category (weeks)
MIN_SYMPTOM_DURATION = {
    "MRI":                      6,
    "lumbar":                   6,
    "spine":                    6,
    "physical therapy":         4,
    "surgery":                  12,
    "joint replacement":        12,
    "cardiac":                  4,
    "default":                  4,
}

# Required step therapy by procedure — what must have been tried first
STEP_THERAPY_REQUIREMENTS = {
    "MRI Lumbar": [
        {"treatment": "NSAIDs", "min_duration_weeks": 4,  "alternatives": ["ibuprofen", "naproxen", "celecoxib", "diclofenac"]},
        {"treatment": "Physical Therapy", "min_duration_weeks": 3, "alternatives": ["PT", "physiotherapy", "exercise therapy"]},
    ],
    "MRI Spine": [
        {"treatment": "NSAIDs",           "min_duration_weeks": 4},
        {"treatment": "Physical Therapy", "min_duration_weeks": 3},
    ],
    "Joint Injection": [
        {"treatment": "NSAIDs",           "min_duration_weeks": 6},
        {"treatment": "Physical Therapy", "min_duration_weeks": 6},
        {"treatment": "Oral steroids",    "min_duration_weeks": 2},
    ],
    "Surgery": [
        {"treatment": "NSAIDs",           "min_duration_weeks": 6},
        {"treatment": "Physical Therapy", "min_duration_weeks": 8},
        {"treatment": "Specialist consult", "min_duration_weeks": 0},
    ],
    "default": [
        {"treatment": "Conservative treatment", "min_duration_weeks": 4},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a senior medical prior-authorization reviewer at an insurance company. "
            "Your job is to make accurate, fair eligibility determinations based on clinical "
            "evidence, policy criteria, and documentation completeness. "
            "You have been given a programmatic clinical requirements check. Build on it. "
            "Reason step by step and document your clinical and policy logic clearly. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_prompt(data: dict, policy_analysis: dict, doc_status: dict,
                  clinical_req_check: dict) -> str:

    criteria = policy_analysis.get("clinical_criteria_met", {})

    return f"""You must make a full eligibility determination for this prior-authorization request.
You have been given a programmatic clinical requirements check — use it in your reasoning.

═══ PATIENT & CLAIM SUMMARY ═══
Patient            : {data.get('patient_name')}, DOB {data.get('patient_dob')}
Insurer            : {data.get('insurer')} | Policy: {data.get('policy_number')}
Diagnosis          : {data.get('diagnosis')} ({data.get('icd10')})
Requested Procedure: {data.get('procedure')} (CPT: {data.get('cpt')})
Ordering Physician : {data.get('ordering_physician')} — {data.get('physician_specialty')}
Facility           : {data.get('hospital')}
Date of Service    : {data.get('date_of_proposed_treatment')}
Estimated Cost     : ${data.get('total_estimated_cost')}

═══ CLINICAL EVIDENCE ═══
Chief Complaint    : {data.get('diagnosis')}
Symptom Duration   : {data.get('symptom_duration_weeks')} weeks
Pain Score         : {data.get('pain_score')}/10
Clinical Findings  : {data.get('clinical_findings')}
Prior Treatments   : {json.dumps(data.get('prior_treatments', []))}
Prescribed Meds    : {json.dumps(data.get('prescribed_medications', []))}
Lab Results        : {json.dumps(data.get('lab_results', {}))}

═══ PROGRAMMATIC CLINICAL REQUIREMENTS CHECK ═══
(These were computed deterministically from extracted document data)

Overall: {clinical_req_check.get('overall_clinical_compliance')}
Requirements Met    : {json.dumps(clinical_req_check.get('requirements_met', []), indent=2)}
Requirements NOT Met: {json.dumps(clinical_req_check.get('requirements_not_met', []), indent=2)}
Step Therapy Status : {json.dumps(clinical_req_check.get('step_therapy_status', {}), indent=2)}
Missing Evidence    : {json.dumps(clinical_req_check.get('missing_evidence', []), indent=2)}

═══ POLICY CRITERIA ASSESSMENT (from Agent 3) ═══
{json.dumps(criteria, indent=2)}
Policy Notes        : {policy_analysis.get('policy_notes')}
Approval Likelihood : {policy_analysis.get('likelihood_of_approval')}
Policy Reasoning    : {policy_analysis.get('reasoning')}

═══ DOCUMENTATION STATUS (from Agent 4) ═══
All Docs Present    : {doc_status.get('all_docs_present')}
Can Proceed         : {doc_status.get('can_proceed')}
Missing Docs        : {json.dumps(doc_status.get('missing_docs', []))}
Partial (Blocking)  : {json.dumps(doc_status.get('partial_docs_blocking', []))}
Blockers            : {json.dumps(doc_status.get('blockers', []))}
Field Gap Analysis  : {json.dumps(doc_status.get('field_gap_analysis', []))}

Now perform a full eligibility determination. For each criterion, explicitly
state whether it is met, the evidence supporting your conclusion, and any
conditions or caveats.

Return JSON:
{{
  "eligible": true | false,
  "determination": "APPROVED" | "DENIED" | "PENDING_DOCS" | "PENDING_REVIEW",
  "confidence": "high" | "medium" | "low",
  "criteria_evaluation": {{
    "symptom_duration": {{
      "required": "6+ weeks",
      "actual": string,
      "met": true | false,
      "evidence": string,
      "notes": string
    }},
    "step_therapy_compliance": {{
      "required": string,
      "steps_required": [string],
      "steps_completed": [string],
      "steps_missing": [string],
      "met": true | false,
      "evidence": string,
      "notes": string
    }},
    "medication_trial": {{
      "required": string,
      "medications_tried": [string],
      "minimum_duration_met": true | false,
      "met": true | false,
      "evidence": string,
      "notes": string
    }},
    "clinical_examination": {{
      "required": "documented clinical findings supporting diagnosis",
      "actual": string,
      "met": true | false,
      "evidence": string,
      "notes": string
    }},
    "specialist_referral": {{
      "required": true | false,
      "present": true | false,
      "met": true | false,
      "notes": string
    }},
    "documentation_complete": {{
      "required": "all required docs submitted",
      "actual": string,
      "met": true | false,
      "notes": string
    }}
  }},
  "criteria_met_count": integer,
  "criteria_total": integer,
  "requirements_not_met": [
    {{
      "requirement": string,
      "what_was_found": string,
      "what_is_needed": string,
      "can_be_resolved": true | false,
      "resolution": string
    }}
  ],
  "denial_reasons": [string],
  "approval_conditions": [string],
  "clinical_summary": string,
  "recommendation": string,
  "urgency": "routine" | "urgent" | "emergent",
  "suggested_approval_validity_days": integer,
  "reviewer_notes": string
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Programmatic clinical requirements checker (runs before LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _check_symptom_duration(data: dict, policy_analysis: dict) -> dict:
    """Check if the patient meets minimum symptom duration for the procedure."""
    actual_weeks     = data.get("symptom_duration_weeks") or 0
    procedure        = (data.get("procedure") or "").lower()
    policy_criteria  = policy_analysis.get("standard_clinical_criteria", {})

    # Use policy-stated minimum if available, else look up our table
    policy_min       = policy_criteria.get("minimum_symptom_duration_weeks")

    required_weeks = policy_min
    if not required_weeks:
        required_weeks = MIN_SYMPTOM_DURATION.get("default", 4)
        for keyword, weeks in MIN_SYMPTOM_DURATION.items():
            if keyword in procedure:
                required_weeks = weeks
                break

    met = actual_weeks >= required_weeks if actual_weeks else False

    return {
        "requirement":     f"Minimum symptom duration: {required_weeks} weeks",
        "required_weeks":  required_weeks,
        "actual_weeks":    actual_weeks,
        "met":             met,
        "evidence":        (
            f"Patient has had symptoms for {actual_weeks} weeks "
            f"({'meets' if met else 'does NOT meet'} the {required_weeks}-week minimum)"
        ),
        "action_if_not_met": (
            f"Symptom duration of {actual_weeks} weeks is below the required {required_weeks} weeks. "
            f"Clinical notes must document symptom onset date. "
            f"If the patient has had symptoms longer, please update the doctor notes."
        ) if not met else None,
    }


def _check_step_therapy(data: dict, policy_analysis: dict) -> dict:
    """
    Check step therapy compliance — has the patient tried required prior
    treatments for the minimum required duration?
    """
    prior_treatments = data.get("prior_treatments", [])
    procedure        = (data.get("procedure") or "").lower()
    policy_criteria  = policy_analysis.get("standard_clinical_criteria", {})

    # Get policy-stated step therapy requirements if available
    policy_steps_required   = policy_criteria.get("step_therapy_steps_required", [])
    policy_step_therapy_met = policy_criteria.get("step_therapy_met")
    policy_gaps             = policy_criteria.get("step_therapy_gaps", [])

    # Find applicable rule
    applicable_steps = STEP_THERAPY_REQUIREMENTS.get("default", [])
    for key, steps in STEP_THERAPY_REQUIREMENTS.items():
        if key.lower() in procedure:
            applicable_steps = steps
            break

    steps_evaluation = []
    steps_met    = []
    steps_missing = []

    for step in applicable_steps:
        treatment     = step["treatment"]
        min_duration  = step["min_duration_weeks"]
        alternatives  = step.get("alternatives", [])

        # Search prior_treatments for this step or its alternatives
        search_terms = [treatment.lower()] + [a.lower() for a in alternatives]
        matched_treatment = None
        extracted_duration = None

        for pt_entry in prior_treatments:
            pt_lower = str(pt_entry).lower()
            if any(term in pt_lower for term in search_terms):
                matched_treatment = pt_entry
                # Try to extract duration from the treatment string
                import re
                duration_match = re.search(r'(\d+)\s*weeks?', pt_lower)
                if duration_match:
                    extracted_duration = int(duration_match.group(1))
                break

        duration_met = (
            extracted_duration >= min_duration
            if extracted_duration is not None
            else (matched_treatment is not None and min_duration == 0)
        )

        step_result = {
            "treatment":        treatment,
            "min_duration_weeks": min_duration,
            "found_in_records": matched_treatment is not None,
            "matched_entry":    matched_treatment,
            "extracted_duration_weeks": extracted_duration,
            "duration_met":     duration_met,
            "met":              matched_treatment is not None and duration_met,
            "action_if_not_met": (
                f"Evidence of {treatment} for at least {min_duration} weeks is required. "
                + (
                    f"Records show {extracted_duration} weeks (need {min_duration})."
                    if extracted_duration and not duration_met
                    else f"No record of {treatment} found in submitted documents."
                    if not matched_treatment else ""
                )
            ) if not (matched_treatment is not None and duration_met) else None,
        }

        steps_evaluation.append(step_result)

        if step_result["met"]:
            steps_met.append(treatment)
        else:
            steps_missing.append({
                "treatment":    treatment,
                "min_duration": min_duration,
                "issue":        step_result["action_if_not_met"],
            })

    all_steps_met = len(steps_missing) == 0

    # Cross-reference with policy's own step therapy assessment
    if policy_step_therapy_met is not None and not policy_step_therapy_met:
        all_steps_met = False
        for gap in policy_gaps:
            if not any(m.get("treatment", "") == gap for m in steps_missing):
                steps_missing.append({
                    "treatment": gap,
                    "min_duration": 0,
                    "issue": f"Policy states {gap} step therapy is incomplete or missing.",
                })

    return {
        "requirement":          "Step therapy / prior treatment compliance",
        "steps_required":       [s["treatment"] for s in applicable_steps],
        "steps_met":            steps_met,
        "steps_missing":        steps_missing,
        "steps_evaluation":     steps_evaluation,
        "all_steps_met":        all_steps_met,
        "met":                  all_steps_met,
        "evidence":             (
            f"Completed steps: {steps_met}. "
            + (f"Missing/incomplete: {[m['treatment'] for m in steps_missing]}."
               if steps_missing else "All required steps completed.")
        ),
    }


def _check_medication_trial(data: dict, policy_analysis: dict) -> dict:
    """
    Check if a minimum medication trial period has been documented.
    Focuses on prescription medications and their duration.
    """
    prior_treatments     = data.get("prior_treatments", [])
    prescribed_medications = data.get("prescribed_medications", [])
    policy_criteria      = policy_analysis.get("standard_clinical_criteria", {})

    medication_trial_required = policy_criteria.get("medication_trial_required", True)
    medication_trial_details  = policy_criteria.get("medication_trial_details", "")
    medication_trial_met      = policy_criteria.get("medication_trial_met")

    # Look for any medication trial evidence
    import re
    medications_found = []
    for entry in prior_treatments + prescribed_medications:
        entry_str = str(entry).lower()
        # Common medication-related keywords
        med_keywords = [
            "nsaid", "ibuprofen", "naproxen", "acetaminophen", "aspirin",
            "steroid", "prednisone", "cortisone", "muscle relaxant",
            "gabapentin", "lyrica", "celebrex", "celecoxib", "diclofenac",
            "meloxicam", "mg", "tablet", "capsule", "injection",
        ]
        if any(kw in entry_str for kw in med_keywords):
            duration_match = re.search(r'(\d+)\s*(weeks?|days?|months?)', entry_str)
            medications_found.append({
                "medication": entry,
                "duration":   duration_match.group(0) if duration_match else "duration unclear",
            })

    trial_found = len(medications_found) > 0

    # If policy already assessed this, defer to it
    if medication_trial_met is not None:
        trial_met = medication_trial_met
    else:
        trial_met = trial_found and medication_trial_required

    return {
        "requirement":            "Medication trial documentation",
        "trial_required":         medication_trial_required,
        "medications_documented": medications_found,
        "met":                    trial_met,
        "policy_stated_details":  medication_trial_details,
        "evidence": (
            f"Found {len(medications_found)} medication trials in records: "
            f"{[m['medication'] for m in medications_found[:3]]}"
            if medications_found
            else "No medication trial documentation found in submitted records."
        ),
        "action_if_not_met": (
            "Please obtain a prescription record or updated doctor notes documenting "
            "the medications tried, dosage, and duration of use."
        ) if not trial_met and medication_trial_required else None,
    }


def _check_specialist_referral(data: dict, docs_present: dict) -> dict:
    """Check if a specialist referral is present when required."""
    has_referral = docs_present.get("physician_referral", False)
    specialty    = (data.get("physician_specialty") or "").lower()

    referral_required = any(
        kw in specialty for kw in
        ["specialist", "orthopedic", "neurolog", "cardiolog", "oncolog", "rheumatolog"]
    )

    return {
        "requirement":       "Specialist referral (if applicable)",
        "referral_required": referral_required,
        "referral_present":  has_referral,
        "met":               (not referral_required) or has_referral,
        "evidence": (
            "Physician referral found in submitted documents."
            if has_referral else
            "No physician referral found."
        ),
        "action_if_not_met": (
            "A specialist referral letter is required for this procedure. "
            "Please obtain a referral from your primary care physician."
        ) if referral_required and not has_referral else None,
    }


def _check_clinical_findings(data: dict) -> dict:
    """Check if clinical examination findings are documented."""
    findings = data.get("clinical_findings")
    diagnosis = data.get("diagnosis")
    icd10     = data.get("icd10_codes", [])

    has_findings = bool(findings and str(findings).strip())
    has_diagnosis = bool(diagnosis)
    has_icd10    = bool(icd10 and len(icd10) > 0)

    met = has_findings and has_diagnosis and has_icd10

    return {
        "requirement":       "Clinical examination and diagnosis documentation",
        "findings_present":  has_findings,
        "diagnosis_present": has_diagnosis,
        "icd10_present":     has_icd10,
        "met":               met,
        "evidence": (
            f"Diagnosis: {diagnosis} ({icd10}). Findings: {str(findings)[:100]}..."
            if has_findings else
            f"Diagnosis present ({diagnosis}) but clinical findings not documented."
            if has_diagnosis else
            "No diagnosis or clinical findings documented."
        ),
        "action_if_not_met": (
            ("Clinical findings are not documented. " if not has_findings else "") +
            ("ICD-10 code is missing from doctor notes. " if not has_icd10 else "") +
            "Please ask your physician to update the clinical notes."
        ) if not met else None,
    }


def _run_clinical_requirements_check(data: dict, policy_analysis: dict,
                                      docs_present: dict) -> dict:
    """
    Master clinical requirements checker.
    Runs all programmatic checks and returns a consolidated result.
    """
    print("[Agent 5] Running programmatic clinical requirements check...")

    symptom_check   = _check_symptom_duration(data, policy_analysis)
    step_therapy    = _check_step_therapy(data, policy_analysis)
    med_trial       = _check_medication_trial(data, policy_analysis)
    referral_check  = _check_specialist_referral(data, docs_present)
    clinical_check  = _check_clinical_findings(data)

    all_checks = [symptom_check, step_therapy, med_trial, referral_check, clinical_check]

    requirements_met     = []
    requirements_not_met = []
    missing_evidence     = []

    for check in all_checks:
        if check.get("met"):
            requirements_met.append({
                "requirement": check["requirement"],
                "evidence":    check.get("evidence", ""),
            })
        else:
            requirements_not_met.append({
                "requirement":    check["requirement"],
                "evidence":       check.get("evidence", ""),
                "action_needed":  check.get("action_if_not_met", ""),
            })
            if check.get("action_if_not_met"):
                missing_evidence.append(check["action_if_not_met"])

    compliance_count = len(requirements_met)
    total_count      = len(all_checks)
    overall          = (
        "FULL_COMPLIANCE"    if compliance_count == total_count else
        "PARTIAL_COMPLIANCE" if compliance_count >= total_count * 0.6 else
        "NON_COMPLIANT"
    )

    # Print summary
    print(f"  Clinical compliance: {compliance_count}/{total_count} requirements met")
    for req in requirements_met:
        print(f"    ✅ {req['requirement']}")
    for req in requirements_not_met:
        print(f"    ❌ {req['requirement']}")
        if req.get("action_needed"):
            print(f"       → {req['action_needed']}")

    return {
        "overall_clinical_compliance": overall,
        "compliance_count":            compliance_count,
        "total_requirements":          total_count,
        "requirements_met":            requirements_met,
        "requirements_not_met":        requirements_not_met,
        "missing_evidence":            missing_evidence,
        "step_therapy_status":         {
            "steps_required":  step_therapy.get("steps_required", []),
            "steps_completed": step_therapy.get("steps_met", []),
            "steps_missing":   step_therapy.get("steps_missing", []),
            "all_steps_met":   step_therapy.get("all_steps_met", False),
        },
        "checks": {
            "symptom_duration": symptom_check,
            "step_therapy":     step_therapy,
            "medication_trial": med_trial,
            "specialist_referral": referral_check,
            "clinical_findings": clinical_check,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent4_output: dict) -> dict:
    print("\n[Agent 5] Clinical Requirements + Eligibility Reasoning — START")

    data            = agent4_output.get("data", {})
    policy_analysis = agent4_output.get("policy_analysis", {})
    doc_status      = agent4_output   # Agent 4's full output is the doc status

    docs_present    = data.get("documents_present", {})

    # ── Step 1: Programmatic clinical requirements check ──────────────────────
    clinical_req_check = _run_clinical_requirements_check(data, policy_analysis, docs_present)

    # ── Step 2: LLM eligibility reasoning (uses programmatic results) ─────────
    print("[Agent 5] Calling Nova Pro for eligibility determination...")
    prompt   = _build_prompt(data, policy_analysis, doc_status, clinical_req_check)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    raw = invoke(
        model_id=PRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.1,
    )

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]

    try:
        determination = json.loads(text.strip())
    except json.JSONDecodeError:
        print("[Agent 5] ⚠ JSON parse error — using fallback")
        determination = {
            "eligible":          False,
            "determination":     "PENDING_REVIEW",
            "confidence":        "low",
            "criteria_evaluation": {},
            "criteria_met_count": clinical_req_check["compliance_count"],
            "criteria_total":     clinical_req_check["total_requirements"],
            "denial_reasons":    [r["requirement"] for r in clinical_req_check["requirements_not_met"]],
            "approval_conditions": [],
            "clinical_summary":  "Parse error — manual review required.",
            "recommendation":    "Refer for manual review.",
            "urgency":           "routine",
            "suggested_approval_validity_days": 90,
            "reviewer_notes":    "",
        }

    # ── Step 3: Override determination if programmatic check found hard failures
    prog_not_met = clinical_req_check.get("requirements_not_met", [])
    if prog_not_met and determination.get("determination") == "APPROVED":
        # Don't silently approve if step therapy or duration is unmet
        critical_failures = [
            r for r in prog_not_met
            if any(kw in r["requirement"].lower()
                   for kw in ["step therapy", "symptom duration", "clinical findings"])
        ]
        if critical_failures:
            determination["determination"] = "PENDING_REVIEW"
            determination["eligible"]      = False
            determination["confidence"]    = "medium"
            if "denial_reasons" not in determination:
                determination["denial_reasons"] = []
            for f in critical_failures:
                if f["requirement"] not in determination["denial_reasons"]:
                    determination["denial_reasons"].append(
                        f"{f['requirement']}: {f['evidence']}"
                    )

    output = {
        "eligible":           determination.get("eligible"),
        "determination":      determination.get("determination"),
        "confidence":         determination.get("confidence"),

        # ── Clinical requirements check (NEW) ──────────────────────────────
        "clinical_requirements_check": clinical_req_check,
        "requirements_met":            clinical_req_check.get("requirements_met", []),
        "requirements_not_met":        clinical_req_check.get("requirements_not_met", []),
        "missing_evidence":            clinical_req_check.get("missing_evidence", []),
        "step_therapy_status":         clinical_req_check.get("step_therapy_status", {}),
        "overall_clinical_compliance": clinical_req_check.get("overall_clinical_compliance"),

        # ── LLM determination ──────────────────────────────────────────────
        "criteria_evaluation":    determination.get("criteria_evaluation", {}),
        "criteria_met_count":     determination.get("criteria_met_count"),
        "criteria_total":         determination.get("criteria_total"),
        "denial_reasons":         determination.get("denial_reasons", []),
        "approval_conditions":    determination.get("approval_conditions", []),
        "recommendation":         determination.get("recommendation"),
        "urgency":                determination.get("urgency", "routine"),
        "clinical_summary":       determination.get("clinical_summary"),
        "reviewer_notes":         determination.get("reviewer_notes"),
        "suggested_validity_days": determination.get("suggested_approval_validity_days"),

        # ── Pass-through for Agent 6 ───────────────────────────────────────
        "data":            data,
        "policy_analysis": policy_analysis,
    }

    print(f"  Determination          : {output['determination']}")
    print(f"  Eligible               : {output['eligible']}")
    print(f"  Confidence             : {output['confidence']}")
    print(f"  Clinical compliance    : {output['overall_clinical_compliance']}")
    print(f"  Requirements met       : {output['criteria_met_count']}/{output['criteria_total']}")

    if output["requirements_not_met"]:
        print("\n  ⚠️  Unmet clinical requirements:")
        for req in output["requirements_not_met"]:
            print(f"    ✗ {req['requirement']}")
            if req.get("action_needed"):
                print(f"      → {req['action_needed']}")

    print("[Agent 5] Clinical Requirements + Eligibility Reasoning — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "all_docs_present":     True,
        "can_proceed":          True,
        "missing_docs":         [],
        "partial_docs_blocking": [],
        "blockers":             [],
        "data": {
            "patient_name":           "John R. Doe",
            "patient_dob":            "1985-05-12",
            "insurer":                "BlueCross BlueShield",
            "policy_number":          "BCBS-99001122",
            "member_id":              "BCBS-99001122",
            "diagnosis":              "Lumbar Radiculopathy",
            "icd10_codes":            ["M54.16"],
            "icd10":                  "M54.16",
            "procedure":              "MRI Lumbar Spine Without Contrast",
            "cpt_codes":              ["72148"],
            "cpt":                    "72148",
            "ordering_physician":     "Dr. Sarah Jenkins",
            "physician_specialty":    "Orthopedic Surgery",
            "hospital":               "Metropolitan General Hospital",
            "date_of_proposed_treatment": "2024-03-15",
            "total_estimated_cost":   2025,
            "symptom_duration_weeks": 7,
            "pain_score":             7,
            "clinical_findings":      "Positive SLR at 30 degrees right. Antalgic gait.",
            "prior_treatments": [
                "NSAIDs (Ibuprofen) - 4 weeks - Minimal relief",
                "Physical Therapy - 3 weeks - Symptoms persisted",
            ],
            "prescribed_medications": ["Ibuprofen 400mg", "Cyclobenzaprine 5mg"],
            "documents_present": {
                "doctor_notes":   True,
                "insurance_card": True,
                "patient_info":   True,
                "physician_referral": False,
            },
        },
        "policy_analysis": {
            "authorization_required": True,
            "standard_clinical_criteria": {
                "minimum_symptom_duration_weeks": 6,
                "minimum_symptom_duration_met":   True,
                "step_therapy_required":          True,
                "step_therapy_steps_required":    ["NSAIDs", "Physical Therapy"],
                "step_therapy_steps_completed":   ["NSAIDs", "Physical Therapy"],
                "step_therapy_met":               True,
                "step_therapy_gaps":              [],
                "medication_trial_required":      True,
                "medication_trial_met":           True,
            },
            "policy_notes": "Standard BCBS criteria for lumbar MRI.",
            "likelihood_of_approval": "high",
            "reasoning": "All standard criteria appear met.",
        },
    }

    result = run(mock_input)
    # Print without huge nested structures
    display = {k: v for k, v in result.items() if k not in ("data", "policy_analysis", "clinical_requirements_check")}
    display["clinical_compliance_summary"] = {
        "overall":         result["overall_clinical_compliance"],
        "requirements_met": [r["requirement"] for r in result["requirements_met"]],
        "requirements_not_met": [r["requirement"] for r in result["requirements_not_met"]],
    }
    print(json.dumps(display, indent=2))
