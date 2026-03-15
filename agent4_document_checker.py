"""
agent4_document_checker.py
───────────────────────────
Agent 4 — Document Requirement Checker
• Uses documents_present from Agent 1 directly (no re-inference)
• Falls back to field-level inference if running standalone
• Merges unmet requirements and missing info from Agent 3
• Tags every document with an importance level for THIS specific claim:
    CRITICAL   — submission will be rejected without it
    IMPORTANT  — strongly recommended; increases approval odds
    OPTIONAL   — helpful context but not required for this case
    NOT_NEEDED — not applicable to this diagnosis/procedure/insurer
• Hard stop if blockers exist — prints exactly what user must submit
• Uses Nova Micro

Input  : dict from Agent 3
Output : dict  {
    "all_docs_present": bool,
    "missing_docs": [...],
    "can_proceed": bool,
    "document_importance": { doc_name: { level, reason, can_skip } },
    "importance_summary": { critical: [...], important: [...],
                            optional: [...], not_needed: [...] }
}
"""

import json

from bedrock_client import invoke, MICRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Importance levels
# ─────────────────────────────────────────────────────────────────────────────

IMPORTANCE_LEVELS = {
    "CRITICAL":   "❌ CRITICAL   — submission WILL be rejected without this",
    "IMPORTANT":  "⚠️  IMPORTANT  — strongly recommended; improves approval odds",
    "OPTIONAL":   "ℹ️  OPTIONAL   — helpful context, but not required for this case",
    "NOT_NEEDED": "✅ NOT NEEDED  — not applicable to this diagnosis / procedure",
}

# Static baseline importance per document type — used as a starting hint
# to the LLM; the LLM can override these based on the actual claim context.
BASELINE_IMPORTANCE = {
    "lab_report":                    "OPTIONAL",
    "doctor_notes":                  "CRITICAL",
    "patient_info":                  "CRITICAL",
    "insurance_card":                "CRITICAL",
    "pretreatment_estimate":         "IMPORTANT",
    "prior_treatment_documentation": "IMPORTANT",
    "procedure_order":               "CRITICAL",
    "physician_referral":            "IMPORTANT",
}


# ─────────────────────────────────────────────────────────────────────────────
# Document inventory
# ─────────────────────────────────────────────────────────────────────────────

def _inventory_available_docs(data: dict) -> dict:
    """
    Use documents_present from Agent 1 if available.
    Falls back to field-level inference for standalone use.
    """
    if "documents_present" in data:
        return data["documents_present"]

    return {
        "lab_report":                    bool(any(v for v in data.get("lab_results", {}).values() if v)),
        "doctor_notes":                  bool(data.get("clinical_findings") or data.get("diagnosis")),
        "patient_info":                  bool(data.get("patient_name") and data.get("patient_dob")),
        "insurance_card":                bool(data.get("insurer") and data.get("policy_number")),
        "pretreatment_estimate":         bool(data.get("total_estimated_cost")),
        "prior_treatment_documentation": bool(data.get("prior_treatments") and len(data.get("prior_treatments", [])) > 0),
        "procedure_order":               bool(data.get("procedure") and data.get("cpt")),
        "physician_referral":            False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prompts — gap analysis
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a prior-authorization document specialist. "
            "Compare available documents against policy requirements, identify gaps, "
            "and assign an importance level to every document for this specific claim. "
            "Be precise about what is missing and what the user truly needs to submit. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_gap_prompt(required_docs: list, available_docs: dict,
                      policy_notes: str, unmet_requirements: list,
                      missing_information: list) -> str:
    return f"""You are checking documentation completeness for a prior-authorization request.

DOCUMENTS REQUIRED BY POLICY:
{json.dumps(required_docs, indent=2)}

DOCUMENTS WE CURRENTLY HAVE (True = present, False = missing):
{json.dumps(available_docs, indent=2)}

UNMET CLINICAL REQUIREMENTS (from policy analysis):
{json.dumps(unmet_requirements, indent=2)}

MISSING INFORMATION (from policy analysis):
{json.dumps(missing_information, indent=2)}

POLICY NOTES:
{policy_notes}

For each required document, determine if we have it or an equivalent.
Be flexible — e.g. "office visit notes" covers "physician notes" or "clinical findings".
For each unmet requirement, specify exactly what document or information the user must provide.

Return JSON:
{{
  "document_status": {{
    "<required_doc_name>": {{
      "status": "present" | "missing" | "partial",
      "matched_to": string | null,
      "notes": string
    }}
  }},
  "all_docs_present": true | false,
  "missing_docs": [string],
  "partial_docs": [string],
  "present_docs": [string],
  "blockers": [string],
  "user_action_required": [
    {{
      "item": string,
      "reason": string,
      "how_to_resolve": string
    }}
  ],
  "recommendations": [string]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Prompts — importance tagging
# ─────────────────────────────────────────────────────────────────────────────

def _build_importance_prompt(data: dict, available_docs: dict,
                              required_docs: list, policy_notes: str) -> str:
    return f"""You are a prior-authorization specialist assigning importance tags to documents
for a specific insurance claim. Evaluate each document in the context of THIS claim only.

CLAIM CONTEXT:
- Insurer          : {data.get('insurer')}
- Plan Type        : {data.get('plan_type')}
- Diagnosis        : {data.get('diagnosis')} ({data.get('icd10')})
- Procedure        : {data.get('procedure')} (CPT {data.get('cpt')})
- Physician Spec.  : {data.get('physician_specialty')}
- Prior Treatments : {json.dumps(data.get('prior_treatments', []))}
- Symptom Duration : {data.get('symptom_duration_weeks')} weeks

DOCUMENTS WE HAVE:
{json.dumps(available_docs, indent=2)}

DOCUMENTS REQUIRED BY POLICY:
{json.dumps(required_docs, indent=2)}

POLICY NOTES:
{policy_notes}

BASELINE IMPORTANCE HINTS (you may override based on context):
{json.dumps(BASELINE_IMPORTANCE, indent=2)}

For EVERY document key in "DOCUMENTS WE HAVE" AND every document in
"DOCUMENTS REQUIRED BY POLICY", assign an importance level.

Importance levels — choose exactly one per document:
  CRITICAL   : The insurer will reject the submission outright without this.
  IMPORTANT  : Not strictly mandatory, but absence significantly lowers approval
               odds or will trigger a Request for Additional Information (RAI).
  OPTIONAL   : Provides supporting context; this insurer/procedure does NOT require
               it and its absence will NOT delay or prevent approval.
  NOT_NEEDED : Does not apply to this diagnosis, procedure, or insurer at all —
               the user should not waste time gathering it.

Return JSON:
{{
  "document_importance": {{
    "<document_name>": {{
      "importance": "CRITICAL" | "IMPORTANT" | "OPTIONAL" | "NOT_NEEDED",
      "reason": string,
      "can_skip": true | false,
      "present": true | false
    }}
  }},
  "importance_summary": {{
    "critical":   [string],
    "important":  [string],
    "optional":   [string],
    "not_needed": [string]
  }},
  "skip_safe_message": string
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Importance report printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_importance_report(output: dict) -> None:
    """Print a formatted document importance report to the console."""
    importance = output.get("document_importance", {})
    skip_msg   = output.get("skip_safe_message", "")

    if not importance:
        return

    print("\n" + "─" * 70)
    print("  DOCUMENT IMPORTANCE REPORT")
    print("─" * 70)

    tier_order = ["CRITICAL", "IMPORTANT", "OPTIONAL", "NOT_NEEDED"]
    for tier in tier_order:
        docs_in_tier = [
            (doc, info) for doc, info in importance.items()
            if info.get("importance") == tier
        ]
        if not docs_in_tier:
            continue

        print(f"\n  {IMPORTANCE_LEVELS[tier]}")
        for doc, info in docs_in_tier:
            present_tag = "✓ have   " if info.get("present") else "✗ missing"
            skip_tag    = "  [safe to skip]" if info.get("can_skip") else ""
            print(f"    • {doc:<42} [{present_tag}]{skip_tag}")
            print(f"      → {info.get('reason', '')}")

    if skip_msg:
        print(f"\n  ℹ️  WHAT YOU CAN SAFELY SKIP FOR THIS CLAIM:")
        words = skip_msg.split()
        line  = "     "
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = "     " + word
            else:
                line += (" " if line.strip() else "") + word
        if line.strip():
            print(line)

    print("─" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent3_output: dict) -> dict:
    print("\n[Agent 4] Document Requirement Checker — START")

    data            = agent3_output.get("data", {})
    policy_analysis = agent3_output.get("policy_analysis", {})
    required_docs   = policy_analysis.get("required_documents", [])
    policy_notes    = policy_analysis.get("policy_notes", "")

    unmet_requirements  = agent3_output.get("unmet_requirements", [])
    missing_information = agent3_output.get("missing_information", [])

    if not required_docs:
        required_docs = [
            "Completed prior authorization request form",
            "Office visit notes (within 60 days)",
            "Documentation of conservative treatment failure",
            "Referring physician NPI and specialty",
            "ICD-10 and CPT codes",
            "Estimated date of service",
        ]

    print(f"  Required docs ({len(required_docs)}): {required_docs}")

    available_docs = _inventory_available_docs(data)
    present = [k for k, v in available_docs.items() if v]
    absent  = [k for k, v in available_docs.items() if not v]
    print(f"  Available docs : {present}")
    print(f"  Absent docs    : {absent}")

    # ── Step 1: Gap analysis ─────────────────────────────────────────────────
    gap_prompt = _build_gap_prompt(required_docs, available_docs, policy_notes,
                                   unmet_requirements, missing_information)
    messages = [{"role": "user", "content": [{"text": gap_prompt}]}]

    print("[Agent 4] Calling Nova Micro for document gap analysis...")
    raw = invoke(
        model_id=MICRO_MODEL_ID,
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
        doc_analysis = json.loads(text.strip())
    except json.JSONDecodeError:
        doc_analysis = {
            "all_docs_present":     False,
            "missing_docs":         ["Parse error — manual review required"],
            "present_docs":         present,
            "blockers":             ["Could not parse document analysis"],
            "user_action_required": [],
        }

    # ── Step 2: Importance tagging ───────────────────────────────────────────
    imp_prompt   = _build_importance_prompt(data, available_docs, required_docs, policy_notes)
    imp_messages = [{"role": "user", "content": [{"text": imp_prompt}]}]

    print("[Agent 4] Calling Nova Micro for document importance tagging...")
    imp_raw = invoke(
        model_id=MICRO_MODEL_ID,
        messages=imp_messages,
        system=SYSTEM_PROMPT,
        max_tokens=900,
        temperature=0.1,
    )

    imp_text = imp_raw.strip()
    if imp_text.startswith("```"):
        imp_text = imp_text.split("```")[1]
        if imp_text.lower().startswith("json"):
            imp_text = imp_text[4:]
    try:
        importance_result = json.loads(imp_text.strip())
    except json.JSONDecodeError:
        # Fallback: apply baseline statically
        importance_result = {
            "document_importance": {
                doc: {
                    "importance": BASELINE_IMPORTANCE.get(doc, "OPTIONAL"),
                    "reason":     "Baseline importance applied (LLM parse error).",
                    "can_skip":   BASELINE_IMPORTANCE.get(doc, "OPTIONAL") in ("OPTIONAL", "NOT_NEEDED"),
                    "present":    available_docs.get(doc, False),
                }
                for doc in list(available_docs.keys())
            },
            "importance_summary": {
                "critical":   [k for k, v in BASELINE_IMPORTANCE.items() if v == "CRITICAL"],
                "important":  [k for k, v in BASELINE_IMPORTANCE.items() if v == "IMPORTANT"],
                "optional":   [k for k, v in BASELINE_IMPORTANCE.items() if v == "OPTIONAL"],
                "not_needed": [],
            },
            "skip_safe_message": (
                "Based on standard policy, lab reports are typically optional for imaging "
                "pre-authorization unless the insurer specifically requests them."
            ),
        }

    # ── Assemble output ──────────────────────────────────────────────────────
    output = {
        "all_docs_present":     doc_analysis.get("all_docs_present", False),
        "missing_docs":         doc_analysis.get("missing_docs", []),
        "partial_docs":         doc_analysis.get("partial_docs", []),
        "present_docs":         doc_analysis.get("present_docs", []),
        "blockers":             doc_analysis.get("blockers", []),
        "user_action_required": doc_analysis.get("user_action_required", []),
        "recommendations":      doc_analysis.get("recommendations", []),
        "document_status":      doc_analysis.get("document_status", {}),
        "available_inventory":  available_docs,
        "missing_information":  missing_information,
        # ── Importance tagging ───────────────────────────────────────────────
        "document_importance":  importance_result.get("document_importance", {}),
        "importance_summary":   importance_result.get("importance_summary", {}),
        "skip_safe_message":    importance_result.get("skip_safe_message", ""),
        # ────────────────────────────────────────────────────────────────────
        "data":                 data,
    }

    # Merge unmet clinical requirements from Agent 3
    a3_missing_docs = agent3_output.get("missing_documents", [])
    if a3_missing_docs:
        output["missing_docs"] = list(set(output["missing_docs"] + a3_missing_docs))

    if unmet_requirements:
        extra_blockers = [
            f"{u.get('requirement')} — {u.get('what_is_needed')}"
            for u in unmet_requirements
        ]
        output["blockers"] = list(set(output["blockers"] + extra_blockers))

    if missing_information:
        output["blockers"] = list(set(
            output["blockers"] + [f"Missing information: {m}" for m in missing_information]
        ))

    print(f"  All docs present : {output['all_docs_present']}")
    print(f"  Missing docs     : {output['missing_docs']}")
    print(f"  Blockers         : {output['blockers']}")

    # ── Importance report ────────────────────────────────────────────────────
    _print_importance_report(output)

    # ── Hard stop if blockers ────────────────────────────────────────────────
    if output["blockers"] or output["missing_docs"]:
        print("\n" + "!" * 70)
        print("  ACTION REQUIRED — Cannot proceed with submission")
        print("!" * 70)
        if output["missing_docs"]:
            print("\n  Missing Documents — please submit the following:")
            for doc in output["missing_docs"]:
                imp_info = output["document_importance"].get(doc, {})
                tag      = imp_info.get("importance", "?")
                print(f"    [{tag}]  {doc}")
                if imp_info.get("reason"):
                    print(f"             → {imp_info['reason']}")
        if output["blockers"]:
            print("\n  Blockers — must be resolved before authorization:")
            for b in output["blockers"]:
                print(f"    [BLOCKER] {b}")
        if output.get("user_action_required"):
            print("\n  Specific Actions Required:")
            for action in output["user_action_required"]:
                print(f"    > {action.get('item')}")
                print(f"      Reason     : {action.get('reason')}")
                print(f"      How to fix : {action.get('how_to_resolve')}")
        print("!" * 70 + "\n")

    output["can_proceed"] = (
        len(output["blockers"]) == 0 and len(output["missing_docs"]) == 0
    )

    print(f"  Can proceed      : {output['can_proceed']}")
    print("[Agent 4] Document Requirement Checker — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "data": {
            "patient_name":          "John R. Doe",
            "patient_dob":           "1985-05-12",
            "insurer":               "BlueCross BlueShield",
            "plan_type":             "PPO",
            "diagnosis":             "Lumbar Radiculopathy",
            "icd10":                 "M54.16",
            "icd10_codes":           ["M54.16"],
            "procedure":             "MRI Lumbar Spine Without Contrast",
            "cpt":                   "72148",
            "ordering_physician":    "Dr. Sarah Jenkins",
            "physician_specialty":   "Orthopedic Surgery",
            "prior_treatments":      ["NSAIDs 4 weeks", "PT 3 weeks"],
            "symptom_duration_weeks": 7,
            "total_estimated_cost":  2025,
            "clinical_findings":     "Positive SLR at 30 degrees",
            "lab_results":           {"hba1c": "5.6%"},
            "documents_present": {
                "lab_report":                    True,
                "doctor_notes":                  True,
                "patient_info":                  True,
                "insurance_card":                True,
                "pretreatment_estimate":         True,
                "prior_treatment_documentation": True,
                "procedure_order":               True,
                "physician_referral":            False,
            },
        },
        "policy_analysis": {
            "required_documents": [
                "Completed prior authorization request form",
                "Office visit notes within 60 days",
                "Documentation of conservative treatment failure",
                "Referring physician NPI and specialty",
                "ICD-10 and CPT codes",
                "Estimated date of service",
            ],
            "policy_notes": "Standard BCBS prior auth for MRI Lumbar Spine.",
        },
        "unmet_requirements":  [],
        "missing_documents":   [],
        "missing_information": [],
    }
    result = run(mock_input)
    display = {k: v for k, v in result.items() if k != "data"}
    print(json.dumps(display, indent=2, default=str))
