"""
agent4_document_checker.py  (RELIABILITY FIX)
───────────────────────────────────────────────
Agent 4 — Document Requirement Checker

RELIABILITY FIX:
  Old Agent 4 used the LLM's `required_documents` list as its source of truth.
  Since that list was hallucinated by Agent 3's LLM, Agent 4 would flag documents
  like "Patient's medical history" as missing even when all relevant content
  was already present in submitted documents — and it did so inconsistently.

  New Agent 4 uses `canonical_requirements` from Agent 3 — the deterministic
  checklist result. That checklist was already evaluated against actual submitted
  document content. Agent 4's job is now to:

  ① Re-verify each canonical requirement using full document CONTENT values
     (not just field names) — catch cases where a doc is present but empty
  ② Run field-level completeness checks on partial documents
  ③ Generate precise, actionable user-facing messages
  ④ Produce a stable, deterministic can_proceed decision

  The LLM in Agent 4 is now used ONLY for generating human-readable instructions.
  The pass/fail decision is made programmatically.
"""

import json
from bedrock_client import invoke, MICRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Critical fields per document type
# ─────────────────────────────────────────────────────────────────────────────

CRITICAL_FIELDS_BY_DOC = {
    "Doctor Notes": [
        "diagnosis", "icd10_codes", "clinical_findings",
        "requested_procedure", "cpt_codes", "ordering_physician",
    ],
    "Insurance Card": [
        "insurer_name", "policy_number", "member_id",
    ],
    "Patient Information Sheet": [
        "patient_name", "dob",
    ],
    "Lab Report": [
        "test_name", "results",
    ],
    "Medical Pretreatment Estimate": [
        "total_estimated_cost",
    ],
    "Physician Referral": [
        "referring_physician", "reason_for_referral",
    ],
    "Prescription": [
        "medication_name", "duration",
    ],
    "Prior Treatment Documentation": [
        "prior_treatments",
    ],
}

FIELD_HELP = {
    "diagnosis":           "primary diagnosis name",
    "icd10_codes":         "ICD-10 code(s)",
    "clinical_findings":   "physical examination findings",
    "requested_procedure": "name of the requested procedure",
    "cpt_codes":           "CPT procedure code(s)",
    "ordering_physician":  "ordering physician name",
    "insurer_name":        "insurance company name",
    "policy_number":       "policy number",
    "member_id":           "member ID",
    "patient_name":        "patient full name",
    "dob":                 "date of birth",
    "test_name":           "lab test name",
    "results":             "lab results",
    "total_estimated_cost": "total estimated cost",
    "referring_physician": "referring physician name",
    "reason_for_referral": "reason for referral",
    "medication_name":     "medication name",
    "duration":            "medication duration",
    "prior_treatments":    "list of prior treatments with duration and outcomes",
}


def _has_real_value(val) -> bool:
    """Return True only if a field has a meaningful non-empty value."""
    if val is None:
        return False
    if isinstance(val, list):
        return len(val) > 0 and any(
            v and str(v).strip() not in ("", "null", "None") for v in val
        )
    return str(val).strip() not in ("", "null", "None")


def _check_document_completeness(doc_type: str, content: dict) -> dict:
    """
    Programmatically check a document's critical fields.
    Returns which fields are present vs missing.
    """
    critical = CRITICAL_FIELDS_BY_DOC.get(doc_type, [])
    present  = [f for f in critical if _has_real_value(content.get(f))]
    missing  = [f for f in critical if not _has_real_value(content.get(f))]
    score    = round(len(present) / len(critical), 2) if critical else 1.0

    return {
        "completeness_score":      score,
        "present_critical_fields": present,
        "missing_critical_fields": missing,
        "is_complete":             len(missing) == 0,
        "is_partial":              0 < len(missing) < len(critical),
        "is_empty":                len(present) == 0,
    }


def _build_user_fix_instruction(doc_type: str, missing_fields: list) -> str:
    """Build a plain-English instruction for the user."""
    field_descs = [
        f"`{f}` ({FIELD_HELP.get(f, f.replace('_', ' '))})"
        for f in missing_fields
    ]
    fields_str = ", ".join(field_descs)

    templates = {
        "Doctor Notes": (
            f"Please ask your physician to update the clinical notes to include: {fields_str}."
        ),
        "Insurance Card": (
            f"Please provide a clearer copy of your insurance card that clearly shows: {fields_str}."
        ),
        "Patient Information Sheet": (
            f"Please complete the patient information form — missing: {fields_str}."
        ),
        "Lab Report": (
            f"The lab report is missing: {fields_str}. Please obtain an updated report."
        ),
        "Medical Pretreatment Estimate": (
            f"The pretreatment estimate is missing: {fields_str}. "
            f"Please ask the facility to update it."
        ),
        "Physician Referral": (
            f"The referral letter is missing: {fields_str}. "
            f"Please ask the referring physician to include these details."
        ),
        "Prescription": (
            f"The prescription is missing: {fields_str}. "
            f"Please ask the prescribing physician to update it."
        ),
    }
    return templates.get(
        doc_type,
        f"Please update {doc_type} to include: {fields_str}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent3_output: dict) -> dict:
    print("\n[Agent 4] Document Requirement Checker — START")

    data                 = agent3_output.get("data", {})
    checklist_result     = agent3_output.get("checklist_result", {})
    canonical_requirements = agent3_output.get("canonical_requirements", [])
    policy_analysis      = agent3_output.get("policy_analysis", {})
    optional_missing     = agent3_output.get("optional_missing", [])

    # Get Agent 1's rich content structures
    document_content_map = data.get("document_content_map", {})
    completeness_summary = data.get("completeness_summary", {})
    null_fields_by_doc   = data.get("null_fields_by_doc", {})
    field_source_map     = data.get("field_source_map", {})

    # ── Step 1: Re-verify each canonical requirement with content values ───────
    print("[Agent 4] Verifying canonical requirements against document content...")

    verified_present     = []
    content_gaps         = []   # doc present but key fields are null/empty
    truly_missing        = []   # doc type not submitted at all

    for req in canonical_requirements:
        if req.get("currently_available"):
            # Checklist said it's satisfied — verify the content is actually there
            satisfying_docs = req.get("satisfying_docs", [req.get("document_type")])
            found_with_content = False

            for doc_type in satisfying_docs:
                content = document_content_map.get(doc_type, {})
                if not content:
                    continue

                check = _check_document_completeness(doc_type, content)

                if check["is_complete"] or check["completeness_score"] >= 0.5:
                    verified_present.append({
                        "requirement":      req["document_name"],
                        "satisfied_by":     doc_type,
                        "completeness":     check["completeness_score"],
                    })
                    found_with_content = True
                    break
                elif check["is_partial"]:
                    # Present but has missing critical fields
                    content_gaps.append({
                        "requirement":       req["document_name"],
                        "document_type":     doc_type,
                        "completeness":      check["completeness_score"],
                        "missing_fields":    check["missing_critical_fields"],
                        "present_fields":    check["present_critical_fields"],
                        "is_blocker":        True,
                        "user_instruction":  _build_user_fix_instruction(
                            doc_type, check["missing_critical_fields"]
                        ),
                    })
                    found_with_content = True   # doc exists, just partial
                    break

            if not found_with_content:
                truly_missing.append({
                    "document_name":   req["document_name"],
                    "document_type":   req["document_type"],
                    "info_required":   req["info_to_include"],
                    "why_needed":      f"Required for pre-authorization step: {req['document_name']}",
                    "satisfying_docs": req.get("satisfying_docs", []),
                })
        else:
            # Checklist already flagged as not satisfied
            # Pull the detailed message from the missing_required list
            checklist_missing = checklist_result.get("missing_required", [])
            msg = next(
                (m["missing_msg"] for m in checklist_missing
                 if m["step"] == _find_step_for_req(req, checklist_missing)),
                f"Required document for: {req['document_name']}"
            )
            truly_missing.append({
                "document_name":   req["document_name"],
                "document_type":   req["document_type"],
                "info_required":   req["info_to_include"],
                "why_needed":      msg,
                "satisfying_docs": req.get("satisfying_docs", []),
            })

    # ── Step 2: Check completeness of ALL submitted documents ─────────────────
    print("[Agent 4] Checking completeness of submitted documents...")
    all_doc_checks = {}

    for doc_type, content in document_content_map.items():
        check = _check_document_completeness(doc_type, content)
        all_doc_checks[doc_type] = check
        if check["missing_critical_fields"]:
            # Only add to content_gaps if not already there
            already_flagged = any(
                g["document_type"] == doc_type for g in content_gaps
            )
            if not already_flagged and check["is_partial"]:
                content_gaps.append({
                    "requirement":       f"{doc_type} completeness",
                    "document_type":     doc_type,
                    "completeness":      check["completeness_score"],
                    "missing_fields":    check["missing_critical_fields"],
                    "present_fields":    check["present_critical_fields"],
                    "is_blocker":        check["completeness_score"] < 0.5,
                    "user_instruction":  _build_user_fix_instruction(
                        doc_type, check["missing_critical_fields"]
                    ),
                })

    # Separate blocking partials from soft warnings
    partial_blocking = [g for g in content_gaps if g.get("is_blocker")]
    partial_soft     = [g for g in content_gaps if not g.get("is_blocker")]

    # ── Step 3: Build user action list ───────────────────────────────────────
    user_actions = []

    for doc in truly_missing:
        user_actions.append({
            "priority":   "HIGH",
            "item":       f"Submit: {doc['document_name']}",
            "reason":     doc["why_needed"],
            "document_to_update_or_provide": doc["document_type"],
            "specific_fields_needed":        [],
            "how_to_resolve": (
                f"Please provide a {doc['document_type']} that contains: "
                f"{doc['info_required']}. "
                f"Any of these document types would satisfy this: "
                f"{', '.join(doc['satisfying_docs'])}."
            ),
        })

    for gap in partial_blocking:
        user_actions.append({
            "priority":   "HIGH",
            "item":       f"Update: {gap['document_type']}",
            "reason":     f"Document is incomplete — missing: {gap['missing_fields']}",
            "document_to_update_or_provide": gap["document_type"],
            "specific_fields_needed":        gap["missing_fields"],
            "how_to_resolve": gap["user_instruction"],
        })

    for gap in partial_soft:
        user_actions.append({
            "priority":   "LOW",
            "item":       f"Recommended: complete {gap['document_type']}",
            "reason":     f"Some fields are empty: {gap['missing_fields']}",
            "document_to_update_or_provide": gap["document_type"],
            "specific_fields_needed":        gap["missing_fields"],
            "how_to_resolve": gap["user_instruction"],
        })

    for opt in optional_missing:
        user_actions.append({
            "priority":   "LOW",
            "item":       f"Optional: {opt['document_name']}",
            "reason":     "Not required but strengthens the application",
            "document_to_update_or_provide": opt["document_type"],
            "specific_fields_needed":        [],
            "how_to_resolve": opt["why_needed"],
        })

    # ── Determine can_proceed ─────────────────────────────────────────────────
    can_proceed = len(truly_missing) == 0 and len(partial_blocking) == 0

    output = {
        "all_docs_present":       can_proceed,
        "can_proceed":            can_proceed,

        # ── Verified results ──────────────────────────────────────────────
        "verified_present":       verified_present,
        "missing_docs":           truly_missing,
        "partial_docs_blocking":  partial_blocking,
        "partial_docs_soft":      partial_soft,
        "document_completeness":  all_doc_checks,

        # ── User actions ──────────────────────────────────────────────────
        "user_action_required":   user_actions,
        "blockers":               [
            {"blocker": u["reason"], "document_needed": u["document_to_update_or_provide"],
             "action_for_user": u["how_to_resolve"]}
            for u in user_actions if u["priority"] == "HIGH"
        ],

        # ── Pass-through ──────────────────────────────────────────────────
        "available_inventory":    document_content_map,
        "field_source_map":       field_source_map,
        "missing_information":    [],
        "data":                   data,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"  Can proceed              : {can_proceed}")
    print(f"  Verified present         : {len(verified_present)}")
    print(f"  Truly missing            : {len(truly_missing)}")
    print(f"  Partial (blocking)       : {len(partial_blocking)}")
    print(f"  Partial (soft)           : {len(partial_soft)}")

    if not can_proceed:
        print("\n" + "!" * 70)
        print("  ACTION REQUIRED — Cannot proceed with submission")
        print("!" * 70)

        if truly_missing:
            print("\n  📋 MISSING DOCUMENTS:")
            for doc in truly_missing:
                print(f"\n    ▸ [{doc['document_type']}] {doc['document_name']}")
                print(f"      Must contain : {doc['info_required']}")
                print(f"      Why needed   : {doc['why_needed'][:80]}")
                if doc.get("satisfying_docs"):
                    print(f"      Can be in any of: {doc['satisfying_docs']}")

        if partial_blocking:
            print("\n  ✏️  INCOMPLETE DOCUMENTS (blocking):")
            for p in partial_blocking:
                score_pct = int(p.get("completeness", 0) * 100)
                print(f"\n    ▸ {p['document_type']}  ({score_pct}% complete)")
                print(f"      Missing      : {p['missing_fields']}")
                print(f"      Action       : {p['user_instruction']}")

        print("!" * 70)

    elif partial_soft:
        print("\n  ⚠️  Soft warnings (non-blocking):")
        for p in partial_soft:
            print(f"    • {p['document_type']} — missing: {p['missing_fields']}")

    print("[Agent 4] Document Requirement Checker — DONE\n")
    return output


def _find_step_for_req(req: dict, missing_list: list) -> str:
    """Helper to match a canonical req to a checklist step."""
    name = req.get("document_name", "").lower()
    for m in missing_list:
        if m.get("label", "").lower() in name or name in m.get("label", "").lower():
            return m.get("step", "")
    return ""
