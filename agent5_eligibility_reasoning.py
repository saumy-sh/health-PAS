"""
agent5_eligibility_reasoning.py
─────────────────────────────────
Agent 5 — Medical Requirements Checker + Approval Probability

Role:
  Given:
    • documents            — prose summaries from Agent 1
    • medical_requirements — list from Agent 3 (clinical conditions to be met)
    • Agent 4 output       — document check results

  This agent:
    1. Checks EVERY document for each requirement — citing the source document.
    2. When values conflict across documents, ALWAYS flags the conflict and uses
       the MOST CONSERVATIVE (lowest/shortest) value for threshold comparisons.
    3. Determines MET / PARTIAL / NOT_MET with full evidence chain.
    4. Computes approval_probability and determination.

Input  : dict from Agent 4
Output : {
    "approval_probability":  float,
    "determination":         str,
    "requirements_checked":  list,
    "requirements_met":      list,
    "requirements_not_met":  list,
    "requirements_partial":  list,
    "denial_reasons":        list,
    "approval_conditions":   list,
    "clinical_summary":      str,
    "recommendation":        str,
    "data":                  agent4 output (passed through)
  }
"""

import json
import re

from bedrock_client import invoke, PRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a senior medical prior-authorization reviewer at an insurance company. "
            "Your job is to rigorously assess whether the patient's medical situation meets "
            "every clinical requirement for pre-authorization. "
            "\n\n"
            "CRITICAL RULES you must always follow:\n"
            "1. Read EVERY document provided. Do not stop at the first piece of supporting evidence.\n"
            "2. CONFLICT DETECTION: If different documents state different values for the same fact "
            "(e.g. symptom duration is '2 weeks' in one document but '7 weeks' in another), "
            "you MUST flag this as a conflict. List both values and both source documents.\n"
            "3. CONSERVATIVE RULE: When there is a conflict on a numeric threshold (duration, "
            "weeks of treatment, pain score, etc.), you MUST use the LOWEST / SHORTEST / MOST "
            "CONSERVATIVE value for the purpose of determining whether the requirement is met. "
            "Do NOT cherry-pick the value that supports approval.\n"
            "4. CITE YOUR SOURCES: For every piece of evidence, state exactly which document "
            "type it came from (e.g. 'Clinical Note states...', 'Pre-treatment Estimate states...').\n"
            "5. If a requirement threshold is NOT met using the conservative value, mark it NOT_MET "
            "even if another document would suggest it is met.\n"
            "6. Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_assessment_prompt(medical_requirements: list, documents: list,
                              doc_check: dict) -> str:
    if not medical_requirements:
        req_block = "(No specific medical requirements were identified for this procedure.)"
    else:
        req_block = "\n".join(
            f"{i+1}. [{r.get('importance', 'required').upper()}] {r.get('requirement')}\n"
            f"   Description : {r.get('description')}\n"
            f"   Threshold   : {r.get('threshold') or 'not specified'}"
            for i, r in enumerate(medical_requirements)
        )

    # Label each document with its type so the LLM can cite sources precisely
    doc_block = "\n\n".join(
        f"=== DOCUMENT {i+1}: {d['document_type']} ===\n{d['content']}"
        for i, d in enumerate(documents)
    )

    doc_status_block = (
        f"  Missing documents  : {[m['document_type'] for m in doc_check.get('missing_documents', [])]}\n"
        f"  Partial documents  : {[p['document_type'] for p in doc_check.get('partial_documents', [])]}\n"
        f"  Can proceed (docs) : {doc_check.get('can_proceed', True)}"
    )

    return f"""You must assess whether the prior-authorization request meets every medical/clinical requirement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEDICAL / CLINICAL REQUIREMENTS TO CHECK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{req_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBMITTED DOCUMENTS (read ALL of them for each requirement):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{doc_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENT COMPLETENESS STATUS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{doc_status_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSESSMENT INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For EACH requirement:

STEP 1 — SCAN ALL DOCUMENTS
  Search every document for any mention of the relevant fact (duration, treatment, score, etc.).
  List every value found and which document it came from.

STEP 2 — DETECT CONFLICTS
  If different documents give different values for the same fact, this IS a conflict.
  Example: "Clinical Note says 2-week history; Pre-treatment Estimate says 7 weeks" — CONFLICT.
  You MUST report both values and both source documents in the "evidence" field.

STEP 3 — APPLY CONSERVATIVE RULE
  For numeric thresholds (weeks, days, scores): use the LOWEST / SHORTEST value found.
  If the conservative value meets the threshold → MET.
  If the conservative value does NOT meet the threshold → NOT_MET (even if another doc suggests otherwise).
  If only one document mentions the value and it is ambiguous → PARTIAL.

STEP 4 — ASSIGN STATUS
  - MET:     Conservative value clearly meets or exceeds the threshold. No conflicts OR conflict resolved conservatively and still meets threshold.
  - PARTIAL: Some evidence present but incomplete, ambiguous, or conservative value is borderline.
  - NOT_MET: Conservative value does not meet threshold, or information is absent from ALL documents.

STEP 5 — COMPUTE PROBABILITY
  Base the approval_probability on:
  - Fraction of REQUIRED requirements that are MET (these weight heavily)
  - NOT_MET required requirements are strong negative signals
  - PARTIAL requirements reduce probability but don't eliminate it
  - Missing documents further reduce probability
  - Conflicts that resolved NOT_MET count as NOT_MET

Probability scale:
  0.85–1.0  → APPROVED        (all required criteria clearly met, strong consistent documentation)
  0.65–0.84 → LIKELY_APPROVED (most criteria met, minor gaps or soft conflicts)
  0.40–0.64 → PENDING_REVIEW  (mixed — some criteria met, some unclear or conflicting)
  0.20–0.39 → LIKELY_DENIED   (multiple required criteria not met)
  0.0–0.19  → DENIED          (critical criteria clearly not met)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY this exact JSON (no markdown, no extra text):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "requirements_checked": [
    {{
      "requirement": "requirement name",
      "importance": "required" | "recommended",
      "status": "met" | "partial" | "not_met",
      "values_found": [
        {{"source": "Document type name", "value": "exact value found in that document"}}
      ],
      "conflict_detected": true | false,
      "conservative_value_used": "the value used for threshold comparison (lowest/shortest)",
      "evidence": "Full citation: which document said what. If conflict, state both values and both sources explicitly.",
      "notes": "Any additional context, e.g. why this was marked partial instead of not_met"
    }}
  ],
  "approval_probability": 0.0,
  "determination": "APPROVED" | "LIKELY_APPROVED" | "PENDING_REVIEW" | "LIKELY_DENIED" | "DENIED",
  "denial_reasons": ["specific reason if not approved"],
  "approval_conditions": ["what would need to change to improve the determination"],
  "clinical_summary": "One paragraph. Must explicitly mention any conflicting values found across documents.",
  "recommendation": "Concise action recommendation"
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
    print(f"  [Agent5] ❌ JSON parse failed. Raw: {raw[:300]}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing: programmatic conflict check as a safety net
# ─────────────────────────────────────────────────────────────────────────────

def _programmatic_conflict_check(checks: list) -> list:
    """
    Safety net: if the LLM marked a requirement as 'met' but also set
    conflict_detected=True AND the conservative_value_used doesn't meet the
    threshold implied by the requirement description, downgrade to 'partial'.

    This prevents the LLM from saying "conflict detected but still MET"
    when the conservative value is clearly insufficient.
    """
    for check in checks:
        if not check.get("conflict_detected"):
            continue

        values_found = check.get("values_found", [])
        if len(values_found) < 2:
            continue

        # Try to extract numeric values and find the minimum
        nums = []
        for v in values_found:
            raw_val = str(v.get("value", ""))
            # Extract first number from strings like "2 weeks", "7 weeks", "4 weeks"
            match = re.search(r'(\d+(?:\.\d+)?)', raw_val)
            if match:
                nums.append(float(match.group(1)))

        if len(nums) >= 2 and check.get("status") == "met":
            min_val = min(nums)
            # Try to extract threshold number from the conservative_value_used or notes
            conservative_raw = str(check.get("conservative_value_used", ""))
            threshold_match = re.search(r'(\d+(?:\.\d+)?)', str(check.get("notes", "") + conservative_raw))

            # If we found the minimum value is meaningfully lower than the max,
            # flag it — the LLM should have handled this, but as a safety net
            # we downgrade to PARTIAL to force human review
            if max(nums) > 0 and (min_val / max(nums)) < 0.5:
                # Values differ by more than 50% — this is a significant conflict
                # Downgrade from "met" to "partial" to be safe
                print(f"  [Agent5] ⚠ Conflict safety net triggered for '{check.get('requirement')}': "
                      f"values={nums}, min={min_val}. Downgrading from 'met' to 'partial'.")
                check["status"] = "partial"
                check["notes"] = (check.get("notes", "") +
                                  f" [Safety net: conflicting values {nums} detected; "
                                  f"downgraded to PARTIAL pending manual review]")

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent4_output: dict) -> dict:
    print("\n[Agent 5] Medical Requirements Checker + Approval Probability — START")

    agent3_data          = agent4_output.get("data", {})
    medical_requirements = agent3_data.get("medical_requirements", [])
    documents            = agent4_output.get("documents", [])

    print(f"  Medical requirements to check : {len(medical_requirements)}")
    print(f"  Submitted documents           : {len(documents)}")

    print("[Agent 5] Calling Nova Pro — assessing medical requirements...")
    prompt   = _build_assessment_prompt(medical_requirements, documents, agent4_output)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    raw = invoke(
        model_id=PRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=3000,
        temperature=0.0,   # deterministic — no creative interpretation of evidence
    )

    parsed = _safe_parse(raw)

    if not parsed:
        can_proceed   = agent4_output.get("can_proceed", False)
        fallback_prob = 0.5 if can_proceed else 0.2
        parsed = {
            "requirements_checked": [],
            "approval_probability": fallback_prob,
            "determination":        "PENDING_REVIEW",
            "denial_reasons":       ["LLM assessment failed — manual review required"],
            "approval_conditions":  [],
            "clinical_summary":     "Assessment could not be completed automatically.",
            "recommendation":       "Refer for manual review.",
        }

    checks        = parsed.get("requirements_checked", [])
    probability   = float(parsed.get("approval_probability", 0.5))
    probability   = max(0.0, min(1.0, probability))
    determination = parsed.get("determination", "PENDING_REVIEW")

    # Apply programmatic conflict safety net
    checks = _programmatic_conflict_check(checks)

    # Re-compute probability if safety net downgraded any checks
    required_checks  = [c for c in checks if c.get("importance") == "required"]
    if required_checks:
        met_required     = sum(1 for c in required_checks if c.get("status") == "met")
        partial_required = sum(1 for c in required_checks if c.get("status") == "partial")
        not_met_required = sum(1 for c in required_checks if c.get("status") == "not_met")
        total_required   = len(required_checks)

        # Recompute only if safety net made changes — use conservative formula
        if not_met_required > 0 or partial_required > 0:
            base = (met_required + 0.4 * partial_required) / total_required
            # Each NOT_MET required item is a hard penalty
            penalty = not_met_required * 0.25
            recomputed = max(0.0, min(1.0, base - penalty))
            if abs(recomputed - probability) > 0.1:
                print(f"  [Agent5] Recomputing probability: LLM={probability:.2f} → safety_net={recomputed:.2f}")
                probability = round(recomputed, 2)
                # Remap determination
                if probability >= 0.85:
                    determination = "APPROVED"
                elif probability >= 0.65:
                    determination = "LIKELY_APPROVED"
                elif probability >= 0.40:
                    determination = "PENDING_REVIEW"
                elif probability >= 0.20:
                    determination = "LIKELY_DENIED"
                else:
                    determination = "DENIED"

    # Further adjust if documents are missing
    if not agent4_output.get("can_proceed", True):
        blocking_count = (
            len(agent4_output.get("missing_documents", [])) +
            len([p for p in agent4_output.get("partial_documents", []) if p.get("priority") == "HIGH"])
        )
        probability = round(max(0.0, probability * max(0.3, 1.0 - blocking_count * 0.15)), 2)
        if probability < 0.4:
            determination = "PENDING_REVIEW"

    requirements_met     = [c["requirement"] for c in checks if c.get("status") == "met"]
    requirements_not_met = [c["requirement"] for c in checks if c.get("status") == "not_met"]
    requirements_partial = [c["requirement"] for c in checks if c.get("status") == "partial"]
    conflicts_found      = [c["requirement"] for c in checks if c.get("conflict_detected")]

    # Print summary
    print(f"\n  Approval Probability : {probability:.0%}")
    print(f"  Determination        : {determination}")
    print(f"  Requirements met     : {len(requirements_met)}/{len(checks)}")
    if conflicts_found:
        print(f"  ⚠ Conflicts detected : {conflicts_found}")
    for r in requirements_met:
        print(f"    ✅ {r}")
    for r in requirements_partial:
        print(f"    ⚠️  {r} (partial)")
    for r in requirements_not_met:
        print(f"    ❌ {r}")

    denial_reasons = parsed.get("denial_reasons", [])
    if denial_reasons:
        print(f"\n  Denial reasons:")
        for reason in denial_reasons:
            print(f"    • {reason}")

    print(f"\n  Recommendation: {parsed.get('recommendation', 'N/A')}")

    output = {
        "approval_probability":  probability,
        "determination":         determination,
        "requirements_checked":  checks,
        "requirements_met":      requirements_met,
        "requirements_not_met":  requirements_not_met,
        "requirements_partial":  requirements_partial,
        "conflicts_detected":    conflicts_found,
        "denial_reasons":        denial_reasons,
        "approval_conditions":   parsed.get("approval_conditions", []),
        "clinical_summary":      parsed.get("clinical_summary", ""),
        "recommendation":        parsed.get("recommendation", ""),
        # Pass-through for Agent 6
        "documents":             documents,
        "data":                  agent4_output,
    }

    print("[Agent 5] Medical Requirements Checker — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulates the exact conflict in the reported issue:
    # Clinical Note says "2-week history", Pre-treatment Estimate says "7 weeks"
    mock_input = {
        "can_proceed": True,
        "missing_documents": [],
        "partial_documents": [],
        "documents": [
            {
                "document_type": "Clinical Note",
                "content": (
                    "Patient: John R. Doe. 2-week history of severe radiating lower back pain (sciatica). "
                    "Pain Score: 7/10. Positive SLR at 30 degrees. "
                    "Prior treatments: NSAIDs tried. Requesting MRI Lumbar Spine (CPT 72148)."
                ),
            },
            {
                "document_type": "Pre-treatment Estimate",
                "content": (
                    "Patient presents with severe radiating lower back pain (sciatica) for 7 weeks. "
                    "Prior treatments attempted: NSAIDs (Ibuprofen) - 4 weeks - Minimal relief. "
                    "Physical Therapy - 3 weeks - Symptoms persisted. "
                    "Physician recommends MRI Lumbar Spine without Contrast. CPT 72148."
                ),
            },
        ],
        "data": {
            "medical_requirements": [
                {
                    "requirement": "Minimum symptom duration",
                    "description": "Patient must have had symptoms for at least 6 weeks",
                    "threshold": "6 weeks",
                    "importance": "required",
                },
                {
                    "requirement": "Conservative therapy trial",
                    "description": "Patient must have tried NSAIDs or physical therapy for at least 4 weeks",
                    "threshold": "4 weeks",
                    "importance": "required",
                },
            ],
        },
    }
    result = run(mock_input)
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("documents", "data")},
        indent=2
    ))
