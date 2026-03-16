"""
agent4_document_checker.py
───────────────────────────
Agent 4 — Missing Document Checker

Role:
  Given:
    • documents             — prose summaries from Agent 1
    • document_requirements — list from Agent 3 describing what docs must be submitted
                              and what information each must contain

  This agent uses an LLM to compare the submitted document content against each
  document requirement and identifies:
    • Which requirements are SATISFIED by the submitted documents
    • Which documents are MISSING (no submitted document covers the requirement)
    • Which documents are PARTIALLY PRESENT (document exists but key info is absent)

  Each missing/partial document gets:
    • priority       — HIGH (blocks approval) or LOW (recommended)
    • info_needed    — exactly what information is required
    • reason         — why it is needed for pre-authorization

  This agent checks ONLY document-related requirements.
  Medical/clinical requirements are checked by Agent 5.

Input  : dict from Agent 3
Output : {
    "can_proceed":        bool,
    "satisfied":          [{requirement summary}],
    "missing_documents":  [{document_type, priority, info_needed, reason}],
    "partial_documents":  [{document_type, priority, missing_info, present_info}],
    "documents":          [agent1 docs — passed through],
    "document_requirements": [passed through for reference],
    "data":               agent3 output (passed through)
  }
"""

import json
import re

from bedrock_client import invoke, MICRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a prior-authorization document reviewer. "
            "You check whether submitted medical documents satisfy specified requirements. "
            "Compare each requirement against the actual document content and determine "
            "whether it is fully satisfied, partially satisfied, or missing entirely. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_check_prompt(doc_requirements: list, documents: list) -> str:
    req_block = "\n".join(
        f"{i+1}. [{r.get('document_type')}]\n"
        f"   Purpose     : {r.get('purpose')}\n"
        f"   Info needed : {r.get('info_needed')}"
        for i, r in enumerate(doc_requirements)
    )

    doc_block = "\n\n".join(
        f"--- {d['document_type']} ---\n{d['content']}"
        for d in documents
    )

    return f"""You are reviewing submitted medical documents against pre-authorization requirements.

DOCUMENT REQUIREMENTS (what must be submitted):
{req_block}

SUBMITTED DOCUMENTS (actual content):
{doc_block}

For EACH requirement above, determine:
- SATISFIED: a submitted document fully covers this requirement with the needed information
- PARTIAL: a submitted document partially covers this (exists but some info is missing)
- MISSING: no submitted document satisfies this requirement at all

For each result, identify which submitted document (if any) covers it, and what specific
information is present vs what is still missing.

Return JSON:
{{
  "checks": [
    {{
      "requirement_index": 1,
      "document_type_required": "string",
      "status": "satisfied" | "partial" | "missing",
      "satisfied_by": "document type from submitted docs that covers this, or null",
      "info_present": "what relevant information was found",
      "info_missing": "what specific information is still absent, or null if satisfied",
      "priority": "HIGH" | "LOW",
      "priority_reason": "why HIGH or LOW priority"
    }}
  ]
}}

Assign HIGH priority to requirements that are critical for approval (insurance info,
diagnosis, procedure, physician credentials). LOW priority for supplementary documents.
"""


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
    print(f"  [Agent4] ❌ JSON parse failed. Raw: {raw[:300]}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent3_output: dict) -> dict:
    print("\n[Agent 4] Missing Document Checker — START")

    doc_requirements = agent3_output.get("document_requirements", [])
    documents        = agent3_output.get("documents", [])

    print(f"  Document requirements to check : {len(doc_requirements)}")
    print(f"  Submitted documents            : {len(documents)}")

    if not doc_requirements:
        print("  [Agent4] No document requirements to check — proceeding.")
        return {
            "can_proceed":         True,
            "satisfied":           [],
            "missing_documents":   [],
            "partial_documents":   [],
            "documents":           documents,
            "document_requirements": doc_requirements,
            "data":                agent3_output,
        }

    print("[Agent 4] Calling Nova Micro — checking document requirements against submitted content...")
    prompt   = _build_check_prompt(doc_requirements, documents)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    raw = invoke(
        model_id=MICRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.1,
    )

    parsed = _safe_parse(raw)
    checks = parsed.get("checks", [])

    # Categorise results
    satisfied  = []
    missing    = []
    partial    = []

    for check in checks:
        idx    = check.get("requirement_index", 1) - 1
        req    = doc_requirements[idx] if 0 <= idx < len(doc_requirements) else {}
        status = check.get("status", "missing").lower()

        entry = {
            "document_type":  check.get("document_type_required", req.get("document_type", "Unknown")),
            "satisfied_by":   check.get("satisfied_by"),
            "info_present":   check.get("info_present", ""),
            "info_missing":   check.get("info_missing"),
            "priority":       check.get("priority", "HIGH"),
            "priority_reason": check.get("priority_reason", ""),
            "purpose":        req.get("purpose", ""),
            "info_needed":    req.get("info_needed", ""),
        }

        if status == "satisfied":
            satisfied.append(entry)
        elif status == "partial":
            partial.append(entry)
        else:
            missing.append(entry)

    # can_proceed = no HIGH priority items are missing or partial
    blocking = [
        item for item in (missing + partial)
        if item.get("priority") == "HIGH"
    ]
    can_proceed = len(blocking) == 0

    # Print summary
    print(f"\n  Results:")
    for s in satisfied:
        print(f"    ✅ [{s['document_type']}] — satisfied by: {s['satisfied_by']}")
    for p in partial:
        print(f"    ⚠️  [{p['document_type']}] — PARTIAL ({p['priority']}) missing: {p['info_missing']}")
    for m in missing:
        print(f"    ❌ [{m['document_type']}] — MISSING ({m['priority']})")

    print(f"\n  Can proceed : {can_proceed}")
    if not can_proceed:
        print(f"  Blockers    : {len(blocking)} HIGH priority item(s) missing/incomplete")
        print("\n" + "!" * 70)
        print("  ACTION REQUIRED — Cannot proceed with submission")
        print("!" * 70)
        for item in blocking:
            print(f"\n  🔴 {item['document_type']}  [{item['priority']}]")
            print(f"     Purpose      : {item['purpose']}")
            print(f"     Info needed  : {item['info_needed']}")
            if item.get("info_missing"):
                print(f"     Still missing: {item['info_missing']}")
        print("!" * 70)

    output = {
        "can_proceed":            can_proceed,
        "satisfied":              satisfied,
        "missing_documents":      missing,
        "partial_documents":      partial,
        # Pass-through for downstream agents
        "documents":              documents,
        "document_requirements":  doc_requirements,
        "data":                   agent3_output,
    }

    print("[Agent 4] Missing Document Checker — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "authorization_required": True,
        "procedure_identified": "MRI Lumbar Spine Without Contrast",
        "document_requirements": [
            {
                "document_type": "Doctor's Clinical Notes",
                "purpose": "Establish medical necessity for the procedure",
                "info_needed": "Diagnosis (ICD-10), clinical findings, symptom duration, prior treatments tried, CPT code for requested procedure",
            },
            {
                "document_type": "Insurance Card",
                "purpose": "Identify the correct insurance policy",
                "info_needed": "Insurer name, policy number, member ID, group number",
            },
            {
                "document_type": "Pretreatment Cost Estimate",
                "purpose": "Document expected costs for the procedure",
                "info_needed": "Itemized cost breakdown with CPT codes and total estimated cost",
            },
        ],
        "medical_requirements": [],
        "policy_search_fields": {"insurer_name": "BlueCross BlueShield"},
        "documents": [
            {
                "document_type": "Doctor's Clinical Notes",
                "content": (
                    "Dr. Sarah Jenkins notes. Patient John R. Doe, lumbar radiculopathy M54.16. "
                    "Symptoms 7 weeks. Tried NSAIDs 4 weeks and PT 3 weeks. Requesting MRI CPT 72148."
                ),
            },
            {
                "document_type": "Insurance Card",
                "content": "BlueCross BlueShield PPO. Member: John R. Doe. ID: BCBS-9900122. Policy: BCBS-99001122.",
            },
        ],
    }
    result = run(mock_input)
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("documents", "data")},
        indent=2
    ))
