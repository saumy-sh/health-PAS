"""
agent5_eligibility_reasoning.py
─────────────────────────────────
Agent 5 — Medical Requirements Checker + Approval Probability

Role:
  Given:
    • documents            — prose summaries from Agent 1
    • medical_requirements — list from Agent 3 (clinical conditions to be met,
                             NOT document requirements — those were handled by Agent 4)
    • Agent 4 output       — document check results (can_proceed, missing/partial docs)

  This agent:
    1. Checks each medical/clinical requirement against the evidence in the document
       prose summaries from Agent 1.
    2. Determines which requirements are MET, PARTIALLY MET, or NOT MET, with evidence.
    3. Computes an approval_probability (0.0 – 1.0) based on how many requirements
       are satisfied, weighted by their importance.
    4. Gives an overall recommendation.

  This agent does NOT check document requirements (that was Agent 4's job).

Input  : dict from Agent 4
Output : {
    "approval_probability":    float (0.0 – 1.0),
    "determination":           "APPROVED" | "LIKELY_APPROVED" | "PENDING_REVIEW" | "LIKELY_DENIED" | "DENIED",
    "requirements_checked":    [{requirement, status, evidence, weight}],
    "requirements_met":        [requirement names],
    "requirements_not_met":    [requirement names],
    "denial_reasons":          [strings],
    "recommendation":          str,
    "clinical_summary":        str,
    "data":                    agent4 output (passed through)
  }
"""

import json
import re

from bedrock_client import invoke, PRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a senior medical prior-authorization reviewer. "
            "Your job is to assess whether the patient's medical situation meets "
            "the clinical and medical requirements for pre-authorization approval. "
            "Use ONLY the evidence found in the submitted document summaries. "
            "Be precise: cite specific evidence from the documents. If information "
            "is not mentioned in any document, state it as absent. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


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

    doc_block = "\n\n".join(
        f"--- {d['document_type']} ---\n{d['content']}"
        for d in documents
    )

    doc_status_block = (
        f"  Missing documents  : {[m['document_type'] for m in doc_check.get('missing_documents', [])]}\n"
        f"  Partial documents  : {[p['document_type'] for p in doc_check.get('partial_documents', [])]}\n"
        f"  Can proceed (docs) : {doc_check.get('can_proceed', True)}"
    )

    return f"""Assess whether this prior-authorization request meets the required medical/clinical criteria.

MEDICAL / CLINICAL REQUIREMENTS TO CHECK:
{req_block}

SUBMITTED DOCUMENT SUMMARIES (your evidence base):
{doc_block}

DOCUMENT STATUS FROM PREVIOUS AGENT:
{doc_status_block}

For EACH medical requirement listed above, determine:
- MET: the documents clearly show this requirement is satisfied
- PARTIAL: there is some evidence but it is incomplete or ambiguous
- NOT_MET: the documents show this requirement is clearly not satisfied, or information is absent

Then compute an overall approval probability from 0.0 to 1.0 based on:
- What fraction of REQUIRED requirements are MET (these count more heavily)
- Whether any critical blockers exist (missing documents, not met critical requirements)
- General strength of the clinical evidence presented

Return JSON:
{{
  "requirements_checked": [
    {{
      "requirement": "requirement name",
      "importance": "required" | "recommended",
      "status": "met" | "partial" | "not_met",
      "evidence": "specific text from documents supporting this assessment",
      "notes": "any additional notes or caveats"
    }}
  ],
  "approval_probability": 0.0,
  "determination": "APPROVED" | "LIKELY_APPROVED" | "PENDING_REVIEW" | "LIKELY_DENIED" | "DENIED",
  "denial_reasons": ["list of specific reasons why approval is unlikely or denied"],
  "approval_conditions": ["any conditions that if met would improve probability"],
  "clinical_summary": "one paragraph summarising the clinical picture and recommendation",
  "recommendation": "concise action recommendation"
}}

Probability guidance:
  0.85–1.0 → APPROVED (all required criteria clearly met, strong documentation)
  0.65–0.84 → LIKELY_APPROVED (most criteria met, minor gaps)
  0.40–0.64 → PENDING_REVIEW (mixed — some criteria met, some unclear)
  0.20–0.39 → LIKELY_DENIED (multiple required criteria not met)
  0.0–0.19 → DENIED (critical criteria clearly not met or information entirely absent)
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
    print(f"  [Agent5] ❌ JSON parse failed. Raw: {raw[:300]}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent4_output: dict) -> dict:
    print("\n[Agent 5] Medical Requirements Checker + Approval Probability — START")

    agent3_data        = agent4_output.get("data", {})
    medical_requirements = agent3_data.get("medical_requirements", [])
    documents          = agent4_output.get("documents", [])

    print(f"  Medical requirements to check : {len(medical_requirements)}")
    print(f"  Submitted documents           : {len(documents)}")

    print("[Agent 5] Calling Nova Pro — assessing medical requirements...")
    prompt   = _build_assessment_prompt(medical_requirements, documents, agent4_output)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    raw = invoke(
        model_id=PRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.1,
    )

    parsed = _safe_parse(raw)

    if not parsed:
        # Fallback: base probability on document completeness only
        can_proceed = agent4_output.get("can_proceed", False)
        fallback_prob = 0.5 if can_proceed else 0.2
        parsed = {
            "requirements_checked":  [],
            "approval_probability":  fallback_prob,
            "determination":         "PENDING_REVIEW",
            "denial_reasons":        ["LLM assessment failed — manual review required"],
            "approval_conditions":   [],
            "clinical_summary":      "Assessment could not be completed automatically.",
            "recommendation":        "Refer for manual review.",
        }

    checks         = parsed.get("requirements_checked", [])
    probability    = float(parsed.get("approval_probability", 0.5))
    # Clamp to [0, 1]
    probability    = max(0.0, min(1.0, probability))
    determination  = parsed.get("determination", "PENDING_REVIEW")

    # Adjust probability downward if documents are missing
    if not agent4_output.get("can_proceed", True):
        blocking_count = len(agent4_output.get("missing_documents", [])) + \
                         len([p for p in agent4_output.get("partial_documents", []) if p.get("priority") == "HIGH"])
        probability *= max(0.3, 1.0 - blocking_count * 0.15)
        probability = round(probability, 2)
        if probability < 0.4:
            determination = "PENDING_REVIEW"

    requirements_met     = [c["requirement"] for c in checks if c.get("status") == "met"]
    requirements_not_met = [c["requirement"] for c in checks if c.get("status") == "not_met"]
    requirements_partial = [c["requirement"] for c in checks if c.get("status") == "partial"]

    # Print summary
    print(f"\n  Approval Probability : {probability:.0%}")
    print(f"  Determination        : {determination}")
    print(f"  Requirements met     : {len(requirements_met)}/{len(checks)}")

    if requirements_met:
        for r in requirements_met:
            print(f"    ✅ {r}")
    if requirements_partial:
        for r in requirements_partial:
            print(f"    ⚠️  {r} (partial)")
    if requirements_not_met:
        for r in requirements_not_met:
            print(f"    ❌ {r}")

    denial_reasons = parsed.get("denial_reasons", [])
    if denial_reasons:
        print(f"\n  Denial reasons:")
        for reason in denial_reasons:
            print(f"    • {reason}")

    print(f"\n  Recommendation: {parsed.get('recommendation', 'N/A')}")

    output = {
        "approval_probability":   probability,
        "determination":          determination,
        "requirements_checked":   checks,
        "requirements_met":       requirements_met,
        "requirements_not_met":   requirements_not_met,
        "requirements_partial":   requirements_partial,
        "denial_reasons":         denial_reasons,
        "approval_conditions":    parsed.get("approval_conditions", []),
        "clinical_summary":       parsed.get("clinical_summary", ""),
        "recommendation":         parsed.get("recommendation", ""),
        # Pass-through for Agent 6
        "documents":              documents,
        "data":                   agent4_output,
    }

    print("[Agent 5] Medical Requirements Checker — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "can_proceed": True,
        "missing_documents": [],
        "partial_documents": [],
        "documents": [
            {
                "document_type": "Doctor's Clinical Notes",
                "content": (
                    "Dr. Sarah Jenkins, Orthopedic Surgery. Patient John R. Doe, 38M. "
                    "Lumbar radiculopathy M54.16, symptoms 7 weeks. Pain 7/10. "
                    "Positive SLR. Prior treatments: NSAIDs (Ibuprofen) 4 weeks — minimal relief; "
                    "Physical Therapy 3 weeks — symptoms persisted. Requesting MRI Lumbar CPT 72148."
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
                    "requirement": "NSAID trial",
                    "description": "Patient must have tried NSAIDs for at least 4 weeks with inadequate relief",
                    "threshold": "4 weeks",
                    "importance": "required",
                },
                {
                    "requirement": "Physical therapy trial",
                    "description": "Patient must have completed at least 4 weeks of physical therapy",
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
