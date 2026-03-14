"""
agent4_document_checker.py
───────────────────────────
Agent 4 — Document Requirement Checker
• Compares documents already provided vs documents required by policy
• Identifies exactly what is missing before submission
• Uses Nova Micro (text-only, fast)

Input  : dict from Agent 3
Output : dict  { "all_docs_present": bool, "missing_docs": [...], "present_docs": [...] }
"""

import json

from bedrock_client import invoke, MICRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Document registry — what we have from Agent 1's extraction
# ─────────────────────────────────────────────────────────────────────────────

def _inventory_available_docs(data: dict) -> dict:
    """
    Build a map of which document types we have evidence for,
    based on what Agent 1 successfully extracted.
    """
    available = {}

    # Lab report
    lab = data.get("lab_results", {})
    available["lab_report"] = any(v for v in lab.values() if v)

    # Doctor notes / office visit notes
    available["office_visit_notes"] = bool(
        data.get("clinical_findings") or data.get("diagnosis")
    )

    # Patient info sheet
    available["patient_information_sheet"] = bool(
        data.get("patient_name") and data.get("patient_dob")
    )

    # Insurance card
    available["insurance_card"] = bool(
        data.get("insurer") and data.get("policy_number")
    )

    # Pre-treatment cost estimate
    available["cost_estimate"] = bool(data.get("total_estimated_cost"))

    # Prior treatment documentation
    available["prior_treatment_documentation"] = bool(
        data.get("prior_treatments") and len(data.get("prior_treatments", [])) > 0
    )

    # Physician NPI (we have name + specialty but NPI not in demo docs)
    available["referring_physician_details"] = bool(data.get("ordering_physician"))

    # Procedure order / prescription
    available["procedure_order"] = bool(data.get("procedure") and data.get("cpt"))

    return available


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a prior-authorization document specialist. "
            "Compare available documents against policy requirements and identify gaps. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_prompt(required_docs: list, available_docs: dict, policy_notes: str) -> str:
    return f"""You are checking documentation completeness for a prior-authorization request.

DOCUMENTS REQUIRED BY POLICY:
{json.dumps(required_docs, indent=2)}

DOCUMENTS WE CURRENTLY HAVE (True = present, False = missing):
{json.dumps(available_docs, indent=2)}

POLICY NOTES:
{policy_notes}

For each required document, determine if we have it or an equivalent.
Be flexible — e.g. "office visit notes" covers "physician notes" or "clinical findings".

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
  "recommendations": [string]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent3_output: dict) -> dict:
    """
    Agent 4 entry point.

    Parameters
    ----------
    agent3_output : dict — output from Agent 3

    Returns
    -------
    dict with document completeness status
    """
    print("\n[Agent 4] Document Requirement Checker — START")

    data           = agent3_output.get("data", {})
    policy_analysis = agent3_output.get("policy_analysis", {})
    required_docs  = policy_analysis.get("required_documents", [])
    policy_notes   = policy_analysis.get("policy_notes", "")

    # Fallback required docs if policy analysis didn't return them
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

    prompt = _build_prompt(required_docs, available_docs, policy_notes)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    print("[Agent 4] Calling Nova Micro for document gap analysis...")
    raw = invoke(
        model_id=MICRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=800,
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
            "all_docs_present": False,
            "missing_docs": ["Parse error — manual review required"],
            "present_docs": present,
        }

    output = {
        "all_docs_present":  doc_analysis.get("all_docs_present", False),
        "missing_docs":      doc_analysis.get("missing_docs", []),
        "partial_docs":      doc_analysis.get("partial_docs", []),
        "present_docs":      doc_analysis.get("present_docs", []),
        "blockers":          doc_analysis.get("blockers", []),
        "recommendations":   doc_analysis.get("recommendations", []),
        "document_status":   doc_analysis.get("document_status", {}),
        "available_inventory": available_docs,
        "data":              data,
    }

    print(f"  All docs present : {output['all_docs_present']}")
    print(f"  Missing          : {output['missing_docs']}")
    print(f"  Blockers         : {output['blockers']}")
    print("[Agent 4] Document Requirement Checker — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "data": {
            "patient_name": "John R. Doe",
            "patient_dob": "1985-05-12",
            "diagnosis": "Lumbar Radiculopathy",
            "icd10_codes": ["M54.16"],
            "cpt": "72148",
            "ordering_physician": "Dr. Sarah Jenkins",
            "prior_treatments": ["NSAIDs 4 weeks", "PT 3 weeks"],
            "total_estimated_cost": 2025,
            "clinical_findings": "Positive SLR at 30 degrees",
            "lab_results": {"hba1c": "5.6%"},
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
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
