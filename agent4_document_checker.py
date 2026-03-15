"""
agent4_document_checker.py
───────────────────────────
Agent 4 — Document Requirement Checker
• Matches each policy-required document against the actual document list from Agent 1
• For each gap: says exactly which document type is needed and what info it must contain
• Merges unmet requirements and missing info from Agent 3
• Hard stop if blockers exist — prints exactly what user must submit AND what doc type to use
• Uses Nova Micro

Input  : dict from Agent 3
Output : dict  { "all_docs_present": bool, "missing_docs": [...], "can_proceed": bool }
"""

import json

from bedrock_client import invoke, MICRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Document inventory — uses Agent 1's document list directly
# ─────────────────────────────────────────────────────────────────────────────

def _inventory_available_docs(data: dict) -> dict:
    """
    Returns a dict of:
      { document_type_string: { keys in content } }
    Built from Agent 1's documents list.
    Falls back to documents_present flags if documents list is absent.
    """
    documents = data.get("documents", [])
    if documents:
        return {
            doc.get("document_type", f"Document_{i}"): list(doc.get("content", {}).keys())
            for i, doc in enumerate(documents)
        }

    # Fallback: convert documents_present flags into a simple inventory
    docs_present = data.get("documents_present", {})
    return {
        doc_type: ["(present but content unknown)"]
        for doc_type, present in docs_present.items()
        if present
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a prior-authorization document specialist. "
            "Compare available documents against policy requirements and identify gaps. "
            "For each gap, specify exactly what document type is needed and what information "
            "it must contain. Be precise. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_prompt(required_docs: list, available_doc_inventory: dict,
                  policy_notes: str, unmet_requirements: list,
                  missing_information: list) -> str:

    return f"""You are checking documentation completeness for a prior-authorization request.

DOCUMENTS REQUIRED BY POLICY:
(Each entry specifies the document name, type, and what info it must contain)
{json.dumps(required_docs, indent=2)}

DOCUMENTS WE CURRENTLY HAVE:
(Document type → list of fields/content available in that document)
{json.dumps(available_doc_inventory, indent=2)}

UNMET CLINICAL REQUIREMENTS (from policy analysis):
{json.dumps(unmet_requirements, indent=2)}

MISSING INFORMATION (from policy analysis):
{json.dumps(missing_information, indent=2)}

POLICY NOTES:
{policy_notes}

Instructions:
1. For each required document, check if any available document covers it (be flexible —
   "physician notes" covers "clinical notes", "doctor notes", "procedure template", etc.)
2. For each gap, specify:
   - The exact document name required
   - The document TYPE (e.g. "Clinical Notes", "Lab Report", "Authorization Form")
   - The specific information that document must contain
   - Why it is needed for the prior-auth
3. For each unmet requirement, identify what document the user needs to provide or update.

Return JSON:
{{
  "document_status": {{
    "<required_doc_name>": {{
      "status": "present" | "missing" | "partial",
      "matched_to": string | null,
      "matched_document_type": string | null,
      "missing_info": [string],
      "notes": string
    }}
  }},
  "all_docs_present": true | false,
  "missing_docs": [
    {{
      "document_name": string,
      "document_type": string,
      "info_required": string,
      "why_needed": string
    }}
  ],
  "partial_docs": [
    {{
      "document_name": string,
      "document_type": string,
      "what_is_present": string,
      "what_is_missing": string
    }}
  ],
  "present_docs": [string],
  "blockers": [
    {{
      "blocker": string,
      "document_needed": string,
      "document_type": string,
      "info_to_include": string
    }}
  ],
  "user_action_required": [
    {{
      "item": string,
      "reason": string,
      "document_to_provide": string,
      "document_type": string,
      "info_to_include": string,
      "how_to_resolve": string
    }}
  ],
  "recommendations": [string]
}}"""


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

    # Normalize required_docs — support both old (list of strings) and new (list of dicts)
    normalized_required = []
    for item in required_docs:
        if isinstance(item, str):
            normalized_required.append({
                "document_name":    item,
                "document_type":    "Unknown",
                "info_to_include":  "",
                "currently_available": False,
            })
        else:
            normalized_required.append(item)

    if not normalized_required:
        normalized_required = [
            {
                "document_name":   "Completed prior authorization request form",
                "document_type":   "Authorization Form",
                "info_to_include": "Patient ID, insurer, CPT, ICD-10, physician NPI",
                "currently_available": False,
            },
            {
                "document_name":   "Office visit notes (within 60 days)",
                "document_type":   "Clinical Notes",
                "info_to_include": "Diagnosis, clinical findings, treatment history",
                "currently_available": False,
            },
            {
                "document_name":   "Documentation of conservative treatment failure",
                "document_type":   "Clinical Notes / Treatment Records",
                "info_to_include": "Prior treatments tried, duration, outcome",
                "currently_available": False,
            },
        ]

    print(f"  Required docs ({len(normalized_required)}):")
    for d in normalized_required:
        print(f"    [{d.get('document_type')}] {d.get('document_name')}")

    available_inventory = _inventory_available_docs(data)
    print(f"  Available documents: {list(available_inventory.keys())}")

    prompt   = _build_prompt(normalized_required, available_inventory, policy_notes,
                              unmet_requirements, missing_information)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    print("[Agent 4] Calling Nova Micro for document gap analysis...")
    raw = invoke(
        model_id=MICRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=1200,
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
            "all_docs_present":    False,
            "missing_docs":        [{"document_name": "Parse error — manual review required",
                                     "document_type": "Unknown", "info_required": "", "why_needed": ""}],
            "present_docs":        list(available_inventory.keys()),
            "blockers":            [{"blocker": "Could not parse document analysis",
                                     "document_needed": "Unknown", "document_type": "Unknown",
                                     "info_to_include": ""}],
            "user_action_required": [],
        }

    # Normalize blockers — support both string and dict formats
    raw_blockers = doc_analysis.get("blockers", [])
    blockers_normalized = []
    for b in raw_blockers:
        if isinstance(b, str):
            blockers_normalized.append({
                "blocker":        b,
                "document_needed": "Unknown",
                "document_type":  "Unknown",
                "info_to_include": "",
            })
        else:
            blockers_normalized.append(b)

    # Merge blockers from Agent 3's unmet requirements
    for u in unmet_requirements:
        blockers_normalized.append({
            "blocker":         f"{u.get('requirement')} — {u.get('what_is_needed')}",
            "document_needed": u.get("document_needed", "Unknown"),
            "document_type":   "Clinical Documentation",
            "info_to_include": u.get("what_is_needed", ""),
        })

    # Merge missing information items into blockers
    for m in missing_information:
        if isinstance(m, dict):
            blockers_normalized.append({
                "blocker":         f"Missing: {m.get('info')}",
                "document_needed": m.get("should_be_in_document", "Unknown"),
                "document_type":   m.get("should_be_in_document", "Unknown"),
                "info_to_include": m.get("info", ""),
            })
        else:
            blockers_normalized.append({
                "blocker":         f"Missing information: {m}",
                "document_needed": "Unknown",
                "document_type":   "Unknown",
                "info_to_include": str(m),
            })

    # Merge missing_docs from Agent 3
    missing_docs = doc_analysis.get("missing_docs", [])
    a3_missing   = agent3_output.get("missing_documents", [])
    for item in a3_missing:
        if isinstance(item, dict):
            # Avoid duplicates by document_name
            existing_names = [d.get("document_name") for d in missing_docs]
            if item.get("document_name") not in existing_names:
                missing_docs.append(item)
        elif isinstance(item, str):
            existing_names = [d.get("document_name") for d in missing_docs]
            if item not in existing_names:
                missing_docs.append({
                    "document_name": item,
                    "document_type": "Unknown",
                    "info_required": "",
                    "why_needed": "",
                })

    output = {
        "all_docs_present":    doc_analysis.get("all_docs_present", False),
        "missing_docs":        missing_docs,
        "partial_docs":        doc_analysis.get("partial_docs", []),
        "present_docs":        doc_analysis.get("present_docs", []),
        "blockers":            blockers_normalized,
        "user_action_required": doc_analysis.get("user_action_required", []),
        "recommendations":     doc_analysis.get("recommendations", []),
        "document_status":     doc_analysis.get("document_status", {}),
        "available_inventory": available_inventory,
        "missing_information": missing_information,
        "data":                data,
    }

    print(f"  All docs present : {output['all_docs_present']}")

    # Hard stop output — print clearly what the user must do
    if output["blockers"] or output["missing_docs"]:
        print("\n" + "!" * 70)
        print("  ACTION REQUIRED — Cannot proceed with submission")
        print("!" * 70)

        if output["missing_docs"]:
            print("\n  Missing Documents — please submit the following:")
            for doc in output["missing_docs"]:
                if isinstance(doc, dict):
                    print(f"    * [{doc.get('document_type', 'Unknown')}] {doc.get('document_name')}")
                    if doc.get("info_required"):
                        print(f"      Must contain : {doc.get('info_required')}")
                    if doc.get("why_needed"):
                        print(f"      Why needed   : {doc.get('why_needed')}")
                else:
                    print(f"    * {doc}")

        if output["blockers"]:
            print("\n  Blockers — must be resolved before authorization:")
            for b in output["blockers"]:
                if isinstance(b, dict):
                    print(f"    [BLOCKER] {b.get('blocker')}")
                    if b.get("document_needed") and b.get("document_needed") != "Unknown":
                        print(f"      Document needed : [{b.get('document_type')}] {b.get('document_needed')}")
                    if b.get("info_to_include"):
                        print(f"      Must contain    : {b.get('info_to_include')}")
                else:
                    print(f"    [BLOCKER] {b}")

        if output.get("user_action_required"):
            print("\n  Specific Actions Required:")
            for action in output["user_action_required"]:
                print(f"    > {action.get('item')}")
                print(f"      Reason          : {action.get('reason')}")
                if action.get("document_to_provide"):
                    print(f"      Provide document: [{action.get('document_type')}] {action.get('document_to_provide')}")
                    if action.get("info_to_include"):
                        print(f"      Must contain    : {action.get('info_to_include')}")
                print(f"      How to fix      : {action.get('how_to_resolve')}")

        print("!" * 70 + "\n")

    output["can_proceed"] = len(output["blockers"]) == 0 and len(output["missing_docs"]) == 0

    print(f"  Can proceed      : {output['can_proceed']}")
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
            "documents": [
                {
                    "document_type": "Lab Report",
                    "content": {"patient_name": "John R. Doe", "hba1c": "5.6%", "creatinine": "0.95"}
                },
                {
                    "document_type": "Doctor Notes / Procedure Template",
                    "content": {
                        "diagnosis": "Lumbar Radiculopathy",
                        "icd10_codes": ["M54.16"],
                        "cpt_codes": ["72148"],
                        "clinical_findings": "Positive SLR at 30 degrees",
                        "prior_treatments": ["NSAIDs 4 weeks", "PT 3 weeks"],
                    }
                },
                {
                    "document_type": "Patient Information Sheet",
                    "content": {"patient_name": "John R. Doe", "dob": "1985-05-12", "insurance": "BCBS"}
                },
                {
                    "document_type": "Insurance Card",
                    "content": {"policy_number": "BCBS-99001122", "member_id": "BCBS-9900122"}
                },
                {
                    "document_type": "Medical Pretreatment Estimate",
                    "content": {"total_estimated_cost": 2025, "cpt_codes": ["72148", "72148-26"]}
                },
            ],
        },
        "policy_analysis": {
            "required_documents": [
                {
                    "document_name": "Completed prior authorization request form",
                    "document_type": "Authorization Form",
                    "info_to_include": "Patient ID, insurer, CPT, ICD-10, physician NPI",
                    "currently_available": False,
                },
                {
                    "document_name": "Office visit notes within 60 days",
                    "document_type": "Clinical Notes",
                    "info_to_include": "Diagnosis, clinical findings, treatment plan",
                    "currently_available": True,
                },
                {
                    "document_name": "Documentation of conservative treatment failure",
                    "document_type": "Clinical Notes / Treatment Records",
                    "info_to_include": "Prior treatments tried, duration, outcome",
                    "currently_available": True,
                },
            ],
            "policy_notes": "Standard BCBS prior auth for MRI Lumbar Spine.",
        },
        "unmet_requirements": [],
        "missing_documents": [],
        "missing_information": [],
    }
    result = run(mock_input)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
