"""
agent6_report_generator.py
──────────────────────────
Agent 6 — Final JSON Report Generator

Role:
  Takes all verified data from the pipeline (Agents 1-5) and synthesizes a 
  clean, comprehensive final JSON report of the pre-authorization outcome.
  This report is used for the final determination and display in the UI.

Input  : agent5_output dict (contains references to all previous agents)
Output : dict (The final structured JSON report)
"""

import json
import os
import tempfile
from datetime import datetime
from backend.s3_utils import upload_to_s3

def run(agent5_output: dict) -> dict:
    print("\n[Agent 6] Final Report Generator — START")

    # Navigate the agent chain (data nested by previous agents)
    # agent5.data -> agent4 output
    # agent4.data -> agent3 output
    # agent3.policy_search_fields -> extracted fields
    a4_output = agent5_output.get("data", {})
    a3_output = a4_output.get("data", {})
    psf = a3_output.get("policy_search_fields", {})

    # 1. Patient Info
    patient_info = {
        "name": psf.get("patient_name"),
        "dob": psf.get("patient_dob"),
        "mrn": psf.get("patient_mrn"),
        "gender": psf.get("patient_gender")
    }

    # 2. Insurance Info
    insurance_info = {
        "insurer": psf.get("insurer_name"),
        "plan_type": psf.get("plan_type"),
        "policy_number": psf.get("policy_number"),
        "member_id": psf.get("member_id"),
        "group_number": psf.get("group_number")
    }

    # 3. Requested Procedure
    physician = psf.get("ordering_physician", "Unknown")
    specialty = psf.get("physician_specialty")
    ordering_physician_str = f"{physician} ({specialty})" if specialty else physician

    requested_procedure = {
        "name": a3_output.get("procedure_identified"),
        "cpt_codes": psf.get("cpt_codes", []),
        "icd10_codes": psf.get("icd10_codes", []),
        "diagnosis": psf.get("diagnosis"),
        "ordering_physician": ordering_physician_str
    }

    # 4. Document Requirements Check
    # Map Agent 4 structures to the requested format
    doc_satisfied = []
    for item in a4_output.get("satisfied", []):
        doc_satisfied.append({
            "document_type": item.get("document_type"),
            "satisfied_by": item.get("satisfied_by"),
            "info_present": item.get("info_present")
        })

    doc_partial = []
    for item in a4_output.get("partial_documents", []):
        doc_partial.append({
            "document_type": item.get("document_type"),
            "info_needed": item.get("info_missing"),
            "priority": item.get("priority"),
            "reason": item.get("priority_reason")
        })

    doc_missing = []
    for item in a4_output.get("missing_documents", []):
        doc_missing.append({
            "document_type": item.get("document_type"),
            "info_needed": item.get("info_needed"),
            "priority": item.get("priority"),
            "reason": item.get("priority_reason")
        })

    document_requirements_check = {
        "satisfied": doc_satisfied,
        "partial": doc_partial,
        "missing": doc_missing
    }

    # 5. Medical Requirements Check
    # Map Agent 5 structures to the requested format
    medical_requirements_check = []
    for item in agent5_output.get("requirements_checked", []):
        medical_requirements_check.append({
            "requirement": item.get("requirement"),
            "status": item.get("status"),
            "evidence": item.get("evidence"),
            "importance": item.get("importance")
        })

    # 6. Approval Assessment
    approval_assessment = {
        "probability": agent5_output.get("approval_probability"),
        "determination": agent5_output.get("determination"),
        "denial_reasons": agent5_output.get("denial_reasons", []),
        "clinical_summary": agent5_output.get("clinical_summary"),
        "recommendation": agent5_output.get("recommendation")
    }

    # 7. Next Steps
    next_steps = agent5_output.get("approval_conditions", [])
    if not next_steps:
        if approval_assessment["determination"] == "APPROVED":
            next_steps = ["Proceed with procedure as scheduled.", "Download approval letter for your records."]
        elif approval_assessment["determination"] == "PENDING_REVIEW":
            next_steps = ["Gather missing documents highlighted above.", "Awaiting manual clinical review."]
        else:
            next_steps = ["Review denial reasons with the ordering physician.", "Consider filing an appeal with additional clinical evidence."]

    # Construct the Final Report
    report = {
        "patient_info": patient_info,
        "insurance_info": insurance_info,
        "requested_procedure": requested_procedure,
        "document_requirements_check": document_requirements_check,
        "medical_requirements_check": medical_requirements_check,
        "approval_assessment": approval_assessment,
        "next_steps": next_steps
    }

    # Save to output folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mrn = str(psf.get("patient_mrn", "unknown")).replace("-", "")
    report_name = f"final_preauth_report_{mrn}_{ts}.json"
    
    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    local_path = os.path.join(out_dir, report_name)
    
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"  [Agent 6] Report generated locally: {local_path}")
    
    print("[Agent 6] Final Report Generator — DONE\n")
    return report

if __name__ == "__main__":
    # Minimal mock for testing
    mock_a5 = {
        "approval_probability": 0.92,
        "determination": "APPROVED",
        "requirements_checked": [
            {"requirement": "Symptom duration", "status": "met", "evidence": "7 weeks", "importance": "required"}
        ],
        "denial_reasons": [],
        "clinical_summary": "Strong evidence.",
        "recommendation": "Approve.",
        "data": {
            "satisfied": [{"document_type": "Clinical Notes", "satisfied_by": "Dr Notes", "info_present": "All"}],
            "partial_documents": [],
            "missing_documents": [],
            "data": {
                "procedure_identified": "MRI Lumbar",
                "policy_search_fields": {
                    "patient_name": "John Doe",
                    "patient_dob": "1985-05-12",
                    "insurer_name": "BCBS"
                }
            }
        }
    }
    print(json.dumps(run(mock_a5), indent=2))
