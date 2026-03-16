"""
orchestrator.py  (v3.0 — Redesigned Pipeline)
───────────────────────────────────────────────
Pre-Authorization Pipeline Orchestrator

Pipeline redesign (v3.0):
  • Agent 1  — Free-form document describer
               Returns: { documents: [{document_type, content (prose)}, ...] }

  • Agent 2  — Policy search info extractor
               Uses Agent 1 prose → extracts structured fields for policy lookup
               Returns: { ready, policy_search_fields, documents }

  • Agent 3  — Policy retrieval + requirement generator
               Looks up policy, identifies procedure, generates two requirement lists:
                 document_requirements  — what docs must be submitted + what info they need
                 medical_requirements   — clinical conditions that must be met
               Returns: { document_requirements, medical_requirements, policy_search_fields, documents }

  • Agent 4  — Missing document checker (LLM-driven)
               Compares document prose against document_requirements
               Returns: { can_proceed, missing_documents, partial_documents }

  • Agent 5  — Medical requirements checker + approval probability
               Checks medical_requirements against document prose
               Returns: { approval_probability, determination, requirements_checked }

  • Agent 6  — EWA Pre-Auth PDF Form Filler
               Fills the 6-page EWA form using policy_search_fields from Agent 2

Usage:
    python orchestrator.py

Or import and call run_pipeline(doc_paths) from another script.
"""

import json
from datetime import datetime
from pathlib import Path

# ── Import all agents ──────────────────────────────────────────────────────────
import agent1_document_intelligence  as agent1
import agent2_policy_checker         as agent2
import agent3_policy_retrieval       as agent3
import agent4_document_checker       as agent4
import agent5_eligibility_reasoning  as agent5
import agent6_form_filler            as agent6


# ─────────────────────────────────────────────────────────────────────────────
# Document paths — update these to match your file locations
# Agent 1 auto-classifies every file from its content alone.
# ─────────────────────────────────────────────────────────────────────────────

DOC_PATHS = {
    "lab_report":            "./data/patient_data/lab_report_ai.png",
    "doctor_notes":          "./data/patient_data/doctors_notes_2weeks.png",
    "patient_info":          "./data/patient_data/patient_ai.png",
    "insurance_card":        "./data/patient_data/insurance_card_ai.png",
    "pretreatment_estimate": "./data/patient_data/medical_pretreatment_estimate_ai.pdf",
    "prescription":          "./data/patient_data/prescription_ai.png",
    "referral_letter":       "./data/patient_data/physician_referral.png",
    "x_ray":                 "./data/patient_data/x_ray.jpg",
}

# ── EWA Pre-Auth PDF form paths ────────────────────────────────────────────────
EWA_FORM_PDF    = "./data/forms/Pre-Auth Form EWA.pdf"
EWA_FIELDS_JSON = "./data/forms/fields_template.json"
OUTPUT_DIR      = "./output"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_stage(stage, name: str, status: str = "RUNNING"):
    icons  = {"RUNNING": "🔄", "DONE": "✅", "SKIP": "⏭", "WARN": "⚠️", "FAIL": "❌"}
    icon   = icons.get(status, "•")
    label  = f"STAGE {stage}"
    print(f"\n{'='*70}")
    print(f"  {icon}  {label} — {name}  [{status}]")
    print(f"{'='*70}")


def _print_halt(reason: str, details: list = None):
    print(f"\n{'!'*70}")
    print(f"  ❌  PIPELINE HALTED — {reason}")
    print(f"{'!'*70}")
    if details:
        for d in details:
            print(f"  {d}")
    print(f"{'!'*70}\n")


def _safe_serialize(obj):
    """Make pipeline results JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    doc_paths: dict = None,
    output_dir: str = OUTPUT_DIR,
    ewa_form_pdf: str = EWA_FORM_PDF,
    ewa_fields_json: str = EWA_FIELDS_JSON,
) -> dict:
    """
    Run the complete pre-authorization pipeline (v3.0).

    Parameters
    ----------
    doc_paths       : dict  — paths to input documents (defaults to DOC_PATHS above)
    output_dir      : str   — directory for all saved output files
    ewa_form_pdf    : str   — path to blank PreAuth_Form_EWA.pdf (6-page EWA form)
    ewa_fields_json : str   — path to fields_template.json with PDF coordinate map

    Returns
    -------
    dict — full pipeline result with all intermediate outputs and final status
    """
    doc_paths  = doc_paths  or DOC_PATHS
    output_dir = str(Path(output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    results    = {}

    print("\n" + "#" * 70)
    print("  🏥  PRE-AUTHORIZATION PIPELINE  —  START  (v3.0)")
    print(f"  Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Docs    : {len(doc_paths)} files")
    print(f"  Output  : {output_dir}")
    print("#" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Document Intelligence
    #   Reads every submitted document and produces a free-form prose description.
    #   Works for any document type including X-rays, scans, photos, PDFs.
    #   Output: { documents: [{document_type, content}, ...] }
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(1, "Document Intelligence")
    try:
        a1 = agent1.run(doc_paths)
    except Exception as e:
        _print_halt("Document intelligence failed", [f"Error: {e}"])
        results["pipeline_status"] = "HALTED_EXTRACTION_FAILED"
        results["error"]           = str(e)
        results["halted_at"]       = "agent1"
        return results

    results["agent1"] = a1
    _print_stage(1, "Document Intelligence", "DONE")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Policy Search Info Extractor
    #   Reads the prose document summaries and extracts the structured fields
    #   needed to identify the correct insurance policy:
    #     patient name/DOB, insurer, policy number, member ID, group number.
    #   Output: { ready, policy_search_fields, documents }
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(2, "Policy Search Info Extractor")
    a2 = agent2.run(a1)
    results["agent2"] = a2
    _print_stage(2, "Policy Search Info Extractor", "DONE")

    if not a2.get("ready"):
        missing = a2.get("missing_critical", [])
        _print_halt(
            "Stage 2 — Cannot identify insurance policy from submitted documents",
            [f"  • Missing: {m}" for m in missing],
        )
        results["pipeline_status"] = "HALTED_POLICY_NOT_IDENTIFIABLE"
        results["missing_critical"] = missing
        results["halted_at"]        = "agent2"
        return results

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Policy Retrieval + Requirement Generator
    #   Given the policy search fields, identifies what procedure/treatment
    #   requires pre-authorization and enumerates TWO requirement lists:
    #     document_requirements — what documents must be submitted + what info each needs
    #     medical_requirements  — clinical conditions to be met (step therapy, etc.)
    #   Does NOT check whether requirements are fulfilled.
    #   Output: { document_requirements, medical_requirements, procedure_identified, ... }
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(3, "Policy Retrieval + Requirement Generator")
    a3 = agent3.run(a2)
    results["agent3"] = a3
    _print_stage(3, "Policy Retrieval + Requirement Generator", "DONE")

    if not a3.get("authorization_required", True):
        print("\n  ℹ️  No pre-authorization required for this procedure.")
        results["pipeline_status"] = "COMPLETE_NO_AUTH_REQUIRED"
        results["note"]            = "Pre-authorization not required for this procedure."
        return results

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Missing Document Checker
    #   Compares the submitted document prose summaries (from Agent 1) against
    #   the document_requirements from Agent 3.
    #   LLM-driven: determines which requirements are satisfied, partially met,
    #   or missing entirely. Assigns HIGH/LOW priority to each gap.
    #   Output: { can_proceed, satisfied, missing_documents, partial_documents }
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(4, "Missing Document Checker")
    a4 = agent4.run(a3)

    if not a4.get("can_proceed", True):
        high_missing  = [m for m in a4.get("missing_documents", [])  if m.get("priority") == "HIGH"]
        high_partial  = [p for p in a4.get("partial_documents", [])  if p.get("priority") == "HIGH"]
        detail_lines  = []

        for item in high_missing:
            detail_lines.append(f"\n  🔴 MISSING: {item['document_type']}")
            detail_lines.append(f"     Purpose    : {item.get('purpose', '')}")
            detail_lines.append(f"     Info needed: {item.get('info_needed', '')}")

        for item in high_partial:
            detail_lines.append(f"\n  🟡 INCOMPLETE: {item['document_type']}")
            detail_lines.append(f"     Still missing: {item.get('info_missing', '')}")

        _print_halt("Stage 4 — Missing or incomplete required documents", detail_lines)
        results["pipeline_status"] = "HALTED_MISSING_DOCUMENTS"
        results["missing_docs"]    = a4.get("missing_documents", [])
        results["partial_docs"]    = a4.get("partial_documents", [])
        results["halted_at"]       = "agent4"
        return results

    results["agent4"] = a4
    _print_stage(4, "Missing Document Checker", "DONE")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 5 — Medical Requirements Checker + Approval Probability
    #   Checks medical_requirements from Agent 3 against the document prose.
    #   Each requirement is assessed as met / partial / not_met with evidence.
    #   Computes an approval_probability (0.0–1.0) and a determination.
    #   Output: { approval_probability, determination, requirements_checked }
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(5, "Medical Requirements + Approval Probability")
    a5 = agent5.run(a4)
    results["agent5"] = a5
    _print_stage(5, "Medical Requirements + Approval Probability", "DONE")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 6 — EWA Pre-Auth PDF Form Filler
    #   Fills the 6-page EWA pre-auth form with all collected data.
    #   Non-blocking: continues if blank EWA PDF is not found.
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage("6", "EWA Pre-Auth PDF Form Filler")
    a6 = {}
    ewa_pdf_path = Path(ewa_form_pdf)

    if not ewa_pdf_path.exists():
        print(f"  ⚠️  EWA form PDF not found at: {ewa_form_pdf}")
        print(f"  Place Pre-Auth Form EWA.pdf at that path to enable PDF filling.")
        print(f"  Continuing without PDF form.")
        results["agent6"] = {"skipped": True, "reason": f"EWA form not found: {ewa_form_pdf}"}
        _print_stage("6", "EWA Pre-Auth PDF Form Filler", "SKIP")

    elif not Path(ewa_fields_json).exists():
        print(f"  ⚠️  Fields template not found at: {ewa_fields_json}")
        results["agent6"] = {"skipped": True, "reason": f"Fields template not found: {ewa_fields_json}"}
        _print_stage("6", "EWA Pre-Auth PDF Form Filler", "SKIP")

    else:
        try:
            a6 = agent6.run(
                agent5_output=a5,
                output_dir=output_dir,
                form_pdf_path=str(ewa_pdf_path),
                fields_template_path=ewa_fields_json,
            )
            results["agent6"] = a6
            if a6.get("filled_pdf_path"):
                _print_stage("6", "EWA Pre-Auth PDF Form Filler", "DONE")
            else:
                _print_stage("6", "EWA Pre-Auth PDF Form Filler", "WARN")
        except Exception as e:
            print(f"  ⚠️  EWA PDF form filling failed: {e}")
            results["agent6"] = {"error": str(e), "filled_pdf_path": None}
            _print_stage("6", "EWA Pre-Auth PDF Form Filler", "WARN")

    # ══════════════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = (datetime.now() - start_time).total_seconds()
    results["pipeline_status"] = "COMPLETE"
    results["elapsed_seconds"] = elapsed

    psf        = a2.get("policy_search_fields", {})
    filled_pdf = a6.get("filled_pdf_path") if a6 else None

    print("\n" + "#" * 70)
    print("  🏥  PRE-AUTHORIZATION PIPELINE  —  COMPLETE  (v3.0)")
    print("#" * 70)
    print(f"\n  Patient             : {psf.get('patient_name')}  DOB: {psf.get('patient_dob')}")
    print(f"  Insurer             : {psf.get('insurer_name')}  |  Member: {psf.get('member_id')}")
    print(f"  Policy number       : {psf.get('policy_number')}  Group: {psf.get('group_number')}")
    print(f"  Diagnosis           : {psf.get('diagnosis')}")
    print(f"  Procedure           : {a3.get('procedure_identified')}")
    print(f"  Approval probability: {a5.get('approval_probability', 0):.0%}")
    print(f"  Determination       : {a5.get('determination')}")
    print(f"  Documents checked   : {len(a4.get('satisfied', []))} satisfied / "
          f"{len(a4.get('missing_documents', []))} missing / "
          f"{len(a4.get('partial_documents', []))} partial")
    print(f"  Medical reqs met    : {len(a5.get('requirements_met', []))} | "
          f"not met: {len(a5.get('requirements_not_met', []))}")
    if filled_pdf:
        print(f"  EWA PDF             : {filled_pdf}")
    else:
        print(f"  EWA PDF             : not generated (check Stage 6 above)")
    print(f"\n  ⏱  Total time       : {elapsed:.1f} seconds")
    print("#" * 70 + "\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save results to JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, output_dir: str = OUTPUT_DIR) -> str:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = str(Path(output_dir) / f"pipeline_results_{ts}.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(_safe_serialize(results), f, indent=2)
    print(f"📁 Full pipeline results saved to: {out_file}")
    return out_file


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_pipeline(
        doc_paths=DOC_PATHS,
        output_dir=OUTPUT_DIR,
        ewa_form_pdf=EWA_FORM_PDF,
        ewa_fields_json=EWA_FIELDS_JSON,
    )
    save_results(results, output_dir=OUTPUT_DIR)
