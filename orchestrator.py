"""
orchestrator.py
───────────────
Pre-Authorization Pipeline Orchestrator
Runs all 9 agents in sequence, passing outputs between them.

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
import agent6_form_generator         as agent6
import agent7_portal_automation      as agent7
import agent8_status_tracker         as agent8
import agent9_appeal_agent           as agent9


# ─────────────────────────────────────────────────────────────────────────────
# Document paths — update these to match your file locations
# ─────────────────────────────────────────────────────────────────────────────

DOC_PATHS = {
    "lab_report":            "./data/patient_data/lab_report_ai.png",
    "doctor_notes":          "./data/patient_data/doctor_notes_ai.png",
    "patient_info":          "./data/patient_data/patient_ai.png",
    "insurance_card":        "./data/patient_data/insurance_card_ai.png",
    "pretreatment_estimate": "./data/patient_data/medical_pretreatment_estimate_ai.pdf",
    "prescription":          "./data/patient_data/prescription_ai.png",
    "referral_letter":       "./data/patient_data/physician_referral.png",
}

OUTPUT_DIR = "./output"   # directory for saved forms and appeal letters


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _print_stage(stage: int, name: str, status: str = "RUNNING"):
    icons = {"RUNNING": "🔄", "DONE": "✅", "SKIP": "⏭", "WARN": "⚠️", "FAIL": "❌"}
    icon = icons.get(status, "•")
    print(f"\n{'='*70}")
    print(f"  {icon}  STAGE {stage}/9 — {name}  [{status}]")
    print(f"{'='*70}")


def run_pipeline(doc_paths: dict = None, output_dir: str = ".") -> dict:
    """
    Run the complete 9-agent pre-authorization pipeline.

    Parameters
    ----------
    doc_paths  : dict — paths to input documents (defaults to DOC_PATHS above)
    output_dir : str  — directory for output files

    Returns
    -------
    dict — full pipeline result with all intermediate outputs
    """
    doc_paths  = doc_paths or DOC_PATHS

    # Ensure output directory exists (all saved artifacts should go here)
    output_dir = str(Path(output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    print("\n" + "#" * 70)
    print("  🏥  PRE-AUTHORIZATION PIPELINE  —  START")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    results = {}

    # ── STAGE 1: Document Intelligence ────────────────────────────────────────
    _print_stage(1, "Document Intelligence")
    a1 = agent1.run(doc_paths)
    results["agent1"] = a1
    _print_stage(1, "Document Intelligence", "DONE")

    # ── STAGE 2: Policy Query Requirement Check ────────────────────────────────
    _print_stage(2, "Policy Query Requirement Checker")
    a2 = agent2.run(a1)
    results["agent2"] = a2
    _print_stage(2, "Policy Query Requirement Checker", "DONE")

    if not a2.get("ready"):
        print(f"\n⚠️  Pipeline HALTED at Stage 2.")
        print(f"   Missing required fields: {a2.get('missing_required')}")
        print(f"   Please collect missing information and re-run.\n")
        results["pipeline_status"] = "HALTED_MISSING_FIELDS"
        results["halted_at"]       = "agent2"
        return results

    # ── STAGE 3: Policy Retrieval (RAG) ───────────────────────────────────────
    _print_stage(3, "Policy Retrieval Agent (RAG)")
    a3 = agent3.run(a2)
    results["agent3"] = a3
    _print_stage(3, "Policy Retrieval Agent (RAG)", "DONE")

    # ── STAGE 4: Document Requirement Check ───────────────────────────────────
    _print_stage(4, "Document Requirement Checker")
    # Merge policy_analysis into a3's data so Agent 4 has it
    a3_enriched = dict(a3)
    a3_enriched["policy_analysis"] = a3.get("policy_analysis", {})
    a4 = agent4.run(a3_enriched)
    if not a4.get("can_proceed", True):
        print("\n[PIPELINE HALTED] Missing required documents.")
        print("Please collect the following and re-run:\n")
        for doc in a4.get("missing_docs", []):
            print(f"  * {doc}")
        results["pipeline_status"] = "HALTED_MISSING_DOCUMENTS"
        results["missing_docs"]    = a4.get("missing_docs", [])
        results["halted_at"]       = "agent4"
        return results
    results["agent4"] = a4
    _print_stage(4, "Document Requirement Checker", "DONE")

    # ── STAGE 5: Eligibility Reasoning ────────────────────────────────────────
    _print_stage(5, "Eligibility / Policy Reasoning")
    # Merge policy_analysis into a4 so Agent 5 can access it
    a4_enriched = dict(a4)
    a4_enriched["policy_analysis"] = a3.get("policy_analysis", {})
    a5 = agent5.run(a4_enriched)
    results["agent5"] = a5
    _print_stage(5, "Eligibility / Policy Reasoning", "DONE")

    # ── STAGE 6: Prior Auth Form Generation ───────────────────────────────────
    _print_stage(6, "Prior Authorization Form Generator")
    a6 = agent6.run(a5, output_dir=output_dir)
    results["agent6"] = a6
    _print_stage(6, "Prior Authorization Form Generator", "DONE")

    # ── STAGE 7: Portal Submission ────────────────────────────────────────────
    _print_stage(7, "Portal Automation Agent")
    a7 = agent7.run(a6)
    results["agent7"] = a7
    _print_stage(7, "Portal Automation Agent", "DONE")

    # ── STAGE 8: Claim Status Tracking ────────────────────────────────────────
    _print_stage(8, "Claim Status Tracker")
    a8 = agent8.run(a7)
    results["agent8"] = a8
    _print_stage(8, "Claim Status Tracker", "DONE")

    # ── STAGE 9: Appeal (conditional) ─────────────────────────────────────────
    _print_stage(9, "Appeal Agent")
    decision = a8.get("decision", "")
    if a5.get("determination") == "DENIED":
        a9 = agent9.run(a8, eligibility=a5, output_dir=output_dir)
    else:
        a9 = {
            "appeal_needed": False,
            "message": f"No appeal required. Determination: {a5.get('determination')}",
        }
    results["agent9"] = a9

    # ── Final Summary ──────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()

    results["pipeline_status"] = "COMPLETE"
    results["final_decision"]  = decision
    results["tracking_number"] = a7.get("tracking_number")
    results["auth_number"]     = a8.get("auth_number")
    results["elapsed_seconds"] = elapsed

    print("\n" + "#" * 70)
    print("  🏥  PRE-AUTHORIZATION PIPELINE  —  COMPLETE")
    print("#" * 70)
    print(f"\n  Patient         : {a1.get('patient_name')}")
    print(f"  Insurer         : {a1.get('insurer')}")
    print(f"  Diagnosis       : {a1.get('diagnosis')} ({a1.get('icd10')})")
    print(f"  Procedure       : {a1.get('procedure')} (CPT {a1.get('cpt')})")
    print(f"  Eligibility     : {a5.get('determination')}")
    print(f"  Tracking Number : {a7.get('tracking_number')}")
    print(f"  Final Decision  : {decision}")
    if a8.get("auth_number"):
        print(f"  Auth Number     : {a8.get('auth_number')}")
    print(f"  Form File       : {a6.get('form_txt_path')}")
    if a9.get("appeal_needed"):
        print(f"  Appeal File     : {a9.get('appeal_txt_path')}")
    print(f"\n  ⏱  Total time   : {elapsed:.1f} seconds")
    print("#" * 70 + "\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save full pipeline results to JSON
# ─────────────────────────────────────────────────────────────────────────────

def _safe_serialize(obj):
    """Make pipeline results JSON-serializable (strip _raw blobs)."""
    if isinstance(obj, dict):
        return {
            k: _safe_serialize(v)
            for k, v in obj.items()
            if k not in ("_raw",)           # skip large raw blobs
            and not k.endswith("_text")     # skip long text strings in summary
        }
    if isinstance(obj, list):
        return [_safe_serialize(i) for i in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_pipeline(DOC_PATHS, output_dir=OUTPUT_DIR)

    # Save full results to the configured output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = str(Path(OUTPUT_DIR) / f"pipeline_results_{ts}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(_safe_serialize(results), f, indent=2)
    print(f"📁 Full pipeline results saved to: {out_file}")
