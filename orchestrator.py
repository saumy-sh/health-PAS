"""
orchestrator.py  (UPDATED — v2.1)
──────────────────────────────────
Pre-Authorization Pipeline Orchestrator
Runs all agents in sequence, passing outputs between them.

Changes from v2.0:
  • Agent 1  — two-pass extraction, JSON repair, completeness scoring
  • Agent 3  — deterministic checklist + narrowed LLM (no hallucinated requirements)
  • Agent 4  — uses canonical_requirements from checklist, not LLM-generated list
  • Agent 5  — programmatic clinical checks (step therapy, symptom duration, etc.)
  • Agent 6  — generates structured JSON + text form (unchanged)
  • Agent 6b — REPLACED: now fills the EWA Pre-Auth PDF form (6 pages) using
               agent6_form_filler.py + fields_template.json with exact PDF
               coordinate mapping. Outputs a submission-ready filled PDF.
               Non-blocking: pipeline continues if the blank form PDF is missing.
  • Halt messages now show precise user instructions (priority-sorted)
  • x_ray / imaging documents supported in DOC_PATHS

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
import agent6_form_generator         as agent6      # JSON/text form (internal use)
import agent6_form_filler            as agent6b     # EWA PDF form filler





# ─────────────────────────────────────────────────────────────────────────────
# Document paths — update these to match your file locations
# Add or remove entries freely; Agent 1 auto-classifies every file.
# ─────────────────────────────────────────────────────────────────────────────

DOC_PATHS = {
    "lab_report":            "./data/patient_data/lab_report_ai.png",
    "doctor_notes":          "./data/patient_data/doctor_notes_ai.png",
    "patient_info":          "./data/patient_data/patient_ai.png",
    "insurance_card":        "./data/patient_data/insurance_card_ai.png",
    "pretreatment_estimate": "./data/patient_data/medical_pretreatment_estimate_ai.pdf",
    "prescription":          "./data/patient_data/prescription_ai.png",
    "referral_letter":       "./data/patient_data/physician_referral.png",
    "x_ray":                 "./data/patient_data/x_ray.jpg",
}

# ── EWA Pre-Auth PDF form paths ────────────────────────────────────────────────
# Blank 6-page EWA form that Agent 6b will fill
EWA_FORM_PDF      = "./data/forms/Pre-Auth Form EWA.pdf"
# Pre-mapped field coordinates (PDF coordinate space, 53 fields across 6 pages)
EWA_FIELDS_JSON   = "./data/forms/fields_template.json"

OUTPUT_DIR = "./output"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_stage(stage, name: str, status: str = "RUNNING"):
    icons  = {"RUNNING": "🔄", "DONE": "✅", "SKIP": "⏭", "WARN": "⚠️", "FAIL": "❌"}
    icon   = icons.get(status, "•")
    label  = f"STAGE {stage}" if isinstance(stage, int) else f"STAGE {stage}"
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
        return {
            k: _safe_serialize(v)
            for k, v in obj.items()
            if k not in ("_raw",)
        }
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
    Run the complete pre-authorization pipeline.

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
    print("  🏥  PRE-AUTHORIZATION PIPELINE  —  START  (v2.1)")
    print(f"  Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Docs    : {len(doc_paths)} files")
    print(f"  Output  : {output_dir}")
    print(f"  EWA PDF : {ewa_form_pdf}")
    print("#" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Document Intelligence
    #   Two-pass extraction: core fields (4096 tokens) + per-doc detail (1500)
    #   Produces: documents[], document_content_map, field_source_map,
    #             completeness_summary, null_fields_by_doc
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(1, "Document Intelligence")
    try:
        a1 = agent1.run(doc_paths)
    except Exception as e:
        _print_halt("Document extraction failed", [f"Error: {e}"])
        results["pipeline_status"] = "HALTED_EXTRACTION_FAILED"
        results["error"]           = str(e)
        results["halted_at"]       = "agent1"
        return results

    results["agent1"] = a1
    _print_stage(1, "Document Intelligence", "DONE")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Policy Query Sanity Check
    #   Validates extracted fields for completeness before policy lookup.
    #   Halts if critical fields (patient, insurer, CPT, ICD-10) are absent.
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(2, "Policy Query Requirement Checker")
    a2 = agent2.run(a1)
    results["agent2"] = a2
    _print_stage(2, "Policy Query Requirement Checker", "DONE")

    if not a2.get("ready"):
        missing = a2.get("missing_required", [])
        _print_halt(
            "Stage 2 — Critical fields missing from extracted documents",
            [f"  • {m['field']} → expected in: {m.get('suggested_document', 'unknown')}"
             for m in missing],
        )
        results["pipeline_status"] = "HALTED_MISSING_FIELDS"
        results["missing_fields"]  = missing
        results["halted_at"]       = "agent2"
        return results

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Policy Retrieval (Deterministic Checklist + LLM Criteria)
    #
    #   RELIABILITY FIX: Agent 3 no longer lets the LLM freely invent
    #   required document lists. Instead:
    #     ① A hardcoded PREAUTH_CHECKLIST evaluates the 7 pre-auth steps
    #        deterministically against confirmed-present document types.
    #        Same input → same result, every run.
    #     ② The LLM is called ONLY to assess clinical criteria (symptom
    #        duration, step therapy, etc.) — not to generate doc requirements.
    #     ③ Output includes `canonical_requirements` (the deterministic list)
    #        which Agent 4 uses as its sole source of truth.
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(3, "Policy Retrieval (Deterministic Checklist + Clinical Criteria)")
    a3 = agent3.run(a2)
    results["agent3"] = a3
    _print_stage(3, "Policy Retrieval", "DONE")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Document Requirement Checker
    #
    #   RELIABILITY FIX: Agent 4 now reads from `canonical_requirements`
    #   (deterministic checklist result from Agent 3), NOT from the LLM's
    #   `required_documents` list.
    #
    #   It re-verifies each requirement against FULL document content values
    #   (not just field names) to catch partial documents.
    #   The halt decision is 100% programmatic — no LLM involved.
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(4, "Document Requirement Checker")

    # Pass a3 directly — it already contains everything Agent 4 needs:
    # canonical_requirements, checklist_result, policy_analysis, and data
    a4 = agent4.run(a3)

    if not a4.get("can_proceed", True):
        # Build a clear, prioritised action list for the user
        high_actions = [
            u for u in a4.get("user_action_required", [])
            if u.get("priority") == "HIGH"
        ]
        detail_lines = []
        for action in high_actions:
            detail_lines.append(f"\n  🔴 {action['item']}")
            detail_lines.append(f"     Reason : {action['reason']}")
            if action.get("specific_fields_needed"):
                detail_lines.append(f"     Fields : {action['specific_fields_needed']}")
            detail_lines.append(f"     Fix    : {action['how_to_resolve']}")

        _print_halt("Stage 4 — Missing or incomplete required documents", detail_lines)
        results["pipeline_status"] = "HALTED_MISSING_DOCUMENTS"
        results["missing_docs"]    = a4.get("missing_docs", [])
        results["user_actions"]    = a4.get("user_action_required", [])
        results["halted_at"]       = "agent4"
        return results

    results["agent4"] = a4
    _print_stage(4, "Document Requirement Checker", "DONE")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 5 — Clinical Requirements + Eligibility Reasoning
    #
    #   Runs programmatic clinical checks FIRST:
    #     • Minimum symptom duration (procedure-specific thresholds)
    #     • Step therapy / medication trial compliance
    #     • Specialist referral presence
    #     • Clinical examination documentation
    #   Then calls LLM for full eligibility determination.
    #   If programmatic checks find critical failures, the LLM determination
    #   is overridden to PENDING_REVIEW (LLM cannot silently approve).
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(5, "Clinical Requirements + Eligibility Reasoning")

    # Merge policy_analysis from Agent 3 into Agent 4 output for Agent 5
    a4_enriched = dict(a4)
    a4_enriched["policy_analysis"] = a3.get("policy_analysis", {})

    a5 = agent5.run(a4_enriched)
    results["agent5"] = a5
    _print_stage(5, "Clinical Requirements + Eligibility Reasoning", "DONE")

    # Warn if clinical requirements are not fully met but continue
    if a5.get("requirements_not_met"):
        print("\n  ⚠️  Some clinical requirements are not fully met:")
        for req in a5.get("requirements_not_met", []):
            print(f"    ✗ {req.get('requirement')}")
            if req.get("action_needed"):
                print(f"      → {req.get('action_needed')}")
        print()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 6 — Prior Authorization Form Generator
    #   Generates structured JSON + formatted text form from extracted data.
    #   NOTE: If agent6_form_generator attempts to save a PDF via reportlab
    #   and fails (known BytesIO issue on Windows), we catch the error and
    #   continue — the EWA PDF in Stage 6b is the primary submission document.
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage(6, "Prior Authorization Form Generator")
    try:
        a6 = agent6.run(a5, output_dir=output_dir)
    except TypeError as e:
        # Catch the reportlab BytesIO bug: "expected str, bytes or os.PathLike, not BytesIO"
        # This happens in _save_as_pdf → c.drawImage(buf, ...) on Windows.
        # Fix in agent6_form_generator.py: replace  c.drawImage(buf, ...)
        #                                  with     c.drawImage(ImageReader(buf), ...)
        # where ImageReader is from: from reportlab.lib.utils import ImageReader
        print(f"  ⚠️  agent6 PDF save failed (reportlab BytesIO bug): {e}")
        print("  Continuing — JSON/text form may still be available; EWA PDF (Stage 6b) is unaffected.")
        # Try to get whatever was saved before the crash
        a6 = getattr(agent6, "_last_output", {
            "form_json":      {},
            "form_text":      "",
            "form_json_path": None,
            "form_txt_path":  None,
            "data":           a5.get("data", {}),
            "eligibility":    {},
        })
    results["agent6"] = a6
    _print_stage(6, "Prior Authorization Form Generator", "DONE")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 6b — EWA Pre-Auth PDF Form Filler
    #
    #   Fills the 6-page "Request for Cashless Hospitalisation" EWA form
    #   (PreAuth_Form_EWA.pdf) using:
    #     ① fields_template.json — 53 pre-mapped field positions in PDF
    #        coordinate space (exact, validated with check_bounding_boxes.py)
    #     ② _build_placeholder_map() — maps all pipeline data to {{placeholders}}
    #     ③ fill_pdf_form_with_annotations.py — overlays text annotations
    #
    #   Fills across all 6 pages:
    #     Page 1 — TPA/hospital details + patient demographics
    #     Page 2 — Treating doctor + clinical findings + ICD-10 + treatment type
    #     Page 3 — Admission details + cost breakdown
    #     Page 4 — Doctor declaration + registration
    #     Page 5 — Patient declaration + date
    #     Page 6 — Hospital declaration (no fillable fields)
    #
    #   Non-blocking: if the blank EWA PDF is missing, pipeline continues
    #   with the JSON/text form from Stage 6 only.
    # ══════════════════════════════════════════════════════════════════════════
    _print_stage("6b", "EWA Pre-Auth PDF Form Filler")
    a6b = {}
    ewa_pdf_path = Path(ewa_form_pdf)

    if not ewa_pdf_path.exists():
        print(f"  ⚠️  EWA form PDF not found at: {ewa_form_pdf}")
        print(f"  Place PreAuth_Form_EWA.pdf at that path to enable PDF filling.")
        print(f"  Continuing with JSON/text form from Stage 6 only.")
        results["agent6b"] = {"skipped": True, "reason": f"EWA form not found: {ewa_form_pdf}"}
        _print_stage("6b", "EWA Pre-Auth PDF Form Filler", "SKIP")

    elif not Path(ewa_fields_json).exists():
        print(f"  ⚠️  Fields template not found at: {ewa_fields_json}")
        print(f"  Place fields_template.json at that path to enable PDF filling.")
        print(f"  Continuing with JSON/text form from Stage 6 only.")
        results["agent6b"] = {"skipped": True, "reason": f"Fields template not found: {ewa_fields_json}"}
        _print_stage("6b", "EWA Pre-Auth PDF Form Filler", "SKIP")

    else:
        try:
            a6b = agent6b.run(
                agent5_output=a5,
                output_dir=output_dir,
                form_pdf_path=str(ewa_pdf_path),
                fields_template_path=ewa_fields_json,
            )
            results["agent6b"] = a6b
            if a6b.get("filled_pdf_path"):
                _print_stage("6b", "EWA Pre-Auth PDF Form Filler", "DONE")
            else:
                print(f"  ⚠️  PDF filling completed but no output path returned.")
                _print_stage("6b", "EWA Pre-Auth PDF Form Filler", "WARN")
        except Exception as e:
            print(f"  ⚠️  EWA PDF form filling failed: {e}")
            print("  Continuing with JSON/text form from Stage 6 only.")
            results["agent6b"] = {"error": str(e), "filled_pdf_path": None}
            _print_stage("6b", "EWA Pre-Auth PDF Form Filler", "WARN")

    
    # ══════════════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = (datetime.now() - start_time).total_seconds()

    results["pipeline_status"] = "COMPLETE"
    
    results["elapsed_seconds"] = elapsed

    filled_pdf    = a6b.get("filled_pdf_path") if a6b else None
    fields_filled = a6b.get("fields_filled", 0) if a6b else 0

    print("\n" + "#" * 70)
    print("  🏥  PRE-AUTHORIZATION PIPELINE  —  COMPLETE  (v2.1)")
    print("#" * 70)
    print(f"\n  Patient             : {a1.get('patient_name')}")
    print(f"  DOB                 : {a1.get('patient_dob')}")
    print(f"  Insurer             : {a1.get('insurer')}  |  Member: {a1.get('member_id')}")
    print(f"  Diagnosis           : {a1.get('diagnosis')} ({a1.get('icd10')})")
    print(f"  Procedure           : {a1.get('procedure')} (CPT {a1.get('cpt')})")
    print(f"  Eligibility         : {a5.get('determination')}  "
          f"(confidence: {a5.get('confidence')})")
    print(f"  Clinical compliance : {a5.get('overall_clinical_compliance', 'N/A')}")
    print(f"  Checklist           : "
          f"{a3.get('checklist_result', {}).get('satisfied_count', '?')}/"
          f"{a3.get('checklist_result', {}).get('total_required', '?')} steps satisfied")
    print(f"     JSON form  : {a6.get('form_json_path', 'N/A')}")
    print(f"     Text form  : {a6.get('form_txt_path', 'N/A')}")
    if filled_pdf:
        print(f"     EWA PDF    : {filled_pdf}  ({fields_filled} fields filled)")
    else:
        print(f"     EWA PDF    : not generated (check Stage 6b warnings above)")
    
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
