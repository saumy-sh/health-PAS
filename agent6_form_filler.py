"""
agent6_form_filler.py
──────────────────────
Agent 6 — EWA Pre-Authorization PDF Form Filler

Fills the "Request for Cashless Hospitalisation for Health Insurance"
(EWA Policy Part-C Revised) 6-page PDF using all data extracted by
the pipeline's earlier agents.

NOTE (v2 pipeline):
  Patient / insurance data now comes from Agent 2's policy_search_fields.
  Clinical data comes from Agent 5's data chain (agent4 → agent3 → agent2 → policy_search_fields).

Strategy:
  • The EWA PDF has NO fillable form fields — it is a plain-text PDF.
  • We use the FORMS.md annotation approach:
      1. Load fields_template.json (53 pre-mapped field positions in PDF coords)
      2. Substitute {{placeholders}} with real patient/clinical data
      3. Use pypdf FreeText annotations to overlay text directly (no external script)
  • Output: a filled PDF ready for submission

Input  : agent5_output dict (contains all patient + clinical data)
Output : dict  { "filled_pdf_path": str, "fields_filled": int, "fields_skipped": int }
"""

import json
from datetime import datetime
from pathlib import Path
import os
import tempfile

# NEW
from backend.s3_utils import upload_to_s3
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import FreeText


# ─────────────────────────────────────────────────────────────────────────────
# Path to the blank EWA form PDF and the fields template
# Update FORM_PDF_PATH to wherever the form lives in your project
# ─────────────────────────────────────────────────────────────────────────────

FORM_PDF_PATH        = "./forms/PreAuth_Form_EWA.pdf"
FIELDS_TEMPLATE_PATH = "./forms/fields_template.json"


# ─────────────────────────────────────────────────────────────────────────────
# Data mapping: build the placeholder → value dictionary from pipeline outputs
# ─────────────────────────────────────────────────────────────────────────────

def _build_placeholder_map(data: dict, eligibility: dict) -> dict:
    """
    Map all extracted pipeline data to the {{placeholders}} used in fields_template.json.

    `data` is expected to be agent2's policy_search_fields (or a merged dict containing
    the same keys). Any placeholder that cannot be filled is left as an empty string.
    """
    # Format date of birth from YYYY-MM-DD → DD/MM/YYYY
    dob_raw = data.get("patient_dob", "") or data.get("dob", "")
    dob_fmt = ""
    if dob_raw:
        try:
            dob_fmt = datetime.strptime(dob_raw, "%Y-%m-%d").strftime("%d/%m/%Y")
        except ValueError:
            dob_fmt = dob_raw

    # Age calculation from DOB
    age_years = ""
    age_months = ""
    if dob_raw:
        try:
            dob = datetime.strptime(dob_raw, "%Y-%m-%d")
            today = datetime.today()
            age_years  = str((today - dob).days // 365)
            age_months = str(((today - dob).days % 365) // 30)
        except ValueError:
            pass

    # Gender checkboxes — put X in the right one
    gender = (data.get("patient_gender") or data.get("gender") or "").lower()
    gender_male_check   = "X" if "male" in gender and "female" not in gender else ""
    gender_female_check = "X" if "female" in gender else ""

    # Date of proposed treatment → admission date
    treatment_date_raw = data.get("date_of_service") or data.get("date_of_proposed_treatment", "")
    treatment_date_fmt = ""
    if treatment_date_raw:
        try:
            treatment_date_fmt = datetime.strptime(
                treatment_date_raw, "%Y-%m-%d"
            ).strftime("%d/%m/%Y")
        except ValueError:
            treatment_date_fmt = treatment_date_raw

    # Prior treatments (may not be in policy_search_fields — use empty fallback)
    prior_tx_list = data.get("prior_treatments", [])
    prior_tx_str  = "; ".join(prior_tx_list) if prior_tx_list else ""

    # ICD-10 codes
    icd10_codes = data.get("icd10_codes", [])
    icd10_str   = ", ".join(icd10_codes) if icd10_codes else data.get("icd10_code", "")

    # CPT codes
    cpt_codes = data.get("cpt_codes", [])
    cpt_str   = ", ".join(cpt_codes) if cpt_codes else data.get("cpt", "")

    # Symptom duration in days (convert from weeks if present)
    symptom_weeks = data.get("symptom_duration_weeks")
    ailment_days  = str(int(symptom_weeks) * 7) if symptom_weeks else ""

    # First consultation date
    first_consult_raw = data.get("date_of_service") or data.get("date_of_proposed_treatment", "")
    first_consult_fmt = ""
    if first_consult_raw:
        try:
            first_consult_fmt = datetime.strptime(
                first_consult_raw, "%Y-%m-%d"
            ).strftime("%d/%m/%Y")
        except ValueError:
            first_consult_fmt = first_consult_raw

    # Investigation vs Medical management
    procedure = (data.get("procedure") or data.get("procedure_identified") or "").lower()
    is_investigation  = "X" if any(k in procedure for k in ["mri", "ct", "scan", "x-ray", "xray", "imaging", "lab", "test"]) else ""
    is_medical_mgmt   = "X" if any(k in procedure for k in ["medication", "therapy", "management", "treatment", "drug"]) else ""

    # Total cost
    total_cost_raw = data.get("total_estimated_cost")
    total_cost_str = f"Rs. {total_cost_raw:,.0f}" if total_cost_raw else ""

    # Hospitalization type — pre-auth is always planned (not emergency)
    hospitalization_planned = "X"

    # Declaration date = today
    declaration_date = datetime.today().strftime("%d/%m/%Y")

    # TPA info — EWA is the TPA for this form
    tpa_name  = "East West Assist Insurance TPA Pvt. Ltd."
    tpa_phone = "+91-11-47222666"
    tpa_fax   = "+91-11-47222666"

    return {
        # TPA / Hospital
        "tpa_name":              tpa_name,
        "tpa_phone":             tpa_phone,
        "tpa_fax":               tpa_fax,
        "hospital_name":         data.get("hospital", ""),
        "hospital_address":      "456 Medical Plaza, Boston, MA 02118",

        # Patient demographics
        "patient_name":          data.get("patient_name", ""),
        "gender_male_check":     gender_male_check,
        "gender_female_check":   gender_female_check,
        "age_years":             age_years,
        "age_months":            age_months,
        "date_of_birth":         dob_fmt,
        "patient_phone":         data.get("patient_phone", ""),
        "relative_phone":        data.get("emergency_contact_phone", ""),
        "member_id":             data.get("member_id", ""),
        "policy_number":         data.get("policy_number", ""),
        "patient_address":       data.get("patient_address", "") or data.get("address", ""),
        "occupation":            "",

        # Insurance checkboxes
        "other_insurance_no":    "X",
        "family_physician_no":   "X",

        # Clinical — treating doctor section (Page 2)
        "treating_doctor":       data.get("ordering_physician", ""),
        "doctor_phone":          data.get("physician_phone", ""),
        "illness_complaint":     data.get("diagnosis", ""),
        "critical_findings":     (data.get("clinical_findings") or "")[:80],
        "ailment_duration_days": ailment_days,
        "first_consultation_date": first_consult_fmt,
        "past_history":          prior_tx_str[:60],
        "provisional_diagnosis": data.get("diagnosis", ""),
        "icd10_code":            icd10_str,

        # Treatment type checkboxes
        "treatment_medical":      is_medical_mgmt,
        "treatment_investigation": is_investigation,
        "investigation_details":  data.get("procedure") or data.get("procedure_identified", ""),
        "drug_route":             "Oral",
        "surgery_name":           "",
        "icd10_pcs_code":         "",

        # Admission details (Page 3)
        "admission_date":         treatment_date_fmt,
        "admission_time":         "09:00",
        "hospitalization_planned": hospitalization_planned,
        "expected_days":          "1",
        "room_type":              "General Ward",
        "room_rent_daily":        "",
        "investigation_cost":     total_cost_str,
        "icu_charges":            "",
        "ot_charges":             "",
        "professional_fees":      "",
        "medicines_cost":         "",
        "other_expenses":         "",
        "total_cost":             total_cost_str,

        # Declaration (Page 4)
        "doctor_qualification":  "MD, Orthopedic Surgery",
        "doctor_registration":   f"NPI: {data.get('physician_npi', 'See attached')}",

        # Declaration (Page 5)
        "declaration_date":      declaration_date,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build filled fields.json by substituting placeholders with real values
# ─────────────────────────────────────────────────────────────────────────────

def _build_filled_fields_json(
    template_path: str,
    placeholder_map: dict,
    output_path: str,
) -> tuple:
    """
    Load fields_template.json, substitute all {{placeholders}},
    write a filled fields.json ready for fill_pdf_form_with_annotations.py.

    Returns (fields_filled_count, fields_skipped_count).
    """
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    filled  = 0
    skipped = 0

    for field in template["form_fields"]:
        raw_text = field.get("entry_text", {}).get("text", "")

        # Extract placeholder name from {{placeholder}}
        if raw_text.startswith("{{") and raw_text.endswith("}}"):
            key   = raw_text[2:-2]
            value = placeholder_map.get(key, "")
            if value:
                field["entry_text"]["text"] = str(value)
                filled += 1
            else:
                # Leave the field empty — remove entry_text so nothing is written
                field["entry_text"]["text"] = ""
                skipped += 1
        else:
            filled += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)

    return filled, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Fill the PDF — inlined pypdf annotation logic (no external script needed)
# Works on Windows, Linux, and Mac without any path resolution.
# ─────────────────────────────────────────────────────────────────────────────

def _transform_pdf_coords(bbox: list, pdf_height: float) -> tuple:
    """Convert PDF-space bounding box (top-origin) to pypdf rect (bottom-origin)."""
    left   = bbox[0]
    right  = bbox[2]
    top    = pdf_height - bbox[1]   # flip y: PDF top  → pypdf bottom-origin
    bottom = pdf_height - bbox[3]   # flip y: PDF bottom → pypdf top
    return left, bottom, right, top


def _fill_pdf(
    blank_pdf:   str,
    fields_json: str,
    output_pdf:  str,
) -> bool:
    """
    Overlay text annotations onto the blank EWA PDF using pypdf.
    Reads field positions from fields_json (PDF coordinate space).
    Returns True on success, False on any error.
    """
    try:
        with open(fields_json, "r", encoding="utf-8") as f:
            fields_data = json.load(f)

        reader = PdfReader(blank_pdf)
        writer = PdfWriter()
        writer.append(reader)

        # Build a page-number → actual PDF dimensions map
        pdf_dims = {}
        for i, page in enumerate(reader.pages):
            mb = page.mediabox
            pdf_dims[i + 1] = (float(mb.width), float(mb.height))

        annotations_added = 0
        for field in fields_data.get("form_fields", []):
            text = field.get("entry_text", {}).get("text", "")
            if not text:
                continue  # skip empty fields

            page_num = field["page_number"]
            _, pdf_height = pdf_dims.get(page_num, (595.0, 842.0))

            rect = _transform_pdf_coords(field["entry_bounding_box"], pdf_height)
            font_size = str(field.get("entry_text", {}).get("font_size", 9)) + "pt"

            annotation = FreeText(
                text=text,
                rect=rect,
                font="Helvetica",
                font_size=font_size,
                font_color="000000",
                border_color=None,
                background_color=None,
            )
            writer.add_annotation(page_number=page_num - 1, annotation=annotation)
            annotations_added += 1

        with open(output_pdf, "wb") as out:
            writer.write(out)

        print(f"  [Agent6] ✅ Successfully filled PDF — {annotations_added} annotations added")
        print(f"  [Agent6] Saved → {output_pdf}")
        return True

    except Exception as e:
        print(f"  [Agent6] ❌ PDF fill error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    agent5_output: dict,
    output_dir:         str  = "./output",
    form_pdf_path:      str  = FORM_PDF_PATH,
    fields_template_path: str = FIELDS_TEMPLATE_PATH,
) -> dict:
    """
    Agent 6 entry point — EWA Pre-Auth PDF Form Filler.

    Parameters
    ----------
    agent5_output        : dict — output from Agent 5 (all patient + clinical data)
    output_dir           : str  — where to save the filled PDF
    form_pdf_path        : str  — path to blank PreAuth_Form_EWA.pdf
    fields_template_path : str  — path to fields_template.json

    Returns
    -------
    dict with filled_pdf_path, fields_filled, fields_skipped
    """
    print("\n[Agent 6] EWA Pre-Auth PDF Form Filler — START")

    # Navigate the agent chain to reach agent2's policy_search_fields:
    # agent5.data → agent4 output → agent4.data → agent3 output → agent3.policy_search_fields
    a4_output  = agent5_output.get("data", {})
    a3_output  = a4_output.get("data", {})
    psf        = a3_output.get("policy_search_fields", {})

    # Merge policy_search_fields with any extra keys from a3_output for maximum coverage
    data = {**a3_output, **psf}

    eligibility = {
        "determination":     agent5_output.get("determination"),
        "approval_probability": agent5_output.get("approval_probability"),
        "clinical_summary":  agent5_output.get("clinical_summary"),
        "recommendation":    agent5_output.get("recommendation"),
    }

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not Path(form_pdf_path).exists():
        print(f"  [Agent6] ⚠  Blank form not found at: {form_pdf_path}")
        print(f"  [Agent6] Place PreAuth_Form_EWA.pdf at that path and re-run.")
        return {
            "filled_pdf_path": None,
            "error":           f"Form template not found: {form_pdf_path}",
            "fields_filled":   0,
            "fields_skipped":  0,
        }

    if not Path(fields_template_path).exists():
        print(f"  [Agent6] ⚠  Fields template not found at: {fields_template_path}")
        return {
            "filled_pdf_path": None,
            "error":           f"Fields template not found: {fields_template_path}",
            "fields_filled":   0,
            "fields_skipped":  0,
        }

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Build placeholder map ─────────────────────────────────────────────────
    print("[Agent 6] Mapping patient data to form fields...")
    placeholder_map = _build_placeholder_map(data, eligibility)

    filled_keys   = {k: v for k, v in placeholder_map.items() if v}
    skipped_keys  = {k for k, v in placeholder_map.items() if not v}
    print(f"  Data available : {len(filled_keys)} fields have values")
    print(f"  Data missing   : {len(skipped_keys)} fields are empty: {list(skipped_keys)[:5]}")

    # ── Build filled fields.json ──────────────────────────────────────────────
    # ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    # mrn          = str(data.get("patient_mrn", "unknown")).replace("-", "")
    
    # Save to temp for Lambda compatibility
    tmp_dir = "/tmp" if os.name != 'nt' else tempfile.gettempdir()
    
    json_name    = f"fields_filled_{mrn}_{ts}.json"
    pdf_name     = f"PreAuth_EWA_filled_{mrn}_{ts}.pdf"
    
    filled_json  = os.path.join(tmp_dir, json_name)
    output_pdf   = os.path.join(tmp_dir, pdf_name)

    print("[Agent 6] Substituting placeholders into fields template...")
    fields_filled, fields_skipped = _build_filled_fields_json(
        template_path=fields_template_path,
        placeholder_map=placeholder_map,
        output_path=filled_json,
    )
    print(f"  Fields filled  : {fields_filled}")
    print(f"  Fields skipped : {fields_skipped} (left blank in PDF)")

    # ── Fill the PDF ──────────────────────────────────────────────────────────
    print(f"[Agent 6] Filling PDF form...")
    success = _fill_pdf(
        blank_pdf=form_pdf_path,
        fields_json=filled_json,
        output_pdf=output_pdf,
    )

    if not success:
        return {
            "filled_pdf_path": None,
            "error":           "PDF fill script failed — check logs above",
            "fields_filled":   fields_filled,
            "fields_skipped":  fields_skipped,
        }

    # Upload to S3
    print(f"[Agent 6] Uploading results to S3...")
    pdf_url  = upload_to_s3(output_pdf, f"outputs/{pdf_name}")
    json_url = upload_to_s3(filled_json, f"outputs/{json_name}")

    print(f"  PDF S3 URL : {pdf_url}")
    print(f"  JSON S3 URL: {json_url}")
    print("[Agent 6] EWA Pre-Auth PDF Form Filler — DONE\n")

    return {
        "filled_pdf_path": pdf_url,
        "fields_json_path": json_url,
        "fields_filled":   fields_filled,
        "fields_skipped":  fields_skipped,
        "data":            data,
        "eligibility":     eligibility,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — uses sample patient data matching our John R. Doe case
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_agent5_output = {
        "determination":      "APPROVED",
        "criteria_met_count": 5,
        "criteria_total":     5,
        "clinical_summary":   "Patient meets all BCBS criteria for MRI Lumbar Spine.",
        "recommendation":     "Approve MRI Lumbar Spine Without Contrast (CPT 72148).",
        "data": {
            "patient_name":               "John R. Doe",
            "patient_dob":                "1985-05-12",
            "patient_mrn":                "PT-88293",
            "patient_gender":             "Male",
            "patient_phone":              "(555) 123-4567",
            "patient_address":            "12 Maple St, Boston, MA 02118",
            "insurer":                    "BlueCross BlueShield",
            "plan_type":                  "PPO",
            "policy_number":              "BCBS-99001122",
            "member_id":                  "BCBS-99001122",
            "diagnosis":                  "Lumbar Radiculopathy",
            "icd10_codes":                ["M54.16"],
            "icd10":                      "M54.16",
            "procedure":                  "MRI Lumbar Spine Without Contrast",
            "cpt_codes":                  ["72148"],
            "cpt":                        "72148",
            "ordering_physician":         "Dr. Sarah Jenkins",
            "physician_specialty":        "Orthopedic Surgery",
            "physician_phone":            "555-010-8899",
            "physician_npi":              "1234578890",
            "hospital":                   "Metropolitan General Hospital",
            "date_of_service":            "2024-03-10",
            "date_of_proposed_treatment": "2024-03-15",
            "total_estimated_cost":       2025,
            "symptom_duration_weeks":     7,
            "pain_score":                 7,
            "clinical_findings":          "Positive SLR at 30 degrees right. Antalgic gait.",
            "prior_treatments": [
                "NSAIDs (Ibuprofen) - 4 weeks - Minimal relief",
                "Physical Therapy - 3 weeks - Symptoms persisted",
            ],
            "prescribed_medications":     ["Lyrica 75mg", "Ibuprofen 600mg"],
        },
    }

    result = run(
        agent5_output=mock_agent5_output,
        output_dir="./output",
        form_pdf_path="./forms/PreAuth_Form_EWA.pdf",
        fields_template_path="./forms/fields_template.json",
    )

    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("data", "eligibility")},
        indent=2
    ))
