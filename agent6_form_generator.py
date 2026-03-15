"""
agent6_form_generator.py
─────────────────────────
Agent 6 — Prior Authorization Form Filler
• Takes eligibility determination + patient data from Agent 5
• Uses Nova Lite to map all extracted data to the Standardized Prior
  Authorization Request Form fields (pre_auth_form.jpg)
• Overlays filled text onto the actual form image using Pillow
• Saves a filled PNG + PDF ready for submission
• NO fake/placeholder form — we fill the REAL standardized form

Pre-auth form layout (pre_auth_form.jpg):
  • Header  : Health Plan, Health Plan Fax #, Date Form Completed and Faxed
  • Section : Service Type checkboxes (Ambulatory/Outpatient → Surgery/Procedure SDO)
  • Section : Provider Information
  • Section : Member Information
  • Section : Diagnosis/Planned Procedure Information

Input  : dict from Agent 5
Output : dict  { "form_png_path": str, "form_pdf_path": str, "form_json": dict }
"""

import json
import io
import textwrap
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas

from bedrock_client import invoke, LITE_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Constants — pixel coordinates for pre_auth_form.jpg
# These were derived by visual analysis of the form image.
# The form image is approximately 1700 x 2200 px at standard resolution.
#
# Coordinate format: (x, y) = top-left of text insertion point
# ─────────────────────────────────────────────────────────────────────────────

# Approximate form image dimensions (will be confirmed at runtime)
FORM_IMG = "./data/forms/pre_auth_form.jpg"

# Field coordinate map: field_key -> (x, y, max_chars)
# All coordinates are in pixels assuming ~1700px wide form image.
# Scaled dynamically at runtime if image dimensions differ.
FIELD_COORDS = {
    # ── Header row ──────────────────────────────────────────────────────────
    "health_plan":          (120,  75, 28),   # "Health Plan:"
    "health_plan_fax":      (580,  75, 20),   # "Health Plan Fax #:"
    "date_completed":       (1100, 75, 16),   # "Date Form Completed and Faxed:"

    # ── Service Type — check Surgery/Procedure (SDO) box ────────────────────
    # Checkbox is at approximately (92, 165) — we mark it with an X
    "service_type_sdo_check": (92, 165, 1),   # "Surgery/Procedure (SDO)" checkbox

    # ── Provider Information ─────────────────────────────────────────────────
    "requesting_provider_name_npi": (150, 490, 40),
    "requesting_provider_phone":    (910, 490, 20),
    "requesting_provider_fax":      (1320, 490, 16),
    "servicing_provider_name_npi":  (150, 540, 40),
    "servicing_provider_phone":     (910, 540, 20),
    "servicing_facility_name_npi":  (150, 590, 40),
    "servicing_facility_phone":     (910, 590, 20),
    "contact_person":               (150, 640, 40),
    "contact_phone":                (910, 640, 20),

    # ── Member Information ───────────────────────────────────────────────────
    "patient_name":         (150,  760, 32),
    "patient_gender_m":     (770,  760,  1),   # Male checkbox
    "patient_dob":          (1000, 760, 14),
    "health_insurance_id":  (150,  810, 28),
    "patient_account":      (770,  810, 24),
    "patient_address":      (150,  860, 50),
    "patient_phone":        (1000, 860, 16),

    # ── Diagnosis / Planned Procedure ────────────────────────────────────────
    "principal_diagnosis":      (150,  970, 40),
    "planned_procedure_desc":   (810,  970, 40),
    "icd9_codes":               (150, 1020, 24),
    "units_requested":          (810, 1020, 8),
    "secondary_diagnosis":      (150, 1075, 40),
    "secondary_procedure":      (810, 1075, 40),
    "icd9_secondary":           (150, 1125, 24),
    "service_start_date":       (150, 1175, 14),
    "service_end_date":         (810, 1175, 14),
}

# Font sizes
FONT_SIZE_NORMAL  = 14
FONT_SIZE_SMALL   = 11
FONT_SIZE_CHECKBOX = 16   # Bold X for checkboxes


# ─────────────────────────────────────────────────────────────────────────────
# LLM — map extracted data to form fields
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are a medical prior-authorization form specialist. "
            "Map all available patient, insurance, and clinical data to the exact "
            "fields of the Standardized Prior Authorization Request Form. "
            "Be concise — field values must fit in form boxes. "
            "Use standard medical abbreviations where needed. "
            "Return ONLY a valid JSON object — no markdown, no preamble."
        )
    }
]


def _build_mapping_prompt(data: dict, eligibility: dict) -> str:
    return f"""Map the following patient and clinical data to the Standardized Prior Authorization
Request Form fields. Keep values SHORT — they must fit in small form boxes.

PATIENT DATA:
{json.dumps(data, indent=2, default=str)}

ELIGIBILITY:
- Determination : {eligibility.get('determination')}
- Urgency       : {eligibility.get('urgency', 'routine')}
- Recommendation: {eligibility.get('recommendation')}

Return a JSON object with EXACTLY these keys (use null for unavailable fields):
{{
  "health_plan": string,               // Insurer name, max 28 chars
  "health_plan_fax": string,           // Insurer fax if known, else null
  "date_completed": string,            // Today's date MM/DD/YYYY
  "requesting_provider_name_npi": string,  // "Dr. Name, NPI: XXXXXXXXXX" or "Dr. Name"
  "requesting_provider_phone": string,
  "requesting_provider_fax": string,
  "servicing_provider_name_npi": string,   // Same as requesting if same provider
  "servicing_provider_phone": string,
  "servicing_facility_name_npi": string,   // Hospital/facility name
  "servicing_facility_phone": string,
  "contact_person": string,
  "contact_phone": string,
  "patient_name": string,
  "patient_dob": string,               // MM/DD/YYYY
  "health_insurance_id": string,       // Member ID
  "patient_account": string,           // MRN or policy number
  "patient_address": string,           // Street, City, State ZIP
  "patient_phone": string,
  "principal_diagnosis": string,       // Short diagnosis description
  "planned_procedure_desc": string,    // "Procedure name (CPT XXXXX)"
  "icd10_codes": string,               // Primary ICD-10 code(s), comma separated
  "units_requested": string,           // "1" for single study
  "secondary_diagnosis": string,       // null if none
  "secondary_procedure": string,       // null if none
  "service_start_date": string,        // MM/DD/YYYY
  "service_end_date": string,          // MM/DD/YYYY or same as start for single
  "service_type": string,              // "Surgery/Procedure (SDO)" or other matching type
  "mark_surgery_sdo": true | false,    // true if procedure is outpatient surgery/imaging
  "mark_outpatient_therapy": true | false  // true if PT/OT/Speech
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Image form filler
# ─────────────────────────────────────────────────────────────────────────────

def _get_font(size: int, bold: bool = False):
    """Try to load a TTF font; fall back to PIL default."""
    try:
        font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(font_name, size)
    except (IOError, OSError):
        pass
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (IOError, OSError):
        pass
    # Ultimate fallback — PIL bitmap font (no size control)
    return ImageFont.load_default()


def _scale_coords(coords: dict, img_w: int, img_h: int,
                  ref_w: int = 1700, ref_h: int = 2200) -> dict:
    """Scale pixel coordinates from reference dimensions to actual image size."""
    sx = img_w / ref_w
    sy = img_h / ref_h
    scaled = {}
    for key, (x, y, max_chars) in coords.items():
        scaled[key] = (int(x * sx), int(y * sy), max_chars)
    return scaled


def _fill_form_image(form_img_path: str, field_values: dict) -> Image.Image:
    """
    Open the pre-auth form image, overlay all filled values, return filled image.
    """
    img = Image.open(form_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    print(f"  [Agent6] Form image size: {img_w} x {img_h} px")

    coords = _scale_coords(FIELD_COORDS, img_w, img_h)

    font_normal  = _get_font(FONT_SIZE_NORMAL)
    font_small   = _get_font(FONT_SIZE_SMALL)
    font_check   = _get_font(FONT_SIZE_CHECKBOX, bold=True)

    TEXT_COLOR    = (10, 10, 10)      # near-black ink
    CHECKBOX_COLOR = (30, 30, 200)    # blue X for checkboxes

    def draw_field(key: str, value: str, font=None, color=TEXT_COLOR):
        if not value or key not in coords:
            return
        x, y, max_chars = coords[key]
        # Truncate to max_chars to avoid overflow
        text = str(value)[:max_chars] if max_chars else str(value)
        f = font or font_normal
        draw.text((x, y), text, fill=color, font=f)

    # ── Header ───────────────────────────────────────────────────────────────
    draw_field("health_plan",    field_values.get("health_plan", ""))
    draw_field("health_plan_fax", field_values.get("health_plan_fax", ""))
    draw_field("date_completed",  field_values.get("date_completed", ""))

    # ── Service type checkbox ─────────────────────────────────────────────────
    if field_values.get("mark_surgery_sdo"):
        draw_field("service_type_sdo_check", "X", font=font_check, color=CHECKBOX_COLOR)
    if field_values.get("mark_outpatient_therapy"):
        # Outpatient Therapy — Physical Therapy checkbox approx (1390, 165)
        img_w2, img_h2 = img.size
        sx = img_w2 / 1700
        sy = img_h2 / 2200
        draw.text((int(1390 * sx), int(165 * sy)), "X",
                  fill=CHECKBOX_COLOR, font=font_check)

    # ── Provider Information ──────────────────────────────────────────────────
    draw_field("requesting_provider_name_npi", field_values.get("requesting_provider_name_npi", ""),
               font=font_small)
    draw_field("requesting_provider_phone",    field_values.get("requesting_provider_phone", ""),
               font=font_small)
    draw_field("requesting_provider_fax",      field_values.get("requesting_provider_fax", ""),
               font=font_small)
    draw_field("servicing_provider_name_npi",  field_values.get("servicing_provider_name_npi", ""),
               font=font_small)
    draw_field("servicing_provider_phone",     field_values.get("servicing_provider_phone", ""),
               font=font_small)
    draw_field("servicing_facility_name_npi",  field_values.get("servicing_facility_name_npi", ""),
               font=font_small)
    draw_field("servicing_facility_phone",     field_values.get("servicing_facility_phone", ""),
               font=font_small)
    draw_field("contact_person",               field_values.get("contact_person", ""),
               font=font_small)
    draw_field("contact_phone",                field_values.get("contact_phone", ""),
               font=font_small)

    # ── Member Information ────────────────────────────────────────────────────
    draw_field("patient_name",       field_values.get("patient_name", ""))
    draw_field("patient_dob",        field_values.get("patient_dob", ""), font=font_small)
    draw_field("health_insurance_id", field_values.get("health_insurance_id", ""))
    draw_field("patient_account",    field_values.get("patient_account", ""), font=font_small)
    draw_field("patient_address",    field_values.get("patient_address", ""), font=font_small)
    draw_field("patient_phone",      field_values.get("patient_phone", ""), font=font_small)

    # Male gender checkbox — mark if male
    gender = str(field_values.get("patient_gender", "")).lower()
    if gender in ("male", "m"):
        draw_field("patient_gender_m", "X", font=font_check, color=CHECKBOX_COLOR)

    # ── Diagnosis / Planned Procedure ─────────────────────────────────────────
    draw_field("principal_diagnosis",   field_values.get("principal_diagnosis", ""),
               font=font_small)
    draw_field("planned_procedure_desc", field_values.get("planned_procedure_desc", ""),
               font=font_small)
    draw_field("icd9_codes",            field_values.get("icd10_codes", ""),
               font=font_small)
    draw_field("units_requested",       field_values.get("units_requested", "1"),
               font=font_small)
    draw_field("secondary_diagnosis",   field_values.get("secondary_diagnosis", ""),
               font=font_small)
    draw_field("secondary_procedure",   field_values.get("secondary_procedure", ""),
               font=font_small)
    draw_field("service_start_date",    field_values.get("service_start_date", ""),
               font=font_small)
    draw_field("service_end_date",      field_values.get("service_end_date", ""),
               font=font_small)

    return img


def _save_as_pdf(img: Image.Image, pdf_path: str) -> str:
    """Save PIL image as a single-page PDF using ReportLab."""
    # Save image to a temp PNG buffer
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Load into ReportLab and embed as full-page image
    img_w, img_h = img.size
    page_w, page_h = letter  # 612 x 792 pts

    c = rl_canvas.Canvas(pdf_path, pagesize=(page_w, page_h))
    # Scale image to fit the page while preserving aspect ratio
    scale = min(page_w / img_w, page_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    x_offset = (page_w - draw_w) / 2
    y_offset = (page_h - draw_h) / 2

    # Draw the filled form image
    c.drawImage(
        buf, x_offset, y_offset, width=draw_w, height=draw_h,
        preserveAspectRatio=True
    )
    c.showPage()
    c.save()
    return pdf_path


# ─────────────────────────────────────────────────────────────────────────────
# Also generate a structured JSON + plain-text summary (kept for records)
# ─────────────────────────────────────────────────────────────────────────────

def _render_form_text(form: dict, field_values: dict) -> str:
    """Render a human-readable summary of what was filled into the form."""
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("    STANDARDIZED PRIOR AUTHORIZATION REQUEST — FILLED VALUES")
    lines.append(sep)
    lines.append(f"  Form completed  : {field_values.get('date_completed')}")
    lines.append(f"  Health Plan     : {field_values.get('health_plan')}")
    lines.append(f"  Service Type    : {form.get('service_type', 'Surgery/Procedure (SDO)')}")
    lines.append("")
    lines.append("  PROVIDER")
    lines.append(f"    Requesting    : {field_values.get('requesting_provider_name_npi')}")
    lines.append(f"    Phone         : {field_values.get('requesting_provider_phone')}")
    lines.append(f"    Facility      : {field_values.get('servicing_facility_name_npi')}")
    lines.append("")
    lines.append("  MEMBER")
    lines.append(f"    Name          : {field_values.get('patient_name')}")
    lines.append(f"    DOB           : {field_values.get('patient_dob')}")
    lines.append(f"    Insurance ID  : {field_values.get('health_insurance_id')}")
    lines.append(f"    Account/MRN   : {field_values.get('patient_account')}")
    lines.append(f"    Address       : {field_values.get('patient_address')}")
    lines.append(f"    Phone         : {field_values.get('patient_phone')}")
    lines.append("")
    lines.append("  DIAGNOSIS / PROCEDURE")
    lines.append(f"    Diagnosis     : {field_values.get('principal_diagnosis')}")
    lines.append(f"    ICD-10        : {field_values.get('icd10_codes')}")
    lines.append(f"    Procedure     : {field_values.get('planned_procedure_desc')}")
    lines.append(f"    Units         : {field_values.get('units_requested')}")
    lines.append(f"    Service Start : {field_values.get('service_start_date')}")
    lines.append(f"    Service End   : {field_values.get('service_end_date')}")
    lines.append(sep)
    lines.append("  * Physician signature required before submission *")
    lines.append(sep)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(agent5_output: dict,
        form_template_path: str = "./data/forms/pre_auth_form.jpg",
        output_dir: str = ".") -> dict:
    """
    Agent 6 entry point.

    Parameters
    ----------
    agent5_output       : dict — output from Agent 5
    form_template_path  : str  — path to the blank pre_auth_form.jpg
    output_dir          : str  — directory to save output files

    Returns
    -------
    dict with form_png_path, form_pdf_path, form_json, form_text
    """
    print("\n[Agent 6] Prior Authorization Form Filler — START")
    print(f"  Template form : {form_template_path}")

    data       = agent5_output.get("data", {})
    eligibility = {
        "determination":          agent5_output.get("determination"),
        "urgency":                agent5_output.get("urgency", data.get("urgency", "routine")),
        "expedited_review_requested": agent5_output.get("expedited_review_requested", False),
        "criteria_met_count":     agent5_output.get("criteria_met_count"),
        "criteria_total":         agent5_output.get("criteria_total"),
        "clinical_summary":       agent5_output.get("clinical_summary"),
        "recommendation":         agent5_output.get("recommendation"),
    }

    # Step 1: LLM maps patient data → form field values
    prompt   = _build_mapping_prompt(data, eligibility)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    print("[Agent 6] Calling Nova Lite to map data to form fields...")
    raw = invoke(
        model_id=LITE_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=1000,
        temperature=0.1,
    )

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        field_values = json.loads(text.strip())
    except json.JSONDecodeError:
        print("[Agent 6] Warning: JSON parse error — using direct field mapping fallback")
        today = datetime.now().strftime("%m/%d/%Y")
        dos   = data.get("date_of_proposed_treatment", "")
        if dos:
            try:
                dos = datetime.strptime(dos, "%Y-%m-%d").strftime("%m/%d/%Y")
            except ValueError:
                pass
        field_values = {
            "health_plan":                  data.get("insurer", ""),
            "health_plan_fax":              "",
            "date_completed":               today,
            "requesting_provider_name_npi": data.get("ordering_physician", ""),
            "requesting_provider_phone":    data.get("physician_phone", ""),
            "requesting_provider_fax":      "",
            "servicing_provider_name_npi":  data.get("ordering_physician", ""),
            "servicing_provider_phone":     data.get("physician_phone", ""),
            "servicing_facility_name_npi":  data.get("hospital", ""),
            "servicing_facility_phone":     "",
            "contact_person":               data.get("ordering_physician", ""),
            "contact_phone":                data.get("physician_phone", ""),
            "patient_name":                 data.get("patient_name", ""),
            "patient_gender":               data.get("patient_gender", "Male"),
            "patient_dob":                  data.get("patient_dob", ""),
            "health_insurance_id":          data.get("member_id", ""),
            "patient_account":              data.get("patient_mrn", ""),
            "patient_address":              data.get("patient_address", ""),
            "patient_phone":                data.get("patient_phone", ""),
            "principal_diagnosis":          data.get("diagnosis", ""),
            "planned_procedure_desc":       f"{data.get('procedure', '')} (CPT {data.get('cpt', '')})",
            "icd10_codes":                  ", ".join(data.get("icd10_codes", [])),
            "units_requested":              "1",
            "secondary_diagnosis":          "",
            "secondary_procedure":          "",
            "service_start_date":           dos,
            "service_end_date":             dos,
            "mark_surgery_sdo":             True,
            "mark_outpatient_therapy":      False,
        }

    print(f"  Field values mapped: {list(field_values.keys())}")

    # Step 2: Fill the actual form image
    print("[Agent 6] Filling form image...")
    filled_img = _fill_form_image(form_template_path, field_values)

    # Step 3: Save outputs
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    mrn = data.get("patient_mrn", "unknown").replace("-", "")

    png_path  = str(Path(output_dir) / f"prior_auth_filled_{mrn}_{ts}.png")
    pdf_path  = str(Path(output_dir) / f"prior_auth_filled_{mrn}_{ts}.pdf")
    json_path = str(Path(output_dir) / f"prior_auth_filled_{mrn}_{ts}.json")
    txt_path  = str(Path(output_dir) / f"prior_auth_filled_{mrn}_{ts}.txt")

    filled_img.save(png_path)
    print(f"  Filled PNG saved  → {png_path}")

    _save_as_pdf(filled_img, pdf_path)
    print(f"  Filled PDF saved  → {pdf_path}")

    # Also save structured JSON and text summary for records
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(field_values, f, indent=2)

    form_text = _render_form_text(field_values, field_values)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(form_text)

    print(f"  JSON record saved → {json_path}")
    print(f"  Text summary saved→ {txt_path}")
    print("[Agent 6] Prior Authorization Form Filler — DONE\n")

    print(form_text)

    return {
        "form_png_path":  png_path,
        "form_pdf_path":  pdf_path,
        "form_json_path": json_path,
        "form_txt_path":  txt_path,
        "form_json":      field_values,
        "form_text":      form_text,
        "data":           data,
        "eligibility":    eligibility,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = {
        "eligible": True,
        "determination": "APPROVED",
        "confidence": "high",
        "criteria_met_count": 4,
        "criteria_total": 4,
        "urgency": "routine",
        "expedited_review_requested": False,
        "clinical_summary": "Patient meets all BCBS criteria for MRI Lumbar Spine authorization.",
        "recommendation": "Approve MRI Lumbar Spine Without Contrast (CPT 72148).",
        "data": {
            "patient_name":              "John R. Doe",
            "patient_dob":               "1985-05-12",
            "patient_mrn":               "PT-88293",
            "patient_gender":            "Male",
            "patient_address":           "12 Maple St, Boston, MA 02118",
            "patient_phone":             "(555) 123-4567",
            "patient_email":             "j.doe85@example.com",
            "insurer":                   "BlueCross BlueShield",
            "plan_type":                 "PPO",
            "policy_number":             "BCBS-99001122",
            "group_number":              "8800221",
            "member_id":                 "BCBS-9900122",
            "diagnosis":                 "Lumbar Radiculopathy",
            "icd10":                     "M54.16",
            "icd10_codes":               ["M54.16"],
            "procedure":                 "MRI Lumbar Spine Without Contrast",
            "cpt":                       "72148",
            "cpt_codes":                 ["72148", "72148-26", "99204", "97161"],
            "ordering_physician":        "Dr. Sarah Jenkins",
            "physician_specialty":       "Orthopedic Surgery",
            "physician_phone":           "555-010-8899",
            "hospital":                  "Metropolitan General Hospital",
            "hospital_address":          "456 Medical Plaza, Boston, MA 02118",
            "date_of_service":           "2024-03-10",
            "date_of_proposed_treatment": "2024-03-15",
            "total_estimated_cost":      2025,
            "prior_treatments":          ["NSAIDs 4 weeks minimal relief", "Physical Therapy 3 weeks"],
            "symptom_duration_weeks":    7,
            "pain_score":                7,
            "clinical_findings":         "Positive SLR at 30 degrees. Antalgic gait. Severe radicular pain.",
            "prescribed_medications":    ["Lyrica 75mg bid", "Ibuprofen"],
            "cost_line_items": [
                {"description": "MRI Lumbar Spine Without Contrast", "cpt_code": "72148",    "estimated_price": 1350},
                {"description": "Radiologist Interpretation",        "cpt_code": "72148-26", "estimated_price": 250},
                {"description": "Orthopedic Consultation",           "cpt_code": "99204",    "estimated_price": 220},
                {"description": "Physical Therapy Evaluation",       "cpt_code": "97161",    "estimated_price": 160},
            ],
        },
    }
    result = run(mock_input, form_template_path="./data/forms/pre_auth_form.jpg")
    print(f"\nFilled form saved to:\n  PNG: {result['form_png_path']}\n  PDF: {result['form_pdf_path']}")
