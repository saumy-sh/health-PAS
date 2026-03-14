import boto3
import json
import base64
import fitz          # PyMuPDF — converts PDF pages to PNG bytes
from pathlib import Path
from PIL import Image
import io

print("✅ All imports successful")



# ─────────────────────────────────────────────────────────────────────────────
# AWS BOTO3 SETUP
# The hackathon sandbox credentials are picked up automatically from:
#   1. Environment variables  (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
#   2. IAM Role attached to the SageMaker / EC2 instance
#   3. ~/.aws/credentials  (if running locally)
# No explicit key passing needed — boto3 uses the credential chain.
# ─────────────────────────────────────────────────────────────────────────────

AWS_REGION = "us-east-1"   # ← change if your sandbox is in a different region

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
)

# ─── Model IDs ────────────────────────────────────────────────────────────────
# ✅ Use these — directly available in your sandbox (us-east-1)
PRO_MODEL_ID     = "amazon.nova-pro-v1:0"        # multimodal, best for Agent 1
LITE_MODEL_ID    = "amazon.nova-lite-v1:0"        # multimodal, fast
MICRO_MODEL_ID   = "amazon.nova-micro-v1:0"       # text-only, cheapest
PREMIER_MODEL_ID = "amazon.nova-premier-v1:0"     # most capable (new!)

print(f"✅ Bedrock client created in region: {AWS_REGION}")
print(f"   Using model: {PRO_MODEL_ID}")



def image_to_base64(image_path: str, max_size: tuple = (1600, 1600)) -> dict:
    """
    Load an image file, optionally resize it, and return a Nova-compatible
    image content block dict.

    Nova Pro supports: jpeg, png, gif, webp
    Payload limit: 25 MB total across all images.
    """
    path = Path(image_path)
    fmt = path.suffix.lstrip(".").lower()
    if fmt == "jpg":
        fmt = "jpeg"

    img = Image.open(path)
    img.thumbnail(max_size, Image.LANCZOS)   # resize in-place, preserves aspect ratio

    buffer = io.BytesIO()
    img.save(buffer, format=fmt.upper())
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    print(f"  📷 {path.name}  [{img.size[0]}×{img.size[1]}px, {len(b64)//1024} KB b64]")
    return {
        "image": {
            "format": fmt,
            "source": {"bytes": b64}
        }
    }


def pdf_pages_to_base64(pdf_path: str, dpi: int = 150) -> list:
    """
    Convert each page of a PDF to a PNG image and return a list of
    Nova-compatible image content block dicts.

    Uses PyMuPDF (fitz) — no Poppler dependency required.
    """
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num, page in enumerate(doc):
        zoom = dpi / 72          # 72 DPI is PDF default
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        print(f"  📄 {Path(pdf_path).name} — page {page_num+1}  [{pix.width}×{pix.height}px, {len(b64)//1024} KB b64]")
        blocks.append({
            "image": {
                "format": "png",
                "source": {"bytes": b64}
            }
        })
    return blocks


print("✅ Document utility functions defined")



# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PATHS  — update these if running from a different working directory
# ─────────────────────────────────────────────────────────────────────────────
DOCS = {
    "lab_report":            "./data/patient_data/lab_report_ai.png",
    "doctor_notes":          "./data/patient_data/doctor_notes_ai.png",
    "patient_info":          "./data/patient_data/patient_ai.png",
    "insurance_card":        "./data/patient_data/insurance_card_ai.png",
    "pretreatment_estimate": "./data/patient_data/medical_pretreatment_estimate_ai.pdf",
}

print("Loading documents...")
image_blocks = []

# Load PNG images
for name in ["lab_report", "doctor_notes", "patient_info", "insurance_card"]:
    image_blocks.append(image_to_base64(DOCS[name]))

# Load PDF (converts each page to PNG)
image_blocks.extend(pdf_pages_to_base64(DOCS["pretreatment_estimate"]))

print(f"\n✅ Total image blocks ready for Nova Pro: {len(image_blocks)}")



# ─── System Prompt ────────────────────────────────────────────────────────────
AGENT1_SYSTEM = [
    {
        "text": """You are a highly accurate medical document intelligence agent.
Your job is to read a set of clinical documents (lab reports, doctor notes,
patient information sheets, insurance cards, and cost estimates) and extract
all relevant fields required for insurance prior-authorization.

RULES:
- Return ONLY a valid JSON object. No markdown, no preamble, no explanation.
- If a field is not found in any document, set its value to null.
- Normalize all field names to snake_case.
- For dates, use YYYY-MM-DD format.
- For currency amounts, return numeric values only (no $ sign).
- Extract ALL CPT codes and ALL ICD-10 codes you can find (as arrays).
"""
    }
]

print("✅ System prompt defined")


# ─── Extraction Schema Prompt ─────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are given the following medical documents as images.
Read every document carefully and extract ALL of the fields listed below.

Return a single JSON object matching EXACTLY this schema:

{
  "patient": {
    "name": string,
    "dob": string,              // YYYY-MM-DD
    "age": integer,
    "gender": string,
    "mrn": string,              // Medical Record Number / Patient ID
    "address": string,
    "city": string,
    "state": string,
    "zip": string,
    "phone": string,
    "email": string,
    "emergency_contact_name": string,
    "emergency_contact_phone": string,
    "emergency_contact_relationship": string
  },
  "insurance": {
    "insurer_name": string,
    "plan_type": string,        // PPO, HMO, EPO, etc.
    "policy_number": string,
    "group_number": string,
    "member_id": string,
    "policyholder_name": string,
    "relationship_to_patient": string,
    "copay_office_visit": number,
    "copay_er": number,
    "copay_specialist": number,
    "administered_by": string,
    "underwritten_by": string,
    "bin": string,
    "pcn": string,
    "rxgrp": string,
    "customer_service_phone": string
  },
  "clinical": {
    "diagnosis": string,                   // human-readable
    "icd10_codes": [string],               // e.g. ["M54.16"]
    "chief_complaint": string,
    "symptom_duration_weeks": integer,
    "pain_score": integer,
    "clinical_findings": string,
    "prior_treatments": [string],          // list of previously tried treatments
    "requested_procedure": string,
    "cpt_codes": [string],                 // e.g. ["72148", "72148-26", "99204"]
    "ordering_physician": string,
    "physician_phone": string,
    "physician_specialty": string
  },
  "facility": {
    "hospital_name": string,
    "hospital_address": string,
    "date_of_service": string,             // YYYY-MM-DD
    "date_of_proposed_treatment": string   // YYYY-MM-DD
  },
  "cost_estimate": {
    "line_items": [
      {
        "description": string,
        "cpt_code": string,
        "estimated_price": number
      }
    ],
    "total_estimated_cost": number
  },
  "lab_results": {
    "hba1c": string,
    "cholesterol_total": string,
    "ldl": string,
    "hdl": string,
    "triglycerides": string,
    "creatinine": string,
    "egfr": string,
    "alt": string,
    "ast": string
  },
  "pharmacy": {
    "pharmacy_name": string,
    "pharmacy_address": string,
    "pharmacy_phone": string
  },
  "prescribed_medications": [string],     // list of medications prescribed
  "extraction_confidence": "high" | "medium" | "low",
  "missing_fields": [string]              // list any fields you could not find
}
"""

print("✅ Extraction schema prompt defined")




# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — NOVA PRO MULTIMODAL API CALL
# All 5 document images are sent in a single request alongside the schema prompt.
# Nova Pro supports up to 25 MB payload and multiple images per turn.
# ─────────────────────────────────────────────────────────────────────────────

# Build the user message content: all image blocks FIRST, then the text prompt
user_content = image_blocks + [{"text": EXTRACTION_PROMPT}]

message_list = [
    {
        "role": "user",
        "content": user_content
    }
]

inf_params = {
    "max_new_tokens": 2000,
    "temperature": 0.1,    # low temperature → deterministic extraction
    "top_p": 0.9,
    "top_k": 20
}

native_request = {
    "messages": message_list,
    "system": AGENT1_SYSTEM,
    "inferenceConfig": inf_params,
}

print("🚀 Sending all documents to Amazon Nova Pro...")
print(f"   Total content blocks: {len(user_content)} ({len(image_blocks)} images + 1 prompt)")

response = bedrock_client.invoke_model(
    modelId=PRO_MODEL_ID,
    body=json.dumps(native_request)
)

request_id = response["ResponseMetadata"]["RequestId"]
model_response = json.loads(response["body"].read())

print(f"\n✅ Response received  |  RequestId: {request_id}")
print(f"   Stop reason: {model_response.get('stopReason', 'N/A')}")
print(f"   Input tokens:  {model_response.get('usage', {}).get('inputTokens', 'N/A')}")
print(f"   Output tokens: {model_response.get('usage', {}).get('outputTokens', 'N/A')}")




# ─────────────────────────────────────────────────────────────────────────────
# PARSE & VALIDATE THE EXTRACTED JSON
# ─────────────────────────────────────────────────────────────────────────────

raw_text = model_response["output"]["message"]["content"][0]["text"]

# Strip accidental markdown fences (```json ... ```) if model adds them
clean_text = raw_text.strip()
if clean_text.startswith("```"):
    clean_text = clean_text.split("```")[1]
    if clean_text.lower().startswith("json"):
        clean_text = clean_text[4:]
clean_text = clean_text.strip()

try:
    extracted_data = json.loads(clean_text)
    print("✅ JSON parsed successfully")
except json.JSONDecodeError as e:
    print(f"❌ JSON parse error: {e}")
    print("Raw output:\n", raw_text)
    extracted_data = {}

# Pretty print
print("\n" + "="*70)
print("📋  EXTRACTED MEDICAL ENTITIES")
print("="*70)
print(json.dumps(extracted_data, indent=2))






