"""
agent1_document_intelligence.py
─────────────────────────────────
Agent 1 — Document Intelligence

Role:
  For every document submitted, determine its type and produce a natural-language
  paragraph describing ALL of its content.  No fixed schema — the LLM reads each
  document and writes a free-form description.

  Works for any document type: clinical notes, insurance cards, lab reports,
  X-ray / imaging scans, prescriptions, referral letters, cost estimates, etc.

Output:
  {
    "documents": [
      {
        "document_type": "Human-readable document type name",
        "content":       "Full prose paragraph describing everything in this document"
      },
      ...
    ]
  }

This output is consumed by every downstream agent in the pipeline.
"""

import json
import re
import base64
import io
from pathlib import Path
from typing import Union

import fitz          # PyMuPDF — pip install pymupdf
from PIL import Image

from bedrock_client import invoke, PRO_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Supported formats
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PDF_EXTENSIONS   = {".pdf"}


# ─────────────────────────────────────────────────────────────────────────────
# File loaders
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_block(image_path: str, max_size: tuple = (1600, 1600)) -> dict:
    path = Path(image_path)
    fmt  = path.suffix.lstrip(".").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in {"png", "jpeg", "webp", "gif"}:
        fmt = "png"
    img = Image.open(path)
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    save_fmt = "PNG" if fmt in {"png", "gif"} else fmt.upper()
    img.save(buf, format=save_fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    out_fmt = "png" if save_fmt == "PNG" else fmt
    print(f"  [Agent1] Image  : {path.name}  ({img.size[0]}x{img.size[1]}px)  → {out_fmt}")
    return {
        "image": {"format": out_fmt, "source": {"bytes": b64}},
        "_filename": path.name,
    }


def _pdf_to_blocks(pdf_path: str, dpi: int = 150) -> list:
    doc    = fitz.open(pdf_path)
    blocks = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        buf = io.BytesIO(pix.tobytes("png"))
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        blocks.append({
            "image": {"format": "png", "source": {"bytes": b64}},
            "_filename": f"{Path(pdf_path).name} (page {i+1})",
        })
        print(f"  [Agent1] PDF page {i+1}/{len(doc)} : {Path(pdf_path).name}")
    doc.close()
    return blocks


def _file_to_blocks(file_path: str) -> list:
    path   = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return [_image_to_block(file_path)]
    elif suffix in PDF_EXTENSIONS:
        return _pdf_to_blocks(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _normalise_input(documents) -> list:
    if isinstance(documents, str):
        return [documents]
    if isinstance(documents, list):
        return documents
    if isinstance(documents, dict):
        return list(documents.values())
    raise TypeError(f"Unsupported input type: {type(documents)}")


def _load_documents(file_paths: list) -> list:
    all_blocks = []
    for fp in file_paths:
        try:
            all_blocks.extend(_file_to_blocks(fp))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [Agent1] ⚠  Skipping '{fp}': {exc}")
    return all_blocks


# ─────────────────────────────────────────────────────────────────────────────
# Robust JSON parser
# ─────────────────────────────────────────────────────────────────────────────

def _safe_parse_json(raw: str, context: str = "") -> dict:
    """Try multiple strategies to parse potentially malformed JSON."""
    text = raw.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 3: extract the largest {...} block
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"  [Agent1] ❌ JSON parse failed{' in ' + context if context else ''}")
    print(f"  [Agent1] Raw response (first 500 chars): {raw[:500]}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = [
    {
        "text": (
            "You are an expert clinical document processing and transcription assistant. "
            "Your role is to perform high-fidelity OCR and exhaustive data extraction from images. "
            "For each document, you MUST extract absolutely ALL information present, leaving nothing out. "
            "Transcribe all key-value pairs, dates, names, provider information, policy details, lab results, "
            "measurements, clinical notes, subjective/objective findings, observations, impressions, and treatment plans. "
            "Include all administrative details such as addresses, phone numbers, and IDs. "
            "Do not summarize or omit information; provide a highly detailed, comprehensive textual representation "
            "of everything found in the document. "
            "Return ONLY valid JSON — no markdown, no preamble."
        )
    }
]

EXTRACTION_PROMPT = """Analyze the provided healthcare document images. For each distinct document detected,
provide its classification and an exhaustive compilation of all data found within it.

GUIDELINES:
1. document_type: Standardized classification (e.g., 'Clinical Note', 'Insurance ID', 'Laboratory Report', 'Imaging Report', 'Physician Referral', 'Pre-treatment Estimate').
2. content: A highly exhaustive and comprehensive textual representation of absolutely EVERYTHING in the document:
   - Administrative details: Hospital/Clinic names, provider names, addresses, contact info, dates of service, Patient name, DOB, Insurance provider, Member/Policy/Group IDs.
   - Clinical context: Diagnoses, ICD-10/CPT/ADA codes, medications, full clinical findings, subjective/objective notes, assessments, and treatment plans.
   - Lab/Imaging results: All test names, values, units, reference ranges, anatomical sites, and clinical observations.
   - Financial/Estimate details: All procedures, quantities, costs, insurance coverage amounts, and patient responsibilities.
   - Ensure EVERY SINGLE PIECE of text, number, and data point on the document is captured in this field. Do not drop any detail, no matter how small.
3. If an image contains multiple separate documents, provide a separate entry for each.

Return ONLY this JSON structure exactly:
{
  "documents": [
    {
      "document_type": "string",
      "content": "exhaustive extraction of all text and data"
    }
  ]
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(documents: Union[str, list, dict]) -> dict:
    """
    Single-pass extraction:
      Load all documents as images → ask LLM to describe each one in prose.
    Returns: { "documents": [{document_type, content}, ...] }
    """
    print("\n[Agent 1] Document Intelligence — START")

    file_paths   = _normalise_input(documents)
    print(f"[Agent 1] Input files ({len(file_paths)}):")
    for fp in file_paths:
        print(f"  • {fp}")

    image_blocks = _load_documents(file_paths)
    print(f"[Agent 1] Total content blocks: {len(image_blocks)}")

    if not image_blocks:
        raise ValueError("[Agent 1] No valid documents could be loaded. Aborting.")

    # Strip internal _filename keys before sending to LLM (not a valid content block key)
    api_blocks = [
        {k: v for k, v in block.items() if k != "_filename"}
        for block in image_blocks
    ]

    # ── Single pass: describe all documents ──────────────────────────────────
    print("[Agent 1] Calling Nova Pro — document description pass (max_tokens=4096)...")
    messages = [{"role": "user", "content": api_blocks + [{"text": EXTRACTION_PROMPT}]}]

    raw = invoke(
        model_id=PRO_MODEL_ID,
        messages=messages,
        system=SYSTEM_PROMPT,
        max_tokens=4096,
        temperature=0.1,
    )

    if raw.startswith("ERROR:"):
        print(f"  [Agent 1] ⚠ Nova Pro error: {raw}")
        print("  [Agent 1] Retrying with Nova Lite (Agent 1 Fallback)...")
        from bedrock_client import LITE_MODEL_ID
        raw = invoke(
            model_id=LITE_MODEL_ID,
            messages=messages,
            system=SYSTEM_PROMPT,
            max_tokens=4096,
            temperature=0.1,
        )

    if raw.startswith("ERROR:"):
        raise RuntimeError(f"[Agent 1] ❌ Pipeline Halted: Both models refused or failed to process images. {raw}")

    parsed = _safe_parse_json(raw, context="document description")
    if not parsed or not parsed.get("documents"):
        raise RuntimeError(
            "[Agent 1] ❌ Failed — could not parse document descriptions.\n"
            f"Raw (first 500 chars): {raw[:500]}"
        )

    documents_list = parsed["documents"]

    # Normalise: ensure every entry has both required keys
    clean_docs = []
    for doc in documents_list:
        if not isinstance(doc, dict):
            continue
        doc_type = str(doc.get("document_type", "Unknown Document")).strip()
        content  = str(doc.get("content", "")).strip()
        if doc_type and content:
            clean_docs.append({"document_type": doc_type, "content": content})

    output = {"documents": clean_docs}

    print("\n[Agent 1] Extraction complete.")
    print(f"  Documents identified: {len(clean_docs)}")
    for doc in clean_docs:
        preview = doc["content"][:80].replace("\n", " ")
        print(f"    [{doc['document_type']}]  {preview}...")

    # Save Agent 1 output to JSON
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "agent1_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  [Agent 1] Output saved to: {out_file}")

    print("[Agent 1] Document Intelligence — DONE\n")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run([
        "data/patient_data/lab_report_ai.png",
        "data/patient_data/docotr_notes_ai.png",
        "data/patient_data/patient_ai.png",
        "data/patient_data/insurance_card_ai.png",
        "data/patient_data/medical_pretreatment_estimate_ai.pdf",
    ])
    print(json.dumps(result, indent=2, default=str))
