# InsuranceHelper — AI Prior Authorization Pipeline

> Automated health insurance pre-authorization using a six-agent AI pipeline powered by Amazon Nova on AWS Bedrock.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Next.js 16](https://img.shields.io/badge/next.js-16-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)
[![HIPAA Compliant](https://img.shields.io/badge/HIPAA-Compliant-red.svg)]()

---

## Table of Contents

1. [Overview](#overview)
2. [Why This Exists](#why-this-exists)
3. [Architecture](#architecture)
4. [Agent Pipeline](#agent-pipeline)
5. [Project Structure](#project-structure)
6. [Prerequisites](#prerequisites)
7. [Environment Variables](#environment-variables)
8. [Build & Test Locally](#build--test-locally)
9. [API Reference](#api-reference)
10. [Policy Knowledge Base](#policy-knowledge-base)
11. [Key Design Decisions](#key-design-decisions)

---

## Overview

InsuranceHelper is a full-stack web application that automates the prior authorization (PA) process for medical procedures. A clinician or administrator uploads patient documents — clinical notes, insurance cards, lab reports, prescriptions, cost estimates — and the system runs a six-stage AI pipeline that:

- Extracts all content from every document via multimodal OCR
- Identifies the patient's insurance policy
- Retrieves the insurer's specific authorization requirements for the procedure
- Verifies which requirements the submitted documents satisfy
- Performs clinical eligibility reasoning, detecting conflicting evidence across documents
- Generates a structured JSON report with approval probability and next steps

The entire pipeline runs in under two minutes.

---

## Why This Exists

Prior authorization is the single largest administrative burden in U.S. healthcare:

| Statistic | Source |
|-----------|--------|
| **94%** of physicians report PA delays necessary care | AMA Survey 2023 |
| **33%** of patients experience an adverse event while waiting | AMA Survey 2023 |
| **45 hours/week** spent by practice staff on PA requests | AMA Benchmark Survey |
| **1 in 4** patients abandon treatment due to PA delays | AMA Survey 2023 |
| **$528M/year** spent on PA administration by U.S. physicians | JAMA Internal Medicine |

InsuranceHelper addresses this by replacing manual form-filling and back-and-forth with insurers with an automated, auditable, AI-driven assessment pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser (Next.js 16)                        │
│  Landing Page  ──►  /analyse  ──►  Step Progress  ──►  Final Report │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP (REST)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend  (port 8001)                      │
│                                                                     │
│   POST /analyse/              →  Upload files, create session       │
│   POST /analyse/document_ocr  →  Agent 1                           │
│   POST /analyse/policy_checker →  Agent 2                          │
│   POST /analyse/policy_retriever → Agent 3                         │
│   POST /analyse/document_checker → Agent 4                         │
│   POST /analyse/eligibility_reasoning → Agent 5                    │
│   POST /analyse/form_filler   →  Agent 6                           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ boto3 (AWS SDK)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AWS Bedrock Runtime                          │
│                                                                     │
│   amazon.nova-pro-v1:0     →  Agents 1, 5  (multimodal)            │
│   amazon.nova-lite-v1:0    →  Agent 3      (text reasoning)        │
│   amazon.nova-micro-v1:0   →  Agents 2, 4  (fast text extraction)  │
└─────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS v4, Framer Motion |
| Backend | Python 3.10+, FastAPI, Uvicorn |
| AI Models | Amazon Nova Pro, Nova Lite, Nova Micro (via AWS Bedrock) |
| Document Processing | PyMuPDF (fitz), Pillow |
| Cloud | AWS Bedrock (us-east-1) |

---

## Agent Pipeline

The pipeline is a sequential chain where each agent receives the output of the previous agent, enriches it, and passes it forward. Documents from Agent 1 are carried through the entire chain so every agent has access to the full extracted content.

```
Documents
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 1 — Document Intelligence                        │
│  Model  : amazon.nova-pro-v1:0  (multimodal)            │
│  Input  : File paths (PDF / PNG / JPG / WEBP)           │
│  Output : { documents: [{ document_type, content }] }   │
│                                                         │
│  Converts every file to images and sends all pages to   │
│  Nova Pro in a single multimodal API call. Produces a   │
│  full prose description of every document — clinical    │
│  notes, insurance cards, lab results, imaging reports,  │
│  prescriptions, cost estimates — regardless of type.    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 2 — Policy Field Extraction                      │
│  Model  : amazon.nova-micro-v1:0  (text-only, fast)     │
│           Fallback: amazon.nova-lite-v1:0               │
│  Input  : Agent 1 output                                │
│  Output : { ready, policy_search_fields,                │
│             missing_critical, documents }               │
│                                                         │
│  Extracts the four required policy fields from the      │
│  combined document text:                                │
│    • insurer_name   — e.g. "BlueCross BlueShield"       │
│    • policy_number  — e.g. "BCBS-99001122"              │
│    • plan_type      — e.g. "PPO"                        │
│    • member_id      — e.g. "BCBS-9900122"               │
│                                                         │
│  If any are missing, pipeline PAUSES and asks user to   │
│  upload an insurance card. On re-upload, new documents  │
│  are merged with all previous Agent 1 content before    │
│  Agent 2 runs again. This is the ONLY allowed blocker.  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 3 — Policy Requirements Retrieval                │
│  Model  : amazon.nova-lite-v1:0  (primary)              │
│  Input  : Agent 2 output                                │
│  Output : { authorization_required, procedure,          │
│             document_requirements[],                    │
│             medical_requirements[] }                    │
│                                                         │
│  First checks a local JSON knowledge base               │
│  (policy_requirements.json) for the insurer + procedure │
│  combination. If not found, falls back to Nova Lite to  │
│  reason about standard authorization requirements.      │
│                                                         │
│  Produces two distinct requirement lists:               │
│    document_requirements — what documents must exist    │
│    medical_requirements  — clinical criteria to be met  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 4 — Document Verification                        │
│  Model  : amazon.nova-micro-v1:0  (text-only, fast)     │
│  Input  : Agent 3 output                                │
│  Output : { can_proceed, satisfied[],                   │
│             missing_documents[], partial_documents[] }  │
│                                                         │
│  Checks every submitted document against the            │
│  document_requirements list from Agent 3.               │
│  Each requirement is marked:                            │
│    satisfied — document present and complete            │
│    partial   — document present but info missing        │
│    missing   — no document covers this requirement      │
│                                                         │
│  INFORMATIONAL ONLY — never blocks the pipeline.        │
│  can_proceed is always forced True by the backend.      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 5 — Clinical Eligibility Assessment              │
│  Model  : amazon.nova-pro-v1:0  (most capable)          │
│  Input  : Agent 4 output                                │
│  Output : { approval_probability, determination,        │
│             requirements_checked[], conflicts_detected, │
│             denial_reasons[], clinical_summary }        │
│                                                         │
│  Checks every medical_requirement against the actual    │
│  document content with three critical rules:            │
│                                                         │
│  1. SCAN ALL DOCUMENTS — reads every document for       │
│     each requirement, not just the first match.         │
│                                                         │
│  2. CONFLICT DETECTION — if two documents give          │
│     different values (e.g. "2 weeks" vs "7 weeks"),     │
│     both are flagged with source citations.             │
│                                                         │
│  3. CONSERVATIVE RULE — for numeric thresholds,         │
│     the LOWEST value found is used. Never cherry-picks  │
│     the value that favours approval.                    │
│                                                         │
│  A programmatic safety net post-processes the LLM       │
│  output: if a conflict is detected and values differ    │
│  by >50%, the status is downgraded from "met" to        │
│  "partial" regardless of LLM output.                    │
│                                                         │
│  Probability scale:                                     │
│    0.85–1.0  → APPROVED                                 │
│    0.65–0.84 → LIKELY_APPROVED                          │
│    0.40–0.64 → PENDING_REVIEW                           │
│    0.20–0.39 → LIKELY_DENIED                            │
│    0.0–0.19  → DENIED                                   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 6 — Final Report Generator                       │
│  Model  : none (pure Python synthesis)                  │
│  Input  : Agent 5 output (contains full chain)          │
│  Output : Structured JSON pre-authorization report      │
│                                                         │
│  Navigates the nested agent chain (5→4→3→2) to pull     │
│  all relevant fields and assemble a clean report:       │
│    patient_info, insurance_info, requested_procedure,   │
│    document_requirements_check, medical_requirements,   │
│    approval_assessment, next_steps                      │
│                                                         │
│  Saved to ./output/final_preauth_report_<mrn>_<ts>.json │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
health-PAS/
│
├── agent1_document_intelligence.py   # OCR + content extraction
├── agent2_policy_checker.py          # Policy field extraction
├── agent3_policy_retrieval.py        # Requirements retrieval
├── agent4_document_checker.py        # Document verification
├── agent5_eligibility_reasoning.py   # Clinical assessment
├── agent6_report_generator.py        # Final report synthesis
│
├── bedrock_client.py                 # Shared AWS Bedrock client + model IDs
├── orchestrator.py                   # Direct pipeline runner (CLI)
├── policy_requirements.json          # Local insurer knowledge base
│
├── backend/
│   ├── main.py                       # FastAPI app entry point
│   ├── routes/
│   │   └── analyse.py                # All /analyse/* endpoints
│   └── s3_utils.py                   # S3 upload helper (optional)
│
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── layout.tsx            # Root layout
│   │       ├── page.tsx              # Landing page
│   │       ├── globals.css           # Global styles
│   │       └── analyse/
│   │           └── page.tsx          # Analysis chat interface
│   └── src/lib/
│       └── api.ts                    # Frontend API client
│
├── data/
│   ├── patient_data/                 # Sample patient documents
│   └── forms/                        # EWA pre-auth form templates
│
├── output/                           # Generated reports (gitignored)
├── requirements.txt
├── .env                              # Local credentials (gitignored)
└── .gitignore
```

---

## Prerequisites

- **Python 3.10+**
- **Node.js 20+** and **npm**
- **AWS account** with Bedrock access enabled in `us-east-1`
- **AWS IAM user** with `AmazonBedrockFullAccess` policy (or scoped to Nova model ARNs)
- The following Amazon Nova models must be enabled in your AWS Bedrock console:
  - `amazon.nova-pro-v1:0`
  - `amazon.nova-lite-v1:0`
  - `amazon.nova-micro-v1:0`

> **Enable model access:** AWS Console → Bedrock → Model access → Request access for all three Nova models. Takes ~1 minute to activate.

---

## Environment Variables

Create a `.env` file in the **project root** (next to `requirements.txt`):

```env
# ── AWS Credentials ───────────────────────────────────────────────────────────
# IAM user credentials with AmazonBedrockFullAccess
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Required if using temporary credentials (STS / SSO / assumed role)
# Leave blank or omit entirely if using long-term IAM user keys
AWS_SESSION_TOKEN=

# AWS region — Nova models are available in us-east-1
AWS_DEFAULT_REGION=us-east-1

# ── Optional: S3 output bucket ────────────────────────────────────────────────
# If set, final reports are also uploaded to this bucket
# Leave blank to use local ./output/ folder only
S3_BUCKET=
```

> **Never commit `.env` to version control.** It is already listed in `.gitignore`.

To get your AWS credentials:
1. AWS Console → IAM → Users → your user → Security credentials
2. Create access key → Application running outside AWS
3. Copy the Access Key ID and Secret Access Key

---

## Build & Test Locally

### 1 — Clone the repository

```bash
git clone https://github.com/your-org/health-PAS.git
cd health-PAS
```

### 2 — Set up Python virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows (PowerShell)
venv\Scripts\Activate.ps1

# Activate — Windows (Command Prompt)
venv\Scripts\activate.bat
```

### 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
```
boto3>=1.34.0
pymupdf==1.24.11
Pillow>=10.0.0
fastapi
uvicorn
python-multipart
python-dotenv
```

### 4 — Create your `.env` file

```bash
# Copy the template and fill in your credentials
cp .env.example .env   # or create .env manually — see Environment Variables above
```

### 5 — Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 6 — Run the backend

Open a terminal, navigate to the `backend/` folder, and start the FastAPI server:

```bash
cd backend
uvicorn main:app --reload --port 8001
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
INFO:     Started reloader process using WatchFiles
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

The API docs are available at: **http://localhost:8001/docs**

### 7 — Run the frontend

Open a **second terminal**, navigate to the `frontend/` folder:

```bash
cd frontend
npm run dev
```

Expected output:
```
  ▲ Next.js 16.1.6 (Turbopack)
  - Local:        http://localhost:3000
  - Ready in 1.2s
```

Open **http://localhost:3000** in your browser.

### 8 — Verify everything is working

1. Visit `http://localhost:3000` — the landing page should load
2. Click **Analyse Documents** → the analysis page opens
3. Upload sample documents from `data/patient_data/`
4. Watch the six-agent pipeline run in the chat interface
5. The final report card appears at step 6

---

## API Reference

All endpoints are prefixed with `/analyse`.

| Method | Endpoint | Description | Input |
|--------|----------|-------------|-------|
| `POST` | `/analyse/` | Upload files, create session | `multipart/form-data` files |
| `POST` | `/analyse/document_ocr` | Agent 1: OCR extraction | `["path1", "path2"]` |
| `POST` | `/analyse/policy_checker` | Agent 2: Policy fields | Agent 1 JSON output |
| `POST` | `/analyse/policy_retriever` | Agent 3: Requirements | Agent 2 JSON output |
| `POST` | `/analyse/document_checker` | Agent 4: Doc verification | Agent 3 JSON output |
| `POST` | `/analyse/eligibility_reasoning` | Agent 5: Clinical assessment | Agent 4 JSON output |
| `POST` | `/analyse/form_filler` | Agent 6: Final report | Agent 5 JSON output |

Each agent endpoint receives the **full output of the previous agent** as its request body — not just the fields it needs. This means the full document list and policy context is always available to every agent.

### Example: Upload files

```bash
curl -X POST http://localhost:8001/analyse/ \
  -F "files=@data/patient_data/insurance_card_ai.png" \
  -F "files=@data/patient_data/doctor_notes_ai.png"
```

Response:
```json
{
  "session_id": "a3f8c2d1-...",
  "files": [
    "/tmp/insurancehelper_analyse/a3f8c2d1-.../insurance_card_ai.png",
    "/tmp/insurancehelper_analyse/a3f8c2d1-.../doctor_notes_ai.png"
  ]
}
```

---

## Policy Knowledge Base

`policy_requirements.json` is a local knowledge base of insurer-specific authorization requirements. Agent 3 checks this first before falling back to LLM reasoning.

Structure:
```json
{
  "BlueCross BlueShield": {
    "procedures": {
      "MRI Lumbar Spine Without Contrast": {
        "cpt_codes": ["72148"],
        "requires_auth": true,
        "document_requirements": [
          {
            "document_type": "Doctor's Clinical Notes",
            "purpose": "Establish medical necessity",
            "info_needed": "Diagnosis ICD-10 code, symptom duration, neurological findings"
          }
        ],
        "medical_requirements": [
          {
            "requirement": "Symptom duration",
            "description": "Patient must have had symptoms for at least 6 weeks",
            "threshold": "6 weeks",
            "importance": "required"
          }
        ]
      }
    }
  }
}
```

Currently includes: **BlueCross BlueShield**, **Cigna**, **UnitedHealthcare**, **Aetna**. Add new insurers or procedures by extending this file — no code changes required.

---

## Key Design Decisions

**Single upload, one re-upload exception**
The pipeline is designed to run start-to-finish on a single document upload. The only allowed mid-pipeline pause is at Agent 2 when insurance policy fields are missing. When the user re-uploads an insurance card, the new document's extracted content is merged with all previously extracted content — nothing is discarded.

**Conservative conflict resolution**
When Agent 5 finds conflicting values across documents (e.g. symptom duration stated as "2 weeks" in a clinical note vs "7 weeks" in a cost estimate), it always uses the lowest value for threshold comparisons. A programmatic safety net further downgrades any "met" status to "partial" when conflicting values differ by more than 50%.

**Agent 4 is informational**
Document completeness checking (Agent 4) never blocks the pipeline. Its results are shown as a checklist in the UI for transparency, but the backend forces `can_proceed = True` before returning. The rationale: incomplete documentation is a finding to report, not a reason to prevent clinical assessment.

**Model selection rationale**
- **Nova Pro** for Agents 1 and 5 — these require either multimodal input (images) or complex multi-document reasoning with conflict detection. Nova Pro is the most capable and handles both.
- **Nova Lite** for Agent 3 — policy retrieval requires moderate reasoning about structured requirements. Nova Lite is faster and sufficient.
- **Nova Micro** for Agents 2 and 4 — pure text extraction and structured comparison tasks. Nova Micro is fastest and cheapest, with Nova Lite as automatic fallback if it fails.
- **No model** for Agent 6 — the final report is pure Python data transformation. No LLM is needed when all the data has already been extracted and assessed.

---

## License

MIT License — see `LICENSE` for details.

---

*Built with Amazon Nova on AWS Bedrock · Statistics from AMA Prior Authorization Surveys 2022–2023*
