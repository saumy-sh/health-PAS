"""
routes/analyse.py
─────────────────
Endpoints for the pre-authorization pipeline.

Flow (always runs straight through, no mid-pipeline upload prompts):
  POST /analyse/                    → upload files, get session_id + file paths
  POST /analyse/document_ocr        → Agent 1: OCR + extraction
  POST /analyse/policy_checker      → Agent 2: extract policy fields
  POST /analyse/policy_retriever    → Agent 3: retrieve requirements
  POST /analyse/document_checker    → Agent 4: verify docs (informational only, never blocks)
  POST /analyse/eligibility_reasoning → Agent 5: clinical assessment
  POST /analyse/form_filler         → Agent 6: final JSON report
"""

import asyncio
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Body
from pydantic import BaseModel

import sys
import os

LAMBDA_TASK_ROOT = os.environ.get("LAMBDA_TASK_ROOT", "")

if LAMBDA_TASK_ROOT and LAMBDA_TASK_ROOT not in sys.path:
    sys.path.insert(0, LAMBDA_TASK_ROOT)

_HERE    = Path(__file__).parent
_BACKEND = _HERE.parent
_ROOT    = _BACKEND.parent

for _p in [str(_ROOT), str(_BACKEND)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────────────────────────────────────

router   = APIRouter(prefix="/analyse", tags=["analysis"])
UPLOAD_TMP = Path(tempfile.gettempdir()) / "insurancehelper_analyse"
UPLOAD_TMP.mkdir(parents=True, exist_ok=True)


class SessionResponse(BaseModel):
    session_id: str
    files: List[str]


def _safe_serialize(obj):
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(i) for i in obj]
    try:
        import json
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


async def in_thread(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/", response_model=SessionResponse)
async def analyse_root(files: List[UploadFile] = File(...)):
    """Upload documents, receive session_id and saved file paths."""
    print(f"\n[SERVER] Uploading {len(files)} file(s)...")
    session_id  = str(uuid.uuid4())
    session_dir = UPLOAD_TMP / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for upload in files:
        safe_name = Path(upload.filename).name
        dest = session_dir / safe_name
        with open(dest, "wb") as f:
            content = await upload.read()
            f.write(content)
        saved_paths.append(str(dest))
        print(f"  Saved: {dest}")

    print(f"[SERVER] Session {session_id} — {len(saved_paths)} file(s) saved.")
    return {"session_id": session_id, "files": saved_paths}


@router.post("/document_ocr")
async def document_ocr(file_paths: List[str] = Body(...)):
    """Agent 1: OCR and extract content from all uploaded documents."""
    import agent1_document_intelligence as agent1
    try:
        print(f"\n[SERVER] Agent 1 — processing {len(file_paths)} file(s)")
        result = await in_thread(agent1.run, file_paths)
        print(f"[SERVER] Agent 1 — identified {len(result.get('documents', []))} document(s)")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 1 FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 1 failed: {e}")


@router.post("/policy_checker")
async def policy_checker(agent1_result: Dict[str, Any] = Body(...)):
    """Agent 2: Extract policy search fields from Agent 1 output."""
    import agent2_policy_checker as agent2
    try:
        print(f"\n[SERVER] Agent 2 — extracting policy fields")
        result = await in_thread(agent2.run, agent1_result)
        print(f"[SERVER] Agent 2 — ready={result.get('ready')}, insurer={result.get('policy_search_fields', {}).get('insurer_name')}")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 2 FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 2 failed: {e}")


@router.post("/policy_retriever")
async def policy_retriever(payload: Dict[str, Any] = Body(...)):
    """Agent 3: Retrieve policy requirements for the identified procedure."""
    import agent3_policy_retrieval as agent3
    try:
        print(f"\n[SERVER] Agent 3 — retrieving policy requirements")
        result = await in_thread(agent3.run, payload)
        print(f"[SERVER] Agent 3 — procedure={result.get('procedure_identified')}, "
              f"doc_reqs={len(result.get('document_requirements', []))}, "
              f"med_reqs={len(result.get('medical_requirements', []))}")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 3 FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 3 failed: {e}")


@router.post("/document_checker")
async def document_checker(agent3_result: Dict[str, Any] = Body(...)):
    """
    Agent 4: Verify submitted documents against requirements.
    This is INFORMATIONAL ONLY — it never blocks the pipeline.
    can_proceed is always treated as True downstream.
    """
    import agent4_document_checker as agent4
    try:
        print(f"\n[SERVER] Agent 4 — verifying documents (informational, non-blocking)")
        result = await in_thread(agent4.run, agent3_result)

        satisfied = len(result.get("satisfied", []))
        missing   = len(result.get("missing_documents", []))
        partial   = len(result.get("partial_documents", []))
        print(f"[SERVER] Agent 4 — satisfied={satisfied}, missing={missing}, partial={partial}")

        # Force can_proceed=True so the frontend always continues
        result["can_proceed"] = True

        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 4 FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 4 failed: {e}")


@router.post("/eligibility_reasoning")
async def eligibility_reasoning(agent4_result: Dict[str, Any] = Body(...)):
    """Agent 5: Clinical eligibility assessment and approval probability."""
    import agent5_eligibility_reasoning as agent5
    try:
        print(f"\n[SERVER] Agent 5 — clinical eligibility assessment")
        result = await in_thread(agent5.run, agent4_result)
        print(f"[SERVER] Agent 5 — determination={result.get('determination')}, "
              f"probability={result.get('approval_probability')}")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 5 FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 5 failed: {e}")


@router.post("/form_filler")
async def form_filler(agent5_result: Dict[str, Any] = Body(...)):
    """
    Agent 6: Generate the final structured JSON pre-authorization report.
    (Named form_filler to match the existing frontend API call.)
    """
    import agent6_report_generator as agent6
    try:
        print(f"\n[SERVER] Agent 6 — generating final report")
        result = await in_thread(agent6.run, agent5_result)
        print(f"[SERVER] Agent 6 — report generated, "
              f"determination={result.get('approval_assessment', {}).get('determination')}")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 6 FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 6 failed: {e}")
