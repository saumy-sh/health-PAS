"""
routes/analyse.py
─────────────────
Granular endpoints for testing each individual agent in the pipeline.
"""

import asyncio
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Body
from pydantic import BaseModel

# ── sys.path setup for Lambda ─────────────────────────────────────────────────
# In Lambda, all files are copied to /var/task (LAMBDA_TASK_ROOT)
# So agents like agent1_document_intelligence.py are at /var/task/
# We need /var/task on sys.path so they can be imported directly
import sys
import os

LAMBDA_TASK_ROOT = os.environ.get("LAMBDA_TASK_ROOT", "")

# Add Lambda task root to sys.path (so agent1, agent2... can be found)
if LAMBDA_TASK_ROOT and LAMBDA_TASK_ROOT not in sys.path:
    sys.path.insert(0, LAMBDA_TASK_ROOT)

# Also add parent dirs for local dev
_HERE = Path(__file__).parent          # backend/routes/
_BACKEND = _HERE.parent                # backend/
_ROOT = _BACKEND.parent                # project root

for _p in [str(_ROOT), str(_BACKEND)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/analyse", tags=["analysis"])

# Temp dir for uploaded files — /tmp is the only writable dir in Lambda
UPLOAD_TMP = Path(tempfile.gettempdir()) / "insurancehelper_analyse"
UPLOAD_TMP.mkdir(parents=True, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str
    files: List[str]

# ── Helper ────────────────────────────────────────────────────────────────────

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

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=SessionResponse)
async def analyse_root(files: List[UploadFile] = File(...)):
    """
    Initial step: upload documents and get a session ID.
    """
    print(f"\n[SERVER] Uploading {len(files)} files...")
    session_id = str(uuid.uuid4())
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

    print(f"[SERVER] Saved files to session {session_id}: {saved_paths}")
    return {"session_id": session_id, "files": saved_paths}


@router.post("/document_ocr")
async def document_ocr(file_paths: List[str] = Body(...)):
    """
    Agent 1: Document Intelligence (OCR).
    Expects: List of absolute file paths.
    """
    import agent1_document_intelligence as agent1

    try:
        print(f"\n[SERVER] Agent 1 START: processing {len(file_paths)} files")
        result = await in_thread(agent1.run, file_paths)
        print(f"[SERVER] Agent 1 SUCCESS: identified {len(result.get('documents', []))} documents")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 1 FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 1 failed: {str(e)}")


@router.post("/policy_checker")
async def policy_checker(agent1_result: Dict[str, Any] = Body(...)):
    """
    Agent 2: Policy Checker.
    Expects: Agent 1 result JSON.
    """
    import agent2_policy_checker as agent2

    try:
        print(f"\n[SERVER] Agent 2 START: processing Agent 1 result (size: {len(str(agent1_result))} chars)")
        result = await in_thread(agent2.run, agent1_result)
        print(f"[SERVER] Agent 2 SUCCESS: generated policy checks (size: {len(str(result))} chars)")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 2 FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 2 failed: {str(e)}")


@router.post("/policy_retriever")
async def policy_retreiver(payload: Dict[str, Any] = Body(...)):
    """
    Agent 3: Policy Retrieval.
    Expects: Agent 2 output.
    """
    import agent3_policy_retrieval as agent3

    try:
        print(f"\n[SERVER] Agent 3 START: processing payload (size: {len(str(payload))} chars)")
        result = await in_thread(agent3.run, payload)
        print(f"[SERVER] Agent 3 SUCCESS: retrieved policies (size: {len(str(result))} chars)")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 3 FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 3 failed: {str(e)}")


@router.post("/document_checker")
async def document_checker(agent3_result: Dict[str, Any] = Body(...)):
    """
    Agent 4: Document Checker.
    Expects: Agent 3 result JSON.
    """
    import agent4_document_checker as agent4

    try:
        print(f"\n[SERVER] Agent 4 START: processing Agent 3 result (size: {len(str(agent3_result))} chars)")
        result = await in_thread(agent4.run, agent3_result)
        print(f"[SERVER] Agent 4 SUCCESS: checked documents (size: {len(str(result))} chars)")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 4 FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 4 failed: {str(e)}")


@router.post("/eligibility_reasoning")
async def eligibility_reasoning(agent4_result: Dict[str, Any] = Body(...)):
    """
    Agent 5: Eligibility Reasoning.
    Expects: Agent 4 result JSON.
    """
    import agent5_eligibility_reasoning as agent5

    try:
        print(f"\n[SERVER] Agent 5 START")
        result = await in_thread(agent5.run, agent4_result)
        print(f"[SERVER] Agent 5 SUCCESS")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 5 FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 5 failed: {str(e)}")


@router.post("/form_filler")
async def form_filler(agent5_result: Dict[str, Any] = Body(...)):
    """
    Agent 6: Form Filler.
    Expects: Agent 5 result JSON.
    """
    import agent6_form_filler as agent6

    try:
        print(f"\n[SERVER] Agent 6 START")
        result = await in_thread(agent6.run, agent5_result)
        print(f"[SERVER] Agent 6 SUCCESS: PDF saved to {result.get('filled_pdf_path')}")
        return _safe_serialize(result)
    except Exception as e:
        print(f"[SERVER] Agent 6 FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 6 failed: {str(e)}")
