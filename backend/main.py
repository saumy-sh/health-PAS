"""
main.py
────────
InsuranceHelper — FastAPI Backend Entry Point

Run with:
    cd backend
    uvicorn main:app --reload --port 8001
"""

import sys
from pathlib import Path

# Ensure parent dir (health-PAS root) is on sys.path so agents can be imported
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes.analyse import router as analyse_router

app = FastAPI(
    title="InsuranceHelper API",
    description="Pre-authorization pipeline backend (Granular Testing)",
    version="1.0.0",
)

# ── CORS — allow the frontend (any origin during dev) ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(analyse_router)

# ── Static Files ────────────────────────────────────────────────────────────────
# Mount the output directory to serve the filled PDFs
output_path = ROOT / "output"
output_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(output_path)), name="static")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "InsuranceHelper API"}
