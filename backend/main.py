"""
main.py
────────
InsuranceHelper — FastAPI Backend Entry Point
"""


import sys
print(">>> STEP 1: starting main.py", flush=True)

from pathlib import Path
print(">>> STEP 2: pathlib imported", flush=True)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
print(">>> STEP 3: sys.path updated", flush=True)

from fastapi import FastAPI
print(">>> STEP 4: fastapi imported", flush=True)

from backend.routes.analyse import router as analyse_router
print(">>> STEP 5: router imported", flush=True)


import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# ✅ Fixed import — use full package path
from backend.routes.analyse import router as analyse_router

app = FastAPI(
    title="InsuranceHelper API",
    description="Pre-authorization pipeline backend",
    version="1.0.0",
)

# Lambda Handler
handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyse_router)

# ✅ Removed static files mount and output_path.mkdir
# Output files will go to S3 instead

@app.get("/")
async def root():
    return {"status": "ok", "service": "InsuranceHelper API"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "InsuranceHelper API"}