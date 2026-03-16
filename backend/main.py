import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from backend.routes.analyse import router as analyse_router

app = FastAPI(title="InsuranceHelper API", version="1.0.0")

# Fix: strip /prod prefix added by API Gateway
handler = Mangum(app, lifespan="off", api_gateway_base_path="/prod")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyse_router)

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "InsuranceHelper API"}