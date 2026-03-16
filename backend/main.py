import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.routes.analyse import router as analyse_router

app = FastAPI(title="InsuranceHelper API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyse_router)

# Mount the Next.js static export
frontend_out = ROOT / "frontend" / "out"
if frontend_out.exists():
    app.mount("/", StaticFiles(directory=str(frontend_out), html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {"status": "ok", "message": "Frontend not built yet. Run npm run build in frontend directory."}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "InsuranceHelper API"}