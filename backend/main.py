import boto3
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from backend.routes.analyse import router as analyse_router

app = FastAPI(
    title="InsuranceHelper API",
    description="Pre-authorization pipeline backend",
    version="1.0.0",
)

handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyse_router)

# S3 bucket for outputs
S3_BUCKET = "insurance-helper-outputs"
s3_client = boto3.client("s3", region_name="us-east-1")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "InsuranceHelper API"}

def upload_to_s3(local_path: str, s3_key: str) -> str:
    """Upload a file to S3 and return a presigned URL valid for 1 hour."""
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600
    )
    return url