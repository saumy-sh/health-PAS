"""
s3_utils.py
───────────
Shared S3 upload utility for Lambda deployment.
"""

import os
import boto3

S3_BUCKET = os.getenv("S3_BUCKET", "insurance-helper-outputs")

_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name="us-east-1")
    return _s3_client

def upload_to_s3(local_path: str, s3_key: str) -> str:
    """Upload a file to S3 and return a presigned URL valid for 1 hour."""
    client = get_s3_client()
    client.upload_file(local_path, S3_BUCKET, s3_key)
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600
    )
    return url