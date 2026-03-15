"""
bedrock_client.py
─────────────────
Shared AWS Bedrock client + model IDs.
Loads credentials from a .env file — no temporary tokens needed.

Setup:
    1. pip install python-dotenv
    2. Copy .env.template to .env and fill in your sandbox keys
"""

import os
import json
import boto3
from pathlib import Path
from dotenv import load_dotenv   # pip install python-dotenv

# ── Load .env from the same directory as this file ───────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
    print(f"[bedrock_client] Loaded credentials from {_env_path}")
else:
    load_dotenv()   # fallback: OS environment variables
    print("[bedrock_client] .env not found — falling back to OS environment")

# ── Validate required vars are present ───────────────────────────────────────
_missing = [v for v in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
            if not os.getenv(v)]
if _missing:
    raise EnvironmentError(
        f"\n❌  Missing AWS credentials: {_missing}\n"
        "    Steps to fix:\n"
        "      1. Copy .env.template  →  .env\n"
        "      2. Fill in AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
        "      3. Re-run your script\n"
    )

# ── Config ────────────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# ── Model IDs (confirmed available in sandbox) ────────────────────────────────
PRO_MODEL_ID     = "amazon.nova-pro-v1:0"       # multimodal – Agents 1, 5, 9
LITE_MODEL_ID    = "amazon.nova-lite-v1:0"       # multimodal – Agents 3, 6, 7
MICRO_MODEL_ID   = "amazon.nova-micro-v1:0"      # text-only  – Agents 2, 4, 8
PREMIER_MODEL_ID = "amazon.nova-premier-v1:0"    # most capable – optional upgrade

# ── Singleton client ─────────────────────────────────────────────────────────
_client = None

def get_client():
    """
    Return a cached Bedrock Runtime boto3 client.
    Credentials are read explicitly from environment variables loaded via .env.
    This avoids using any cached temporary session tokens.
    """
    global _client
    if _client is None:
        _client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            # Permanent IAM user keys (from .env) never expire.
            # If your sandbox only gives you temporary keys, also add:
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            # and refresh them in your .env when they expire.
        )
    return _client


def invoke(model_id: str, messages: list, system: list = None,
           max_tokens: int = 1500, temperature: float = 0.1) -> str:
    """
    Thin wrapper around bedrock invoke_model.
    Returns the text string from the first content block.
    """
    body = {
        "messages": messages,
        "inferenceConfig": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 20,
        },
    }
    if system:
        body["system"] = system

    response = get_client().invoke_model(
        modelId=model_id,
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]
