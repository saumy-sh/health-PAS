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
            region_name=AWS_REGION
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
