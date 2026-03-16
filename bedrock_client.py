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
from dotenv import load_dotenv

# Load .env file
load_dotenv()




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
            aws_session_token=os.getenv("AWS_SESSION_TOKEN")
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
    
    # Check for output and text content
    output = result.get("output", {})
    message = output.get("message", {})
    content = message.get("content", [])
    
    if content and "text" in content[0]:
        text = content[0]["text"]
        
        # Bedrock's Nova sometimes returns a refusal message in the text if blocked
        if "blocked by our content filters" in text.lower():
            return f"ERROR: Content Filter Blocked. The model refused to process this image. Raw response: {text}"
            
        return text
        
    # Check for stopReason if text is missing
    stop_reason = result.get("stopReason")
    if stop_reason == "content_filtered":
        return "ERROR: Content Filter Blocked. (stopReason: content_filtered)"
        
    return f"ERROR: No text output from model. stopReason: {stop_reason}. Full result: {json.dumps(result)}"
