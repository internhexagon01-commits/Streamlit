#!/usr/bin/env python3
"""Test LangChain ChatBedrock without Streamlit interference."""

from dotenv import load_dotenv
load_dotenv()

import os
import boto3
from langchain_aws import ChatBedrock

model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
print(f"Model ID: {model_id}")

# Create boto3 client
print("\n[TEST] Creating boto3 bedrock-runtime client...")
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Get credentials info
session = boto3.Session()
credentials = session.get_credentials()
print(f"Credentials source: {credentials.method}")
print(f"Access Key: {credentials.access_key[:10]}...")

# Create LangChain model with explicit client
print(f"\n[TEST] Creating ChatBedrock with model: {model_id}")
model = ChatBedrock(
    model_id=model_id,
    client=bedrock_client,
    model_kwargs={
        "temperature": 0.0,
        "max_tokens": 20,
    },
    streaming=False,
)

print(f"[TEST] Model created successfully")
print(f"[TEST] Model ID: {model.model_id}")

# Try to invoke
print(f"\n[TEST] Invoking model...")
try:
    response = model.invoke("Say hello in one word")
    print(f"✅ SUCCESS! Response: {response.content}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
