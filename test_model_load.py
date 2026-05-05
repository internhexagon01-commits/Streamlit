#!/usr/bin/env python3
"""Test if the model loads correctly with the right credentials."""

from dotenv import load_dotenv
load_dotenv()

import os
print(f"BEDROCK_MODEL_ID from env: {os.getenv('BEDROCK_MODEL_ID', 'NOT SET')}")

try:
    from src.model.load import load_model
    print("\n[TEST] Loading model...")
    
    model = load_model()
    print(f"[TEST] ✅ Model loaded successfully!")
    print(f"[TEST] Model ID: {model.model_id}")
    
    # Try a simple invocation
    print("\n[TEST] Testing model invocation...")
    response = model.invoke("Say 'Hello' in one word")
    print(f"[TEST] ✅ Model response: {response.content}")
    print("\n[TEST] 🎉 Everything works!")
    
except Exception as e:
    print(f"\n[TEST] ❌ Error: {e}")
    import traceback
    traceback.print_exc()
