#!/usr/bin/env python3
"""Test which Sonnet inference profiles work."""

import boto3
import json
from botocore.exceptions import ClientError

def test_profile(profile_id):
    """Test if an inference profile can be invoked."""
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}]
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=profile_id,
            body=body
        )
        
        print(f"✅ {profile_id}")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"❌ {profile_id} - {error_code}")
        return False
    except Exception as e:
        print(f"❌ {profile_id} - {str(e)[:80]}")
        return False

print("Testing Sonnet Inference Profiles...\n")

profiles = [
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-sonnet-4-6",
    "anthropic.claude-3-sonnet-20240229-v1:0",
]

working = []
for profile in profiles:
    if test_profile(profile):
        working.append(profile)

print(f"\n{'='*60}")
if working:
    print(f"✅ Working profiles: {len(working)}")
    print(f"\nRecommended to use:")
    print(f"   {working[0]}")
    print(f"\nUpdate .env:")
    print(f"   BEDROCK_MODEL_ID={working[0]}")
else:
    print("❌ No working profiles found")
