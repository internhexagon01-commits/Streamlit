#!/usr/bin/env python3
"""Diagnose which AWS credentials boto3 is actually using."""

import boto3
import json

print("="*60)
print("AWS Credentials Diagnostic")
print("="*60)

# Check what boto3 sees
try:
    sts = boto3.client('sts', region_name='us-east-1')
    identity = sts.get_caller_identity()
    print("\n✅ Boto3 can access AWS")
    print(f"   Account: {identity['Account']}")
    print(f"   User ARN: {identity['Arn']}")
    print(f"   User ID: {identity['UserId']}")
except Exception as e:
    print(f"\n❌ Boto3 cannot access AWS: {e}")
    exit(1)

# Check credentials source
session = boto3.Session()
credentials = session.get_credentials()
print(f"\n📍 Credentials source: {credentials.method}")
print(f"   Access Key ID: {credentials.access_key[:10]}...")

# Try to invoke Bedrock directly
print("\n" + "="*60)
print("Testing Bedrock Access")
print("="*60)

try:
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hi"}]
    })
    
    print("\n[TEST] Trying: us.anthropic.claude-sonnet-4-6")
    response = bedrock_runtime.invoke_model(
        modelId="us.anthropic.claude-sonnet-4-6",
        body=body
    )
    
    result = json.loads(response['body'].read())
    print(f"✅ SUCCESS! Model responded: {result['content'][0]['text']}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    
    # Try with explicit credentials
    print("\n[TEST] Trying with explicit credentials from environment...")
    import os
    
    # Check if AWS env vars are set
    if os.getenv('AWS_ACCESS_KEY_ID'):
        print(f"   AWS_ACCESS_KEY_ID: {os.getenv('AWS_ACCESS_KEY_ID')[:10]}...")
        print(f"   AWS_SECRET_ACCESS_KEY: {'*' * 20}")
    else:
        print("   No AWS environment variables found")
    
    print("\n💡 Possible issues:")
    print("   1. Your AWS credentials might be expired")
    print("   2. Bedrock might not be enabled in us-east-1")
    print("   3. IAM permissions might not be propagated yet")
    print("\n   Try running: aws configure")
    print("   And re-enter your credentials")
