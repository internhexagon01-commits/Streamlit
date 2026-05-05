#!/usr/bin/env python3
"""
Test script to verify Bedrock access and find working models.
Run this to diagnose AWS credential and Bedrock permission issues.
"""

import boto3
import json
from botocore.exceptions import ClientError

def test_credentials():
    """Test if AWS credentials are valid."""
    print("=" * 60)
    print("1. Testing AWS Credentials...")
    print("=" * 60)
    try:
        sts = boto3.client('sts', region_name='us-east-1')
        identity = sts.get_caller_identity()
        print(f"✅ AWS credentials are valid!")
        print(f"   Account: {identity['Account']}")
        print(f"   User ARN: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"❌ AWS credentials error: {e}")
        return False

def test_bedrock_access():
    """Test if Bedrock service is accessible."""
    print("\n" + "=" * 60)
    print("2. Testing Bedrock Service Access...")
    print("=" * 60)
    try:
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        models = bedrock.list_foundation_models()
        claude_models = [m for m in models['modelSummaries'] if 'claude' in m['modelId'].lower()]
        print(f"✅ Bedrock service is accessible!")
        print(f"   Found {len(claude_models)} Claude models available")
        return True, claude_models
    except Exception as e:
        print(f"❌ Bedrock access error: {e}")
        return False, []

def test_model_invocation(model_id):
    """Test if a specific model can be invoked."""
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Prepare request body
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": "Hi"
                }
            ]
        })
        
        # Invoke model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        print(f"   ✅ {model_id}")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'UnrecognizedClientException':
            print(f"   ❌ {model_id} - Invalid credentials")
        elif error_code == 'AccessDeniedException':
            print(f"   ⚠️  {model_id} - No permission (need to request access)")
        elif error_code == 'ValidationException':
            print(f"   ⚠️  {model_id} - Model not available in region")
        else:
            print(f"   ❌ {model_id} - {error_code}")
        return False
    except Exception as e:
        print(f"   ❌ {model_id} - {str(e)[:50]}")
        return False

def main():
    print("\n🔍 Bedrock Access Diagnostic Tool\n")
    
    # Test 1: Credentials
    if not test_credentials():
        print("\n❌ Fix AWS credentials first:")
        print("   Run: aws configure")
        return
    
    # Test 2: Bedrock access
    has_access, claude_models = test_bedrock_access()
    if not has_access:
        print("\n❌ Cannot access Bedrock service")
        print("   Check IAM permissions for bedrock:ListFoundationModels")
        return
    
    # Test 3: Model invocation
    print("\n" + "=" * 60)
    print("3. Testing Model Invocation...")
    print("=" * 60)
    
    # Test recommended models
    recommended_models = [
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
    ]
    
    working_models = []
    for model_id in recommended_models:
        if test_model_invocation(model_id):
            working_models.append(model_id)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    if working_models:
        print(f"✅ Found {len(working_models)} working model(s)!")
        print("\nRecommended model to use:")
        print(f"   {working_models[0]}")
        print("\nUpdate your .env file:")
        print(f"   BEDROCK_MODEL_ID={working_models[0]}")
        
        if len(working_models) > 1:
            print("\nOther working models:")
            for model in working_models[1:]:
                print(f"   - {model}")
    else:
        print("❌ No working models found!")
        print("\nPossible issues:")
        print("   1. IAM user needs bedrock:InvokeModel permission")
        print("   2. Models need to be enabled in Bedrock console")
        print("   3. Region us-east-1 may not have model access")
        print("\nTo fix:")
        print("   1. Go to AWS Bedrock console: https://console.aws.amazon.com/bedrock/")
        print("   2. Navigate to 'Model access' in the left sidebar")
        print("   3. Click 'Manage model access'")
        print("   4. Enable Claude models")
        print("   5. Wait for approval (usually instant)")

if __name__ == "__main__":
    main()
