import boto3
import json
import sys
import uuid
import os
from botocore.config import Config
 
# ── Config (set these as environment variables) ───────────────────────
S3_BUCKET  = os.getenv("S3_BUCKET",  "naspocuser-s3")
REGION     = os.getenv("AWS_REGION", "ap-south-1")
ENDPOINT_ARN   = os.getenv("ENDPOINT_ARN ",  "arn:aws:bedrock-agentcore:ap-south-1:767398019214:runtime/atomicAquaLangGraph_Agent-ADIW4YEOjJ")
# ENDPOINT_ARN = f"{AGENT_ARN}/runtime-endpoint/DEFAULT"
# ── File ──────────────────────────────────────────────────────────────
# Usage: python input.py "D:\Customer_data\yourfile.log"
file_path  = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\charitha.akkinepally\Downloads\Jamming and Spoofing\Converted_On_20260324_T1120\ASCII\Jamming and Spoofing.ASCII"
filename   = os.path.basename(file_path)
session_id = f"session-{filename.replace('.', '-')}-{uuid.uuid4().hex[:10]}"
 
# ── Step 1: Upload file to S3 with multipart + progress bar ──────────
from boto3.s3.transfer import TransferConfig
 
s3 = boto3.client("s3", region_name=REGION)
s3_key = f"logs/{filename}"
file_size = os.path.getsize(file_path)
 
# Multipart config — 8MB chunks, 4 parallel threads
transfer_config = TransferConfig(
    multipart_threshold = 8 * 1024 * 1024,   # files > 8MB use multipart
    multipart_chunksize = 8 * 1024 * 1024,   # each part = 8MB
    max_concurrency     = 4,                  # 4 parallel threads
    use_threads         = True
)
 
# Progress tracker
uploaded = {"bytes": 0}
def progress(bytes_transferred):
    uploaded["bytes"] += bytes_transferred
    pct  = (uploaded["bytes"] / file_size) * 100
    done = int(pct / 2)
    bar  = "█" * done + "░" * (50 - done)
    mb_done  = uploaded["bytes"] / (1024 * 1024)
    mb_total = file_size / (1024 * 1024)
    print(f"\r  [{bar}] {pct:.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
 
print(f"Uploading {filename} ({file_size / (1024*1024):.1f} MB) to S3...")
s3.upload_file(
    file_path,
    S3_BUCKET,
    s3_key,
    Config=transfer_config,
    Callback=progress
)
print(f"\nUploaded to s3://{S3_BUCKET}/{s3_key} ✅")
 
# ── Step 2: Send only S3 path to agent (small payload) ───────────────
payload = {
    "s3_key":     s3_key,
    "filename":   filename,
    "session_id": session_id
}
 
# ── Step 3: Invoke agent ──────────────────────────────────────────────
 
# FIX 1: Add timeout config — default 60s is too short for large file processing
agent_config = Config(
    read_timeout=600,          # 10 minutes
    connect_timeout=30,
    retries={"max_attempts": 0}  # no retries on timeout — avoids double wait
)
 
print("Invoking agent...")
client = boto3.client(
    "bedrock-agentcore",
    region_name=REGION,
    config=agent_config
)
 
response = client.invoke_agent_runtime(
    agentRuntimeArn=ENDPOINT_ARN,   # full endpoint ARN, no qualifier param

    payload=json.dumps(payload).encode("utf-8")
)
# FIX 3: correct response key is "payload" not "response"
result = json.loads(response["payload"].read())
# result = json.loads(response["response"].read())
 
print("\nResult:", result)
print(f"\nSession ID: {session_id}")
print("Use this session ID to ask questions about this log file!")