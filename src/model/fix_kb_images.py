"""
fix_kb_images.py
────────────────
Converts all PNG images in your Bedrock KB S3 prefix to searchable .txt files,
then re-uploads them so Bedrock KB can index them as text.

Run this ONCE locally before re-syncing your Bedrock Knowledge Base.

Requirements:
    pip install boto3 pillow pytesseract
    # Also install Tesseract OCR binary:
    #   Windows: https://github.com/UB-Mannheim/tesseract/wiki
    #   Linux:   sudo apt install tesseract-ocr
    #   Mac:     brew install tesseract
"""

import boto3
import io
import os
from PIL import Image
import pytesseract

# ── CONFIG — update these to match your setup ─────────────────────────
BUCKET          = "naspocuser-s3"
KB_PREFIX       = "aws/bedrock/knowledge_bases/SIUS5NYQGD/XDN5MO5DQ9/"
OUTPUT_PREFIX   = "aws/bedrock/knowledge_bases/SIUS5NYQGD/text_docs/"   # new prefix for text files
REGION          = "ap-south-1"

# Windows only — set path to tesseract.exe if not in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── S3 client ─────────────────────────────────────────────────────────
s3 = boto3.client("s3", region_name=REGION)


def list_all_images(bucket: str, prefix: str):
    """List all PNG/JPG objects under the given S3 prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".png", ".jpg", ".jpeg")):
                yield key


def ocr_image_from_s3(bucket: str, key: str) -> str:
    """Download an image from S3 and return its OCR text."""
    response = s3.get_object(Bucket=bucket, Key=key)
    image_bytes = response["Body"].read()
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed (handles RGBA PNGs)
    if image.mode != "RGB":
        image = image.convert("RGB")

    text = pytesseract.image_to_string(image, config="--psm 6")
    return text.strip()


def upload_text_to_s3(bucket: str, key: str, text: str):
    """Upload extracted text as a .txt file to S3."""
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="text/plain"
    )


def main():
    images = list(list_all_images(BUCKET, KB_PREFIX))
    total  = len(images)

    if total == 0:
        print(f"No images found under s3://{BUCKET}/{KB_PREFIX}")
        return

    print(f"Found {total} image(s) to convert.\n")

    success = 0
    skipped = 0
    failed  = 0

    for i, img_key in enumerate(images, 1):
        # Derive output text file key
        filename      = os.path.basename(img_key)
        name_no_ext   = os.path.splitext(filename)[0]
        txt_key       = OUTPUT_PREFIX + name_no_ext + ".txt"

        print(f"[{i}/{total}] Processing: {img_key}")

        try:
            text = ocr_image_from_s3(BUCKET, img_key)

            if not text or len(text) < 20:
                print(f"  ⚠  Skipped — OCR returned almost no text (likely a diagram/image with no text).")
                skipped += 1
                continue

            upload_text_to_s3(BUCKET, txt_key, text)
            print(f"  ✓  Uploaded text ({len(text)} chars) → s3://{BUCKET}/{txt_key}")
            success += 1

        except Exception as e:
            print(f"  ✗  Failed: {e}")
            failed += 1

    print(f"\n{'─'*60}")
    print(f"Done. {success} converted, {skipped} skipped (no text), {failed} failed.")
    print(f"\nNext steps:")
    print(f"  1. Go to AWS Bedrock Console → Knowledge Bases → {os.environ.get('KB_ID', 'SIUS5NYQGD')}")
    print(f"  2. Edit your data source to point to the NEW prefix:")
    print(f"     s3://{BUCKET}/{OUTPUT_PREFIX}")
    print(f"  3. Click 'Sync' to re-index the text files.")
    print(f"  4. Re-deploy your agent.")


if __name__ == "__main__":
    main()