"""
Cloudflare R2 Sync — Upload/download raw data to R2 bucket
============================================================
R2 is used for raw OHLCV, funding, and OI parquet dumps.
S3-compatible API via boto3.

Usage:
    # Upload all raw data
    python scripts/r2_sync.py upload --data-dir data/datasets

    # Download raw data (e.g., on RunPod before building episodes)
    python scripts/r2_sync.py download --data-dir data/datasets

    # List files in bucket
    python scripts/r2_sync.py list
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# R2 config from env
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "https://9507330fe5a8c228ea49f6e5c6c6b659.r2.cloudflarestorage.com")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "moleapp-rl-data")
R2_PREFIX = "raw-data/"  # All raw parquet files go under this prefix


def get_r2_client():
    """Create boto3 S3 client for Cloudflare R2."""
    if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        raise ValueError(
            "R2 credentials not set. Set R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY in .env"
        )

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
        region_name="auto",
    )


def upload_data(data_dir: str = "data/datasets", prefix: str = R2_PREFIX) -> int:
    """
    Upload all .parquet files from data_dir to R2.

    Returns number of files uploaded.
    """
    client = get_r2_client()
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return 0

    files = list(data_path.glob("*.parquet"))
    if not files:
        logger.warning(f"No parquet files found in {data_path}")
        return 0

    uploaded = 0
    for file_path in sorted(files):
        key = f"{prefix}{file_path.name}"
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        logger.info(f"Uploading {file_path.name} ({file_size_mb:.1f} MB) → s3://{R2_BUCKET_NAME}/{key}")

        try:
            client.upload_file(
                str(file_path),
                R2_BUCKET_NAME,
                key,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )
            uploaded += 1
        except Exception as e:
            logger.error(f"Failed to upload {file_path.name}: {e}")

    logger.info(f"Uploaded {uploaded}/{len(files)} files to R2")
    return uploaded


def download_data(data_dir: str = "data/datasets", prefix: str = R2_PREFIX) -> int:
    """
    Download all parquet files from R2 to local data_dir.

    Returns number of files downloaded.
    """
    client = get_r2_client()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # List objects under prefix
    try:
        response = client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=prefix)
    except Exception as e:
        logger.error(f"Failed to list R2 objects: {e}")
        return 0

    contents = response.get("Contents", [])
    if not contents:
        logger.warning(f"No files found in s3://{R2_BUCKET_NAME}/{prefix}")
        return 0

    downloaded = 0
    for obj in contents:
        key = obj["Key"]
        filename = key.removeprefix(prefix)
        if not filename or not filename.endswith(".parquet"):
            continue

        local_path = data_path / filename
        size_mb = obj["Size"] / (1024 * 1024)

        logger.info(f"Downloading {filename} ({size_mb:.1f} MB)")

        try:
            client.download_file(R2_BUCKET_NAME, key, str(local_path))
            downloaded += 1
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")

    logger.info(f"Downloaded {downloaded} files to {data_path}")
    return downloaded


def list_files(prefix: str = R2_PREFIX) -> list[dict]:
    """List all files in R2 bucket under prefix."""
    client = get_r2_client()

    try:
        response = client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=prefix)
    except Exception as e:
        logger.error(f"Failed to list R2 objects: {e}")
        return []

    contents = response.get("Contents", [])
    files = []
    total_size = 0

    for obj in sorted(contents, key=lambda x: x["Key"]):
        size_mb = obj["Size"] / (1024 * 1024)
        total_size += obj["Size"]
        files.append({
            "key": obj["Key"],
            "size_mb": round(size_mb, 2),
            "last_modified": obj["LastModified"].isoformat(),
        })
        print(f"  {obj['Key']:60s}  {size_mb:8.2f} MB  {obj['LastModified']:%Y-%m-%d %H:%M}")

    print(f"\nTotal: {len(files)} files, {total_size / (1024*1024):.1f} MB")
    return files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Cloudflare R2 data sync")
    parser.add_argument("action", choices=["upload", "download", "list"])
    parser.add_argument("--data-dir", default="data/datasets")
    args = parser.parse_args()

    if args.action == "upload":
        upload_data(data_dir=args.data_dir)
    elif args.action == "download":
        download_data(data_dir=args.data_dir)
    elif args.action == "list":
        list_files()
