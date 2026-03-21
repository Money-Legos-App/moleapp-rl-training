"""
Cloudflare R2 Sync — Upload/download data to R2 bucket
========================================================
R2 stores parquets (16 MB) and pre-built episodes (121 MB) for RunPod.
S3-compatible API via boto3.

Usage:
    # Upload parquets
    python scripts/r2_sync.py upload --prefix processed --data-dir data/datasets

    # Upload episodes (recursive — handles BTC/market_data.npy, BTC/features.pkl, etc.)
    python scripts/r2_sync.py upload --prefix episodes --data-dir data/episodes --recursive

    # Download everything to RunPod
    python scripts/r2_sync.py download --prefix processed --data-dir data/datasets
    python scripts/r2_sync.py download --prefix episodes --data-dir data/episodes

    # List files in bucket
    python scripts/r2_sync.py list --prefix episodes
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

# Bucket prefix structure
R2_PREFIXES = {
    "raw": "raw/",               # Raw Binance Vision CSV backups
    "processed": "processed/1h/", # Validated, training-ready parquets
    "episodes": "episodes/",      # Pre-built episodes (market_data.npy + features.pkl per asset)
    "sweeps": "sweeps/",          # Future: 6-month slices for W&B sweeps
    "legacy": "raw-data/",        # Legacy HL dumps (backward compat)
}
R2_PREFIX = R2_PREFIXES["legacy"]  # Default for backward compat


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


def upload_data(
    data_dir: str = "data/datasets",
    prefix: str = R2_PREFIX,
    file_glob: str = "*.parquet",
    recursive: bool = False,
) -> int:
    """
    Upload files from data_dir to R2.

    When recursive=True, walks the entire directory tree and preserves
    relative paths (e.g. data/episodes/BTC/market_data.npy → episodes/BTC/market_data.npy).

    Returns number of files uploaded.
    """
    client = get_r2_client()
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return 0

    if recursive:
        files = [f for f in data_path.rglob("*") if f.is_file()]
    else:
        files = list(data_path.glob(file_glob))

    if not files:
        logger.warning(f"No files found in {data_path}")
        return 0

    uploaded = 0
    total_size = 0
    for file_path in sorted(files):
        rel_path = file_path.relative_to(data_path)
        key = f"{prefix}{rel_path}"
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        total_size += file_path.stat().st_size

        logger.info(f"Uploading {rel_path} ({file_size_mb:.1f} MB) → s3://{R2_BUCKET_NAME}/{key}")

        try:
            client.upload_file(
                str(file_path),
                R2_BUCKET_NAME,
                key,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )
            uploaded += 1
        except Exception as e:
            logger.error(f"Failed to upload {rel_path}: {e}")

    logger.info(f"Uploaded {uploaded}/{len(files)} files ({total_size / (1024*1024):.1f} MB) to R2")
    return uploaded


def download_data(data_dir: str = "data/datasets", prefix: str = R2_PREFIX) -> int:
    """
    Download all files from R2 prefix to local data_dir.

    Preserves subdirectory structure (e.g. episodes/BTC/market_data.npy
    → data/episodes/BTC/market_data.npy).

    Returns number of files downloaded.
    """
    client = get_r2_client()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Paginate through all objects under prefix
    all_contents = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": R2_BUCKET_NAME, "Prefix": prefix, "MaxKeys": 1000}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        try:
            response = client.list_objects_v2(**kwargs)
        except Exception as e:
            logger.error(f"Failed to list R2 objects: {e}")
            return 0

        all_contents.extend(response.get("Contents", []))
        if not response.get("IsTruncated"):
            break
        continuation_token = response["NextContinuationToken"]

    if not all_contents:
        logger.warning(f"No files found in s3://{R2_BUCKET_NAME}/{prefix}")
        return 0

    downloaded = 0
    total_size = 0
    for obj in all_contents:
        key = obj["Key"]
        rel_path = key.removeprefix(prefix)
        if not rel_path:
            continue

        local_path = data_path / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        size_mb = obj["Size"] / (1024 * 1024)
        total_size += obj["Size"]

        logger.info(f"Downloading {rel_path} ({size_mb:.1f} MB)")

        try:
            client.download_file(R2_BUCKET_NAME, key, str(local_path))
            downloaded += 1
        except Exception as e:
            logger.error(f"Failed to download {rel_path}: {e}")

    logger.info(f"Downloaded {downloaded} files ({total_size / (1024*1024):.1f} MB) to {data_path}")
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
    parser.add_argument(
        "--prefix", default="legacy",
        choices=list(R2_PREFIXES.keys()),
        help="R2 bucket prefix (default: legacy)",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Upload all files recursively (for episodes with subdirectories)",
    )
    args = parser.parse_args()

    prefix = R2_PREFIXES[args.prefix]

    if args.action == "upload":
        upload_data(data_dir=args.data_dir, prefix=prefix, recursive=args.recursive)
    elif args.action == "download":
        download_data(data_dir=args.data_dir, prefix=prefix)
    elif args.action == "list":
        list_files(prefix=prefix)
