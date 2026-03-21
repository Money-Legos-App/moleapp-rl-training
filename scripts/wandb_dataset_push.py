"""
W&B Dataset Push — Upload versioned episode datasets to Weights & Biases
=========================================================================
After EpisodeBuilder produces the final 47-dim feature set, this script
pushes a locked, versioned artifact to W&B so RunPod training scripts
always train on a known dataset.

Artifact types:
  - "rl-episodes-v{version}" — contains per-asset market_data.npy + features.pkl
  - Tagged with feature version hash for skew detection

Usage:
    # Build episodes then push
    python scripts/wandb_dataset_push.py --data-dir data/episodes --version 1.0.0

    # Pull dataset on RunPod before training
    python scripts/wandb_dataset_push.py pull --version 1.0.0 --output data/episodes
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import wandb
from dotenv import load_dotenv

from data.preprocessors.feature_engineer import FEATURE_HASH, FEATURE_VERSION

load_dotenv()
logger = logging.getLogger(__name__)

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "moleapp-rl")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)  # Optional: W&B team name
ARTIFACT_TYPE = "dataset"
ARTIFACT_BASE_NAME = "rl-episodes"


def push_episodes(
    data_dir: str = "data/episodes",
    version: str = "1.0.0",
    description: str = "",
) -> str:
    """
    Push episode data to W&B as a versioned artifact.

    Expected directory structure:
        data/episodes/
        ├── BTC/
        │   ├── market_data.npy
        │   └── features.pkl
        ├── ETH/
        │   ├── market_data.npy
        │   └── features.pkl
        └── ...

    Args:
        data_dir: Path to built episode directory
        version: Semantic version string (e.g., "1.0.0")
        description: Optional description of this dataset version

    Returns:
        Artifact full name (e.g., "moleapp-rl/rl-episodes:v1.0.0")
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Episode directory not found: {data_path}")

    # Count assets
    asset_dirs = [d for d in data_path.iterdir() if d.is_dir() and (d / "market_data.npy").exists()]
    if not asset_dirs:
        raise ValueError(f"No asset episode directories found in {data_path}")

    logger.info(f"Found {len(asset_dirs)} assets: {[d.name for d in sorted(asset_dirs)]}")

    # Initialize W&B run
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="dataset-upload",
        tags=["dataset", f"feature-v{FEATURE_VERSION}", f"hash-{FEATURE_HASH}"],
    )

    # Create artifact (W&B handles versioning via :v0, :v1, :latest)
    artifact = wandb.Artifact(
        name=ARTIFACT_BASE_NAME,
        type=ARTIFACT_TYPE,
        description=description or f"RL episode dataset v{version} (feature hash: {FEATURE_HASH})",
        metadata={
            "feature_version": FEATURE_VERSION,
            "feature_hash": FEATURE_HASH,
            "num_assets": len(asset_dirs),
            "assets": sorted([d.name for d in asset_dirs]),
            "dataset_version": version,
            "r2_bucket": os.getenv("R2_BUCKET_NAME", "moleapp-rl-data"),
            "r2_processed_prefix": "processed/1h/",
            "data_source": "binance_vision",
        },
    )

    # Add all asset directories
    total_size = 0
    for asset_dir in sorted(asset_dirs):
        for f in asset_dir.iterdir():
            if f.suffix in (".npy", ".pkl"):
                artifact.add_file(str(f), name=f"{asset_dir.name}/{f.name}")
                total_size += f.stat().st_size

    logger.info(f"Artifact size: {total_size / (1024*1024):.1f} MB across {len(asset_dirs)} assets")

    # Log and push
    run.log_artifact(artifact)
    run.finish()

    full_name = f"{WANDB_PROJECT}/{ARTIFACT_BASE_NAME}"
    logger.info(f"Pushed artifact: {full_name} (version {version})")
    logger.info(f"Feature hash: {FEATURE_HASH} (version {FEATURE_VERSION})")

    return full_name


def pull_episodes(
    version: str = "latest",
    output_dir: str = "data/episodes",
) -> Path:
    """
    Pull episode dataset from W&B.

    Args:
        version: Artifact version ("latest", "v0", "v1", or "1.0.0")
        output_dir: Where to download the dataset

    Returns:
        Path to downloaded dataset directory
    """
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="dataset-download",
    )

    # W&B artifact refs: "name:latest", "name:v0", "name:v1", etc.
    artifact_ref = f"{ARTIFACT_BASE_NAME}:{version}"
    artifact = run.use_artifact(artifact_ref)
    logger.info(f"Resolved artifact: {artifact.name} (version {artifact.version})")

    # Check feature hash compatibility
    remote_hash = artifact.metadata.get("feature_hash", "")
    if remote_hash and remote_hash != FEATURE_HASH:
        logger.warning(
            f"FEATURE SKEW WARNING: Remote hash={remote_hash}, "
            f"local hash={FEATURE_HASH}. Dataset may be incompatible!"
        )

    output_path = Path(output_dir)
    download_path = artifact.download(root=str(output_path))

    run.finish()

    logger.info(f"Downloaded dataset to {download_path}")
    logger.info(f"Remote feature version: {artifact.metadata.get('feature_version', 'unknown')}")
    logger.info(f"Assets: {artifact.metadata.get('assets', [])}")

    return Path(download_path)


def list_artifacts() -> list[dict]:
    """List all dataset artifacts in the W&B project."""
    api = wandb.Api()
    project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT

    try:
        collections = api.artifact_type(ARTIFACT_TYPE, project=project_path).collections()
    except Exception as e:
        logger.error(f"Failed to list artifacts: {e}")
        return []

    results = []
    for collection in collections:
        if ARTIFACT_BASE_NAME in collection.name:
            for version in collection.versions():
                info = {
                    "name": version.name,
                    "version": version.version,
                    "created_at": version.created_at,
                    "size_mb": version.size / (1024 * 1024) if version.size else 0,
                    "feature_hash": version.metadata.get("feature_hash", "?"),
                    "assets": version.metadata.get("num_assets", "?"),
                }
                results.append(info)
                print(
                    f"  {version.name:40s}  "
                    f"{info['size_mb']:8.1f} MB  "
                    f"hash={info['feature_hash']}  "
                    f"assets={info['assets']}  "
                    f"{version.created_at}"
                )

    if not results:
        print("No dataset artifacts found.")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="W&B dataset artifact manager")
    sub = parser.add_subparsers(dest="action", required=True)

    # Push
    push_p = sub.add_parser("push", help="Push episode dataset to W&B")
    push_p.add_argument("--data-dir", default="data/episodes")
    push_p.add_argument("--version", required=True, help="Dataset version (e.g., 1.0.0)")
    push_p.add_argument("--description", default="", help="Version description")

    # Pull
    pull_p = sub.add_parser("pull", help="Pull episode dataset from W&B")
    pull_p.add_argument("--version", default="latest", help="Version to pull")
    pull_p.add_argument("--output", default="data/episodes")

    # List
    sub.add_parser("list", help="List available dataset artifacts")

    args = parser.parse_args()

    if args.action == "push":
        push_episodes(data_dir=args.data_dir, version=args.version, description=args.description)
    elif args.action == "pull":
        pull_episodes(version=args.version, output_dir=args.output)
    elif args.action == "list":
        list_artifacts()
