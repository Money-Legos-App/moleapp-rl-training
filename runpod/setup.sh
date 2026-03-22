#!/usr/bin/env bash
# RunPod Instance Setup — RLlib Training
# ========================================
# Run this script on a fresh RunPod instance (RTX 4090 / A100).
#
# Usage:
#   bash runpod/setup.sh
#
# Prerequisites:
#   - WANDB_API_KEY in environment (or will prompt)
#   - R2 credentials in environment (for episode pull)

set -euo pipefail

echo "=== MoleApp RL Training — RunPod Setup (RLlib) ==="

# ── 1. Install dependencies ──────────────────────────────────────────
echo "[1/6] Installing Python dependencies..."
pip install -e ".[dev]" --quiet

# ── 2. Verify Ray + GPU ──────────────────────────────────────────────
echo "[2/6] Verifying Ray installation..."
python -c "
import ray
ray.init()
resources = ray.cluster_resources()
num_gpus = resources.get('GPU', 0)
num_cpus = resources.get('CPU', 0)
print(f'  Ray initialized: {num_cpus:.0f} CPUs, {num_gpus:.0f} GPUs')
ray.shutdown()
"

# ── 3. Authenticate W&B ─────────────────────────────────────────────
echo "[3/6] Authenticating W&B..."
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY not set. Run: wandb login"
    wandb login
else
    echo "WANDB_API_KEY found in environment"
fi

# ── 4. Pull data from R2 ──────────────────────────────────────────
echo "[4/6] Pulling data from R2..."
if [ -z "${R2_ACCESS_KEY_ID:-}" ]; then
    echo "WARNING: R2 credentials not set. Skipping R2 pull."
    echo "  Set R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
    echo "  Or manually copy data/episodes/ and data/datasets/ to this instance."
else
    echo "  Pulling parquets..."
    python scripts/r2_sync.py download --prefix processed --data-dir data/datasets || echo "  Parquet pull failed"

    echo "  Pulling episodes..."
    python scripts/r2_sync.py download --prefix episodes --data-dir data/episodes || echo "  Episode pull failed"
fi

# ── 5. Verify episode data exists ────────────────────────────────────
echo "[5/6] Checking episode data..."
EPISODE_COUNT=$(ls -d data/episodes/*/market_data.npy 2>/dev/null | wc -l)
if [ "$EPISODE_COUNT" -lt 1 ]; then
    echo "No episodes found. Building from parquets..."
    python -c "
from data.preprocessors.episode_builder import build_all_assets
results = build_all_assets()
print(f'Built episodes for {len(results)} assets')
"
else
    echo "Found episodes for $EPISODE_COUNT assets"
fi

# ── 6. Run tests ────────────────────────────────────────────────────
echo "[6/6] Running tests..."
python -m pytest tests/ -v --tb=short

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Train Shield:    python -m training.train --profile shield --config training/configs/shield_config.yaml"
echo "  2. Train Builder:   python -m training.train --profile builder --config training/configs/builder_config.yaml"
echo "  3. Train Hunter:    python -m training.train --profile hunter --config training/configs/hunter_config.yaml"
echo "  4. HP Sweep:        python -m training.tune_sweep"
echo "  5. Export ONNX:     python -c \"from serving.model_registry import export_to_onnx; ...\""
echo ""
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU detected (CPU-only mode)"
