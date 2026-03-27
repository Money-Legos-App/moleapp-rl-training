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

# ── 0. Python version check ───────────────────────────────────────────
echo "[0/6] Checking Python version..."
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo "ERROR: Python >= 3.11 required, found $PYTHON_VERSION"
    echo "  RunPod template must use Python 3.11+."
    echo "  Try: runpod/pytorch:2.2.0-py3.11-cuda12.1.0-devel-ubuntu22.04"
    exit 1
fi
echo "  Python $PYTHON_VERSION OK"

# ── 1. Install dependencies ──────────────────────────────────────────
echo "[1/6] Installing Python dependencies..."
pip install -e ".[dev]" --quiet

# ── 2. Verify Ray + GPU ──────────────────────────────────────────────
echo "[2/6] Verifying Ray + GPU..."
python -c "
import ray, torch
ray.init()
resources = ray.cluster_resources()
ray_gpus = resources.get('GPU', 0)
ray_cpus = resources.get('CPU', 0)
torch_gpus = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if torch_gpus > 0 else 'none'
print(f'  Ray: {ray_cpus:.0f} CPUs, {ray_gpus:.0f} GPUs')
print(f'  PyTorch: {torch_gpus} GPU(s) — {gpu_name}')
if torch_gpus == 0:
    print('  WARNING: No GPU detected. Training will use CPU only (slow).')
ray.shutdown()
"

# ── 3. Authenticate W&B ─────────────────────────────────────────────
echo "[3/6] Authenticating W&B..."
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY not set. Run: wandb login"
    wandb login
else
    echo "  WANDB_API_KEY found in environment"
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
    echo "  Found episodes for $EPISODE_COUNT assets"
fi

# ── 6. Run tests ────────────────────────────────────────────────────
echo "[6/6] Running tests (skipping slow PPO tests)..."
python -m pytest tests/ -v --tb=short -m "not slow"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  No GPU detected (CPU-only mode)"
echo ""
echo "Training commands (GPU auto-detected, all 15 assets by default):"
echo "  Shield:    python -m training.train --profile shield  --config training/configs/shield_config.yaml"
echo "  Builder:   python -m training.train --profile builder --config training/configs/builder_config.yaml"
echo ""
echo "2-Strategy System: Shield (Flexible USD Vault) + Builder (High-Yield Engine)"
echo ""
echo "Train on specific assets only:"
echo "  python -m training.train --profile shield --config training/configs/shield_config.yaml --assets BTC ETH SOL"
echo ""
echo "Resume from checkpoint:"
echo "  python -m training.train --profile shield --config training/configs/shield_config.yaml --resume models/shield/checkpoints/checkpoint_000050"
