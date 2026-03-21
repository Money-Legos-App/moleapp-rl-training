#!/usr/bin/env bash
# RunPod Instance Setup — Shield Sweep
# ======================================
# Run this script on a fresh RunPod instance (RTX 4090 / A100).
#
# Usage:
#   bash runpod/setup.sh
#
# Prerequisites:
#   - WANDB_API_KEY in environment (or will prompt)
#   - R2 credentials in environment (for episode pull)

set -euo pipefail

echo "=== MoleApp RL Training — RunPod Setup ==="

# ── 1. Install dependencies ──────────────────────────────────────────
echo "[1/5] Installing Python dependencies..."
pip install -e ".[dev]" --quiet

# ── 2. Authenticate W&B ─────────────────────────────────────────────
echo "[2/5] Authenticating W&B..."
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY not set. Run: wandb login"
    wandb login
else
    echo "WANDB_API_KEY found in environment"
fi

# ── 3. Pull data from R2 ──────────────────────────────────────────
echo "[3/5] Pulling data from R2..."
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

# ── 4. Verify episode data exists ────────────────────────────────────
echo "[4/5] Checking episode data..."
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

# ── 5. Run tests ────────────────────────────────────────────────────
echo "[5/5] Running tests..."
python -m pytest tests/ -v --tb=short

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Create sweep:  wandb sweep training/configs/sweep_shield.yaml"
echo "  2. Launch agent:  wandb agent dapps4africa/MoleApp-RL/<SWEEP_ID>"
echo "  3. Multi-agent:   wandb agent --count 10 dapps4africa/MoleApp-RL/<SWEEP_ID>"
echo ""
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU detected (CPU-only mode)"
