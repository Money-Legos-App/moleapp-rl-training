# RunPod Training Guide — MoleApp Alpha RL Agents

Step-by-step guide to provision a RunPod GPU instance, clone the repo, and train Shield / Builder models on all 15 assets.

**2-Strategy System:**
- **Shield** — "Flexible USD Vault" (1x leverage, 25% max position, never pay funding)
- **Builder** — "High-Yield Engine" (2x leverage, 50% max position, block funding >0.03%)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Create a RunPod GPU Pod](#2-create-a-runpod-gpu-pod)
3. [Connect to Your Pod](#3-connect-to-your-pod)
4. [Clone & Setup](#4-clone--setup)
5. [Launch Training](#5-launch-training)
6. [Monitor Training](#6-monitor-training)
7. [Export ONNX Models](#7-export-onnx-models)
8. [Upload Models to R2](#8-upload-models-to-r2)
9. [Costs & GPU Recommendations](#9-costs--gpu-recommendations)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Before you start, gather these credentials:

| Credential | Where to get it |
|---|---|
| **GitHub SSH key** or **PAT** | [github.com/settings/keys](https://github.com/settings/keys) — repo is public, but SSH is faster |
| **WANDB_API_KEY** | [wandb.ai/authorize](https://wandb.ai/authorize) |
| **R2_ACCESS_KEY_ID** | Cloudflare Dashboard > R2 > API Tokens |
| **R2_SECRET_ACCESS_KEY** | Same as above |
| **RunPod account** | [runpod.io](https://runpod.io) — add credits ($10-25 is enough for initial runs) |

---

## 2. Create a RunPod GPU Pod

### Step 2.1 — Choose a Template

Go to **Pods > + GPU Pod** and select a base template:

| Template | Python | Why |
|---|---|---|
| `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | 3.11 | Recommended — matches our `requires-python >= 3.11` |
| `runpod/pytorch:2.2.0-py3.11-cuda12.1.0-devel-ubuntu22.04` | 3.11 | Alternative, slightly older CUDA |

> **Important:** Do NOT use templates with Python 3.10 — our code requires 3.11+. The setup script will check and exit early if the version is wrong.

### Step 2.2 — Choose a GPU

| GPU | VRAM | Cost (approx.) | Training Time (10M steps) | Recommendation |
|---|---|---|---|---|
| **RTX 4090** | 24 GB | ~$0.39/hr | ~20-30h per profile | Best value |
| **A100 40GB** | 40 GB | ~$1.10/hr | ~15-20h per profile | Fastest |
| **RTX 3090** | 24 GB | ~$0.22/hr | ~30-40h per profile | Budget option |
| **A40** | 48 GB | ~$0.76/hr | ~20-25h per profile | Good middle ground |

> A single GPU is sufficient. Our PPO config uses 1 learner + 4 CPU env runners.

### Step 2.3 — Configure Storage

- **Container Disk:** 20 GB (default is fine)
- **Volume Disk:** 50 GB — mount at `/workspace`
  - Parquets + episodes = ~140 MB
  - Checkpoints during training = ~5-10 GB per profile
  - Volume persists across pod restarts (important for long runs)

### Step 2.4 — Set Environment Variables

In the pod creation form, add these env vars:

```
WANDB_API_KEY=your_wandb_api_key
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_ENDPOINT_URL=https://9507330fe5a8c228ea49f6e5c6c6b659.r2.cloudflarestorage.com
R2_BUCKET_NAME=moleapp-rl-data
```

Click **Deploy**.

---

## 3. Connect to Your Pod

Once the pod shows **Running**, you have two options:

### Option A: Web Terminal (quickest)

Click **Connect > Start Web Terminal** in the RunPod dashboard. A browser-based terminal opens.

### Option B: SSH (recommended for long sessions)

1. In the pod details, find the SSH command:
   ```bash
   ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
   ```

2. Or use the RunPod CLI:
   ```bash
   pip install runpod
   runpodctl ssh --pod <pod-id>
   ```

### Option C: VS Code Remote SSH

1. Install the **Remote - SSH** extension in VS Code
2. Add the RunPod SSH config to `~/.ssh/config`:
   ```
   Host runpod-training
       HostName <pod-ip>
       Port <port>
       User root
       IdentityFile ~/.ssh/id_ed25519
   ```
3. Open VS Code > Remote Explorer > Connect to `runpod-training`

---

## 4. Clone & Setup

Once connected to your pod:

```bash
# Navigate to persistent volume
cd /workspace

# Clone the repo (public — no auth needed)
git clone https://github.com/Money-Legos-App/moleapp-rl-training.git
cd moleapp-rl-training

# Run the automated setup (installs deps, pulls data from R2, runs tests)
bash runpod/setup.sh
```

### What `setup.sh` does (6 steps):

```
[0/6] Checks Python >= 3.11
[1/6] pip install -e ".[dev]" — installs Ray, PyTorch, gymnasium, wandb, etc.
[2/6] Verifies Ray sees the GPU + reports VRAM
[3/6] Authenticates W&B via WANDB_API_KEY
[4/6] Pulls parquets + episodes from Cloudflare R2
[5/6] Verifies episode data exists for all 15 assets (builds from parquets if missing)
[6/6] Runs pytest to validate the environment
```

Expected output at the end:
```
=== Setup Complete ===

GPU:
  NVIDIA GeForce RTX 4090, 24564 MiB, 24322 MiB

2-Strategy System: Shield (Flexible USD Vault) + Builder (High-Yield Engine)

Training commands (GPU auto-detected, all 15 assets by default):
  Shield:  python -m training.train --profile shield  --config training/configs/shield_config.yaml
  Builder: python -m training.train --profile builder --config training/configs/builder_config.yaml
```

---

## 5. Launch Training

### Train Shield first (most constrained, fastest to validate)

```bash
# All 15 assets, GPU auto-detected
python -m training.train --profile shield --config training/configs/shield_config.yaml
```

### Run in background (survives SSH disconnect)

```bash
# Use tmux so training continues if your connection drops
tmux new -s shield

python -m training.train --profile shield --config training/configs/shield_config.yaml

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t shield
```

Or use `nohup`:

```bash
nohup python -m training.train --profile shield --config training/configs/shield_config.yaml > logs/shield.log 2>&1 &

# Monitor:
tail -f logs/shield.log
```

### Train both profiles sequentially

```bash
tmux new -s training

# Shield → Builder
python -m training.train --profile shield  --config training/configs/shield_config.yaml && \
python -m training.train --profile builder --config training/configs/builder_config.yaml

# Detach: Ctrl+B then D
```

### Train on specific assets only (faster iteration)

```bash
# Just majors — useful for quick validation runs
python -m training.train --profile shield --config training/configs/shield_config.yaml --assets BTC ETH SOL
```

### Resume from checkpoint

If training was interrupted or you want to continue:

```bash
python -m training.train \
  --profile shield \
  --config training/configs/shield_config.yaml \
  --resume models/shield/checkpoints/checkpoint_000050
```

---

## 6. Monitor Training

### W&B Dashboard

Training logs to **W&B project: `moleapp-rl`**. Go to [wandb.ai](https://wandb.ai) and find your run.

Key metrics to watch:

| Metric | Shield Target | Builder Target |
|---|---|---|
| `train/episode_reward_mean` | Trending up | Trending up |
| `train/total_return` | > 0% | > 0% |
| `train/max_drawdown` | < 10% | < 25% |
| `train/win_rate` | > 60% | > 50% |
| `eval/episode_reward_mean` | Stable, not diverging from train | Same |

**Warning signs:**
- `train/max_drawdown` stuck at the kill threshold (10% / 20%) = agent triggers drawdown kill every episode
- `train/episode_reward_mean` flat after 2M steps = learning stalled, consider HP sweep
- `eval` diverging from `train` = overfitting

### Console output

Every 5 iterations, training prints:
```
[Iter 5] steps=40,960 reward=0.15
[Iter 10] steps=81,920 reward=0.28
```

### GPU utilization

```bash
# In a separate tmux pane or terminal
watch -n 5 nvidia-smi
```

You should see ~50-80% GPU utilization during learner updates, lower during env stepping (which happens on CPU).

---

## 7. Export ONNX Models

After training completes, export models for production:

```bash
python -c "
from serving.model_registry import export_to_onnx, verify_onnx_parity

# Export Shield
artifacts = export_to_onnx(
    checkpoint_path='models/shield/shield_final',
    output_dir='models/onnx',
    profile='shield',
    version='1.0.0',
)
print('Shield exported:', artifacts)

# Verify parity (100 random observations)
ok = verify_onnx_parity(
    checkpoint_path='models/shield/shield_final',
    onnx_path='models/onnx/shield-v1.0.0.onnx',
)
print('Parity check:', 'PASSED' if ok else 'FAILED')
"
```

Repeat for Builder:

```bash
python -c "
from serving.model_registry import export_to_onnx, verify_onnx_parity

artifacts = export_to_onnx(
    checkpoint_path='models/builder/builder_final',
    output_dir='models/onnx',
    profile='builder',
    version='1.0.0',
)
ok = verify_onnx_parity(
    checkpoint_path='models/builder/builder_final',
    onnx_path='models/onnx/builder-v1.0.0.onnx',
)
print(f'builder: exported={list(artifacts.keys())}, parity={\"PASS\" if ok else \"FAIL\"}')
"
```

### Output files

```
models/onnx/
  shield-v1.0.0.onnx          # ~2 MB — the trained policy
  shield-v1.0.0.vecnorm.pkl   # Observation normalization stats
  shield-v1.0.0.meta.pkl      # Feature version, obs_dim, action_dim
  builder-v1.0.0.onnx
  builder-v1.0.0.vecnorm.pkl
  builder-v1.0.0.meta.pkl
```

---

## 8. Upload Models to R2

Upload the trained ONNX models to R2 for the agent-service to pull:

```bash
# Upload ONNX artifacts
python scripts/r2_sync.py upload \
  --data-dir models/onnx \
  --prefix models \
  --recursive
```

The agent-service on Render will pull from `s3://moleapp-rl-data/models/`.

---

## 9. Costs & GPU Recommendations

### Estimated training costs (all 15 assets, 10M timesteps per profile)

| GPU | Per Profile | Both Profiles | Notes |
|---|---|---|---|
| RTX 4090 | ~$8-12 | ~$16-24 | Best price/performance |
| A100 40GB | ~$17-22 | ~$34-44 | Fastest wall-clock time |
| RTX 3090 | ~$7-9 | ~$14-18 | Budget, slightly slower |

### Tips to save money

1. **Start with Shield only** on RTX 4090 — validate the pipeline works before training both
2. **Use spot instances** if available — up to 50% cheaper, but can be preempted (use `--resume` to continue)
3. **Stop the pod** between training runs — volume persists, you only pay for compute when running
4. **Quick validation first**: Run with `--assets BTC` and reduced timesteps to validate setup before the full run

### Reducing timesteps for quick tests

Edit the config or override in a local copy:

```bash
# Quick 1M step test (1/10th of full training)
cp training/configs/shield_config.yaml /tmp/shield_quick.yaml
sed -i 's/total_timesteps: 10_000_000/total_timesteps: 1_000_000/' /tmp/shield_quick.yaml
python -m training.train --profile shield --config /tmp/shield_quick.yaml --assets BTC
```

---

## 10. Troubleshooting

### "Python < 3.11" error

Your RunPod template uses an old Python. Re-deploy with a 3.11+ template:
```
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

### "No GPU detected"

```bash
# Check CUDA is visible
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If `torch.cuda.is_available()` returns `False`, the template may have a CPU-only PyTorch. Reinstall:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### R2 pull fails

```bash
# Verify credentials are set
echo $R2_ACCESS_KEY_ID
echo $R2_ENDPOINT_URL

# Manual test
python -c "
from scripts.r2_sync import list_files
list_files('processed/1h/')
"
```

If credentials are missing, set them:
```bash
export R2_ACCESS_KEY_ID=your_key
export R2_SECRET_ACCESS_KEY=your_secret
export R2_ENDPOINT_URL=https://9507330fe5a8c228ea49f6e5c6c6b659.r2.cloudflarestorage.com
export R2_BUCKET_NAME=moleapp-rl-data
```

### Training crashes with OOM (Out of Memory)

Reduce batch size in the config:
```yaml
train_batch_size_per_learner: 1024   # was 2048
minibatch_size: 32                    # was 64
```

Or reduce env runners:
```yaml
num_env_runners: 2   # was 4
```

### SSH disconnects and training stops

Always use `tmux` or `nohup` (see [Section 5](#5-launch-training)). If you forgot:
```bash
# Check if training is still running
ps aux | grep training.train

# If not, resume from last checkpoint
ls models/shield/checkpoints/
python -m training.train --profile shield --config training/configs/shield_config.yaml \
  --resume models/shield/checkpoints/checkpoint_000050
```

### W&B not logging

```bash
# Verify
python -c "import wandb; wandb.login(); print('OK')"

# If WANDB_API_KEY is set but not working, try explicit login
wandb login $WANDB_API_KEY
```

---

## Quick Reference

```bash
# === FULL WORKFLOW (copy-paste) ===

# 1. Connect to RunPod
ssh root@<pod-ip> -p <port>

# 2. Setup (first time only)
cd /workspace
git clone https://github.com/Money-Legos-App/moleapp-rl-training.git
cd moleapp-rl-training
bash runpod/setup.sh

# 3. Train (in tmux)
tmux new -s train
python -m training.train --profile shield  --config training/configs/shield_config.yaml && \
python -m training.train --profile builder --config training/configs/builder_config.yaml
# Ctrl+B, D to detach

# 4. Export ONNX (after training)
python -c "
from serving.model_registry import export_to_onnx
for p in ['shield', 'builder']:
    export_to_onnx(f'models/{p}/{p}_final', 'models/onnx', p, '1.0.0')
"

# 5. Upload to R2
python scripts/r2_sync.py upload --data-dir models/onnx --prefix models --recursive

# 6. Stop the pod (save money!)
# Go to RunPod dashboard > Stop Pod
```
