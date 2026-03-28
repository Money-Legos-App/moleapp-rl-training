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
6. [Hyperparameter Sweep (V4)](#6-hyperparameter-sweep-v4)
7. [Monitor Training](#7-monitor-training)
8. [Export ONNX Models](#8-export-onnx-models)
9. [Upload Models to R2](#9-upload-models-to-r2)
10. [Costs & GPU Recommendations](#10-costs--gpu-recommendations)
11. [Troubleshooting](#11-troubleshooting)

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

> A single GPU is sufficient. Our PPO config uses 1 learner + 2 CPU env runners. Note: For the 200K-parameter network, GPU utilization is minimal (~2% VRAM) — the bottleneck is CPU-based environment stepping, not gradient computation. The GPU mainly accelerates tensor operations during learner updates.

### Step 2.3 — Configure Storage

- **Container Disk:** 50 GB (default 20 GB fills up fast with Ray temp files)
- **Volume Disk:** 250 GB — mount at `/workspace`
  - Parquets + episodes = ~140 MB
  - Checkpoints during training = ~5-10 GB per profile
  - Ray Tune sweep results = ~20-30 GB per sweep (market data serialized per trial)
  - Volume persists across pod restarts (important for long runs)

> **Critical:** Always redirect Ray's temp and results to the volume — the container disk fills up quickly. See [Section 6](#6-hyperparameter-sweep-v4) for details.

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

# Install tmux for background training (recommended)
apt-get update && apt-get install -y tmux

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
[6/6] Runs pytest to validate environments (PPO overfit tests are skipped — marked @slow)
```

Expected output at the end:
```
126 passed, 2 deselected in ~2 min

=== Setup Complete ===

GPU:
  NVIDIA GeForce RTX 4090, 24564 MiB, 24322 MiB

2-Strategy System: Shield (Flexible USD Vault) + Builder (High-Yield Engine)

Training commands (GPU auto-detected, all 15 assets by default):
  Shield:  python -m training.train --profile shield  --config training/configs/shield_config.yaml
  Builder: python -m training.train --profile builder --config training/configs/builder_config.yaml
```

> **Note:** 2 PPO overfit tests are skipped during setup (marked `@pytest.mark.slow`). These train a PPO model and take several minutes. Run them separately with `python -m pytest tests/ -v -m slow` when the pod is idle.

### If you `git pull` after setup

The `data/` directory is `.gitignore`d, so a `git pull` may clear the cached episodes. Re-pull data without re-running the full setup:

```bash
python scripts/r2_sync.py download --prefix processed --data-dir data/datasets
python scripts/r2_sync.py download --prefix episodes --data-dir data/episodes
```

---

## 5. Launch Training

### Suppress RLlib deprecation warnings

RLlib v2.54+ prints many deprecation warnings (RLModule config, RunConfig, etc.) that are internal to Ray and don't affect training. Suppress them:

```bash
export PYTHONWARNINGS="ignore::DeprecationWarning"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DEDUP_LOGS=1
```

Add these to your shell or prefix the training command.

### Redirect Ray temp files to volume

Ray writes large temp files to `/tmp` by default, which fills the container disk quickly. Always redirect:

```bash
export RAY_TMPDIR=/workspace/ray_tmp
mkdir -p /workspace/ray_tmp
```

### Train Shield first (most constrained, fastest to validate)

```bash
tmux new -s train

# Suppress warnings + redirect temp + train
PYTHONWARNINGS="ignore::DeprecationWarning" RAY_DISABLE_DOCKER_CPU_WARNING=1 \
RAY_TMPDIR=/workspace/ray_tmp \
  python -m training.train --profile shield --config training/configs/shield_config.yaml

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t train
```

### Train both profiles sequentially

```bash
tmux new -s train

PYTHONWARNINGS="ignore::DeprecationWarning" RAY_DISABLE_DOCKER_CPU_WARNING=1 \
RAY_TMPDIR=/workspace/ray_tmp bash -c '
  python -m training.train --profile shield  --config training/configs/shield_config.yaml && \
  python -m training.train --profile builder --config training/configs/builder_config.yaml
'

# Ctrl+B, D to detach
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

### How training data is loaded

Training data (254K+ timesteps across 15 assets) is saved to `/tmp/rl_training_data/` as `.npy` and `.pkl` files. Each Ray env runner loads data from these local files instead of receiving them through Ray's object store — this avoids a 10+ minute serialization bottleneck.

---

## 6. Hyperparameter Sweep (V4)

Before full 10M-step training, run a hyperparameter sweep to find optimal configs for each profile. The sweep uses Ray Tune with ASHA early stopping.

### Why sweep?

Different risk profiles need fundamentally different hyperparameters:
- **Shield** optimizes `risk_adjusted_return` (return / max_drawdown) — tight clip, high gamma, low entropy
- **Builder** optimizes `episode_return_mean` (raw returns) — loose clip, lower gamma, higher entropy

### The "Valley of Death"

Hyperliquid agents need time to learn real-world fee structures (maker/taker fees, slippage, funding rates). Early in training, agents lose money just figuring out the market. The ASHA scheduler's grace period (300K steps) protects late-blooming agents from being killed before they learn — without this, ASHA biases toward "coward" configs that sit in cash and never trade.

### Run the sweep

```bash
tmux new -s sweep

# Clean previous results
rm -rf /workspace/ray_results/*

# Shield sweep (~3 hours on RTX 4090)
PYTHONWARNINGS="ignore::DeprecationWarning" RAY_DISABLE_DOCKER_CPU_WARNING=1 \
RAY_TMPDIR=/workspace/ray_tmp \
  python -m training.tune_sweep --profile shield 2>&1 | tee /workspace/sweep_log.txt

# After Shield completes, run Builder sweep
PYTHONWARNINGS="ignore::DeprecationWarning" RAY_DISABLE_DOCKER_CPU_WARNING=1 \
RAY_TMPDIR=/workspace/ray_tmp \
  python -m training.tune_sweep --profile builder 2>&1 | tee /workspace/builder_sweep_log.txt

# Ctrl+B, D to detach
```

### Sweep configuration (V4)

| Setting | Shield | Builder |
|---|---|---|
| Total steps per trial | 750K | 750K |
| Number of trials | 6 | 6 |
| Grace period (ASHA) | 300K | 300K |
| Optimization metric | `risk_adjusted_return` | `episode_return_mean` |
| LR range | 5e-5 to 3e-4 | 1e-4 to 1e-3 |
| Clip range | 0.08-0.18 (tight) | 0.15-0.30 (loose) |
| Gamma range | 0.995-0.999 (patient) | 0.99-0.997 (near-term) |
| Entropy start | 0.002-0.008 (low) | 0.005-0.02 (high) |

### Dry-run (quick sanity check)

```bash
python -m training.tune_sweep --profile shield --dry-run --dry-run-steps 100
```

### After sweep completes

The sweep logs the best trial's config. Update `training/configs/shield_config.yaml` (or `builder_config.yaml`) with the winning hyperparameters, then run the full 10M-step training.

### Sweep disk management

Sweeps generate significant disk usage (~20-30 GB) because Ray serializes market data per trial:

```bash
# Monitor disk usage during sweep
watch -n 60 'du -sh /workspace/ray_results/* /workspace/ray_tmp/*'

# Clean old sessions between sweep restarts
rm -rf /workspace/ray_results/* /workspace/ray_tmp/ray/session_*

# Check if old sweep runs are mixed in
ls /workspace/ray_results/shield-sweep-v4/ | grep PPO | wc -l
# Should be 6 (one per trial). If more, old runs need cleaning.
```

### Estimated sweep times (RTX 4090, 16K steps/min measured)

| Phase | Time |
|---|---|
| Trial setup | ~1 min |
| Reach grace period (300K) | ~18 min |
| Full trial (750K) | ~47 min |
| **Shield sweep (6 trials, ASHA kills ~3)** | **~3 hours** |
| **Builder sweep (6 trials)** | **~3 hours** |
| **Total both sweeps** | **~6 hours** |

---

## 7. Monitor Training

### W&B Dashboard

Training logs to **W&B project: `moleapp-rl`**. Go to [wandb.ai](https://wandb.ai) and find your run.

Key metrics to watch (RLlib new API stack — metrics under `env_runners/`):

| Metric | Shield Target | Builder Target |
|---|---|---|
| `env_runners/episode_return_mean` | Trending up | Trending up |
| `env_runners/total_return` | > 0% | > 0% |
| `env_runners/max_drawdown` | < 10% | < 25% |
| `env_runners/win_rate` | > 60% | > 50% |
| `env_runners/risk_adjusted_return` | > 1.0 | N/A |

> **Note:** RLlib v2.54+ renamed `episode_reward_mean` to `episode_return_mean`. The old key returns 0.0 on the new API stack. Our callbacks also log `risk_adjusted_return` = total_return / max(max_drawdown, 0.01).

**Warning signs:**
- `max_drawdown` stuck at the kill threshold (10% / 20%) = agent triggers drawdown kill every episode
- `episode_return_mean` flat after 2M steps = learning stalled, run HP sweep
- Large negative returns in first 300K steps are **normal** — agents are learning Hyperliquid fee structures ("valley of death")
- All returns near zero = agent learned to do nothing (coward config) — needs higher entropy or longer grace period

### Console output

Every 5 iterations, training prints:
```
[Iter 5] steps=81,920 reward=-15946.23 return=-0.045 dd=0.067 wr=0.28 risk_adj=-0.67 trades=12
[Iter 10] steps=163,840 reward=-10623.41 return=-0.012 dd=0.054 wr=0.31 risk_adj=-0.22 trades=18
```

> Early negative rewards are expected — agents are learning Hyperliquid fee structures. Returns should improve past 300K steps.

### GPU utilization

```bash
# In a separate tmux pane (Ctrl+B, %) or terminal
watch -n 5 nvidia-smi
```

With a 200K-parameter network, GPU utilization is low (~2% VRAM, ~560 MiB). This is normal — the network is small and the bottleneck is CPU-based environment stepping, not gradient computation. The GPU handles learner tensor operations in short bursts.

### RunPod Telemetry

Check the **Telemetry** tab in the RunPod dashboard for:
- **CPU load**: 8-30% during training (env runners use CPU, 2 runners default)
- **Memory**: 40-60% (market data copies in Ray object store)
- **GPU VRAM**: ~2% (~560 MiB) — normal for 200K param network
- **GPU Utilization**: 0% between iterations, brief spikes during learner updates
- **Disk**: Monitor container disk — redirect Ray temp to volume (see Section 5)
- **Processes**: ~800-1300 is normal (Ray spawns many worker processes)

---

## 8. Export ONNX Models

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

## 9. Upload Models to R2

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

## 10. Costs & GPU Recommendations

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

## 11. Troubleshooting

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

### "No cached episodes" after `git pull`

The `data/` directory is `.gitignore`d. Re-pull data:
```bash
python scripts/r2_sync.py download --prefix processed --data-dir data/datasets
python scripts/r2_sync.py download --prefix episodes --data-dir data/episodes
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

### Container disk full (`No space left on device`)

Ray writes temp files and logs to `/tmp` by default. Redirect to the volume:
```bash
export RAY_TMPDIR=/workspace/ray_tmp
mkdir -p /workspace/ray_tmp /workspace/ray_results
```

Clean up between runs:
```bash
rm -rf /tmp/ray /root/ray_results
rm -rf /workspace/ray_tmp/ray/session_*   # old sessions
```

### Sweep crashes silently (no traceback)

This usually means a Ray resource conflict. Common causes:
1. **GPU resource mismatch**: `num_learners=0` with `.resources(num_gpus=1)` can cause conflicts. The sweep uses CPU-only training by default (200K params don't need GPU).
2. **Object store pressure**: Each env runner holds a copy of 254K-timestep market data. Keep `num_env_runners=2` during sweeps.
3. **Kernel OOM kill**: Check `dmesg | grep -i "oom\|killed"` for silent kills.

Save sweep output for debugging:
```bash
python -m training.tune_sweep --profile shield 2>&1 | tee /workspace/sweep_log.txt
```

### Sweep `process_trial_result` warnings (6-15 seconds)

```
WARNING -- The `process_trial_result` operation took 10.448 s
```

This is normal — Ray serializes large market data arrays in the trial config. It doesn't affect training speed, just the reporting interval between iterations.

### ASHA kills all trials (all configs look bad)

If ASHA kills every trial at the grace period, agents may still be in the "valley of death" learning fee structures. Solutions:
1. Increase `grace_period` (e.g., 300K → 500K) in `tune_sweep.py`
2. Increase `SWEEP_TOTAL_TIMESTEPS` proportionally
3. Check that reward shaping isn't too harsh early in training

### Training hangs at startup (no iterations for 5+ minutes)

This is a Ray serialization bottleneck. The fix (already applied) is to pass file paths instead of raw data through `env_config`. If you see this on an older commit:

```bash
git pull origin main  # get the fix
# re-pull data if needed (see above)
```

### SSH disconnects and training stops

Always use `tmux` (see [Section 5](#5-launch-training)). If you forgot:
```bash
# Check if training is still running
ps aux | grep training.train

# If not, resume from last checkpoint
ls models/shield/checkpoints/
python -m training.train --profile shield --config training/configs/shield_config.yaml \
  --resume models/shield/checkpoints/checkpoint_000050
```

### RLlib deprecation warnings

Warnings like `UnifiedLogger will be removed`, `RLModule(config=...) deprecated`, `CSVLogger deprecated` are **internal to RLlib v2.40+** and cannot be fixed from our code. They don't affect training. Suppress with:

```bash
export PYTHONWARNINGS="ignore::DeprecationWarning"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
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
apt-get update && apt-get install -y tmux
bash runpod/setup.sh

# 3. Environment setup (every session)
export PYTHONWARNINGS="ignore::DeprecationWarning"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_TMPDIR=/workspace/ray_tmp
mkdir -p /workspace/ray_tmp /workspace/ray_results

# 4. Hyperparameter sweep (optional, ~6h for both profiles)
tmux new -s sweep
python -m training.tune_sweep --profile shield 2>&1 | tee /workspace/sweep_log.txt
# Update configs with best params, then:
python -m training.tune_sweep --profile builder 2>&1 | tee /workspace/builder_sweep_log.txt
# Ctrl+B, D to detach

# 5. Full training (in tmux, ~20-30h per profile)
tmux new -s train
python -m training.train --profile shield  --config training/configs/shield_config.yaml && \
python -m training.train --profile builder --config training/configs/builder_config.yaml
# Ctrl+B, D to detach

# 6. Export ONNX (after training)
python -c "
from serving.model_registry import export_to_onnx
for p in ['shield', 'builder']:
    export_to_onnx(f'models/{p}/{p}_final', 'models/onnx', p, '1.0.0')
"

# 7. Upload to R2
python scripts/r2_sync.py upload --data-dir models/onnx --prefix models --recursive

# 8. Stop the pod (save money!)
# Go to RunPod dashboard > Stop Pod
```
