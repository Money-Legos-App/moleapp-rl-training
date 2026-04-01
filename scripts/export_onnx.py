"""
Standalone ONNX Export — bypasses Algorithm.from_checkpoint()
==============================================================
Algorithm.from_checkpoint() tries to recreate the env (which needs
training data files in /tmp). This script loads only the RLModule
weights from the checkpoint, wraps in DeterministicPolicy, and exports.

Usage:
    python scripts/export_onnx.py \
        --checkpoint models/shield/best_checkpoint \
        --profile shield \
        --version 10.0.0 \
        --norm-stats models/shield/shield_norm_stats.pkl \
        --output-dir models/onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must match training obs dim
OBS_DIM = 47
ACTION_DIM = 5


class DeterministicPolicy(nn.Module):
    """Standalone MLP that mirrors RLlib's default FCNet (actor encoder + pi head).

    Architecture (matches RLlib new API stack PPO):
      policy_net: obs(47) → 256 → tanh → 256 → tanh → 128 → tanh
      action_head: 128 → 10 (5 action means + 5 log_stds)
      output: first 5 dims only (deterministic action means)
    """

    def __init__(self, obs_dim: int, action_dim: int, hiddens: list[int], activation: str = "tanh"):
        super().__init__()

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]

        layers = []
        prev_size = obs_dim
        for h in hiddens:
            layers.append(nn.Linear(prev_size, h))
            layers.append(act_fn())
            prev_size = h
        self.policy_net = nn.Sequential(*layers)

        # RLlib outputs 2*action_dim (means + log_stds)
        self.action_head = nn.Linear(prev_size, action_dim * 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy_net(obs)
        action_dist_inputs = self.action_head(features)
        # Deterministic: take only action means (first 5), discard log_stds
        return action_dist_inputs[:, :ACTION_DIM]


def load_rl_module_weights(checkpoint_path: str) -> dict:
    """Load RLModule state dict from an RLlib checkpoint directory.

    RLlib new API stack (v2.54+) stores weights at:
    <checkpoint>/learner_group/learner/rl_module/default_policy/module_state.pkl

    Keys look like:
      encoder.actor_encoder.net.mlp.{0,2,4}.{weight,bias}  (policy MLP)
      encoder.critic_encoder.net.mlp.{0,2,4}.{weight,bias}  (value MLP)
      pi.net.mlp.0.{weight,bias}  (action head: 128→10, means+log_std)
      vf.net.mlp.0.{weight,bias}  (value head: 128→1)
    """
    ckpt_dir = Path(checkpoint_path)

    # Exact path for RLlib new stack
    module_state_path = (
        ckpt_dir / "learner_group" / "learner" / "rl_module"
        / "default_policy" / "module_state.pkl"
    )

    if not module_state_path.exists():
        # Fallback: search for any module_state.pkl
        candidates = list(ckpt_dir.rglob("module_state.pkl"))
        # Prefer the one under default_policy
        for c in candidates:
            if "default_policy" in str(c):
                module_state_path = c
                break
        else:
            if candidates:
                module_state_path = candidates[0]
            else:
                raise FileNotFoundError(f"No module_state.pkl found in {checkpoint_path}")

    logger.info(f"Loading weights from: {module_state_path}")
    with open(module_state_path, "rb") as f:
        state = pickle.load(f)

    # Log keys for debugging
    weight_keys = [k for k in state if hasattr(state[k], "shape")]
    logger.info(f"Found {len(weight_keys)} weight tensors")
    for k in weight_keys:
        logger.info(f"  {k}: {state[k].shape}")

    return state


def map_rllib_weights(rllib_state: dict, model: DeterministicPolicy) -> None:
    """Map RLlib new API stack weight names to our standalone model.

    RLlib keys (new API stack, PPO with separate actor/critic encoders):
      encoder.actor_encoder.net.mlp.0.weight  → policy_net.0.weight  (47→256)
      encoder.actor_encoder.net.mlp.0.bias    → policy_net.0.bias
      encoder.actor_encoder.net.mlp.2.weight  → policy_net.2.weight  (256→256)
      encoder.actor_encoder.net.mlp.2.bias    → policy_net.2.bias
      encoder.actor_encoder.net.mlp.4.weight  → policy_net.4.weight  (256→128)
      encoder.actor_encoder.net.mlp.4.bias    → policy_net.4.bias
      pi.net.mlp.0.weight                     → action_head.weight   (128→10)
      pi.net.mlp.0.bias                       → action_head.bias

    We skip: encoder.critic_encoder.*, vf.*, pi.log_std_clip_param_const
    """
    # Explicit mapping: RLlib key → our model key
    KEY_MAP = {
        "encoder.actor_encoder.net.mlp.0.weight": "policy_net.0.weight",
        "encoder.actor_encoder.net.mlp.0.bias": "policy_net.0.bias",
        "encoder.actor_encoder.net.mlp.2.weight": "policy_net.2.weight",
        "encoder.actor_encoder.net.mlp.2.bias": "policy_net.2.bias",
        "encoder.actor_encoder.net.mlp.4.weight": "policy_net.4.weight",
        "encoder.actor_encoder.net.mlp.4.bias": "policy_net.4.bias",
        "pi.net.mlp.0.weight": "action_head.weight",
        "pi.net.mlp.0.bias": "action_head.bias",
    }

    model_state = model.state_dict()
    mapped = 0

    for rllib_key, model_key in KEY_MAP.items():
        if rllib_key not in rllib_state:
            logger.error(f"Missing RLlib weight: {rllib_key}")
            continue
        if model_key not in model_state:
            logger.error(f"Missing model param: {model_key}")
            continue

        src = rllib_state[rllib_key]
        if isinstance(src, np.ndarray):
            src = torch.from_numpy(src)
        dst_shape = model_state[model_key].shape

        if src.shape != dst_shape:
            logger.error(f"Shape mismatch: {rllib_key} {src.shape} vs {model_key} {dst_shape}")
            continue

        model_state[model_key] = src.float()
        mapped += 1
        logger.info(f"  {rllib_key} → {model_key} ({src.shape})")

    if mapped != len(KEY_MAP):
        raise RuntimeError(f"Only mapped {mapped}/{len(KEY_MAP)} weights — export would be corrupt")

    model.load_state_dict(model_state)
    logger.info(f"Weight mapping complete: {mapped}/{len(KEY_MAP)} tensors loaded")


def export(
    checkpoint_path: str,
    output_dir: str,
    profile: str,
    version: str,
    norm_stats_path: str | None = None,
    hiddens: list[int] | None = None,
):
    if hiddens is None:
        hiddens = [256, 256, 128]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Build standalone model
    model = DeterministicPolicy(OBS_DIM, ACTION_DIM, hiddens, activation="tanh")
    logger.info(f"Built model: {sum(p.numel() for p in model.parameters())} params")

    # 2. Load and map weights
    rllib_state = load_rl_module_weights(checkpoint_path)
    map_rllib_weights(rllib_state, model)
    model.eval()

    # 3. Export to ONNX
    dummy_obs = torch.randn(1, OBS_DIM).float()
    onnx_filename = f"{profile}-v{version}.onnx"
    onnx_path = output_path / onnx_filename

    torch.onnx.export(
        model,
        dummy_obs,
        str(onnx_path),
        input_names=["observation"],
        output_names=["action_mean"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action_mean": {0: "batch_size"},
        },
        opset_version=17,
    )

    # 4. Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX exported and validated: {onnx_path}")

    # 5. Copy norm stats
    artifacts = {"onnx": onnx_path}
    if norm_stats_path and Path(norm_stats_path).exists():
        norm_filename = f"{profile}-v{version}.vecnorm.pkl"
        norm_out = output_path / norm_filename
        import shutil
        shutil.copy2(norm_stats_path, norm_out)
        artifacts["vecnorm"] = norm_out
        logger.info(f"Norm stats copied: {norm_out}")

    # 6. Export metadata
    metadata = {
        "profile": profile,
        "version": version,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "model_type": "PPO",
        "hiddens": hiddens,
        "deterministic": True,
    }
    meta_path = output_path / f"{profile}-v{version}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    artifacts["metadata"] = meta_path

    # 7. Quick sanity check
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    test_obs = np.random.randn(1, OBS_DIM).astype(np.float32)
    result = session.run(["action_mean"], {"observation": test_obs})[0]
    logger.info(f"Sanity check — ONNX output shape: {result.shape}, sample: {result[0][:3]}")

    print("\n=== Export Complete ===")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")

    return artifacts


def main():
    parser = argparse.ArgumentParser(description="Export RLlib checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to RLlib checkpoint dir")
    parser.add_argument("--profile", required=True, choices=["shield", "builder"])
    parser.add_argument("--version", default="10.0.0")
    parser.add_argument("--norm-stats", default=None, help="Path to norm_stats.pkl")
    parser.add_argument("--output-dir", default="models/onnx")
    parser.add_argument("--hiddens", nargs="+", type=int, default=[256, 256, 128])
    args = parser.parse_args()

    export(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        profile=args.profile,
        version=args.version,
        norm_stats_path=args.norm_stats,
        hiddens=args.hiddens,
    )


if __name__ == "__main__":
    main()
