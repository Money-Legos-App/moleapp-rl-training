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
    """Standalone MLP that mirrors RLlib's default FCNet architecture."""

    def __init__(self, obs_dim: int, action_dim: int, hiddens: list[int], activation: str = "tanh"):
        super().__init__()

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]

        # Policy network (same structure as RLlib default FC)
        layers = []
        prev_size = obs_dim
        for h in hiddens:
            layers.append(nn.Linear(prev_size, h))
            layers.append(act_fn())
            prev_size = h
        self.policy_net = nn.Sequential(*layers)

        # Action mean head (continuous: 2*action_dim for mean + log_std)
        self.action_head = nn.Linear(prev_size, action_dim * 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy_net(obs)
        action_dist_inputs = self.action_head(features)
        # Take only action means (first action_dim dims), ignore log_std
        return action_dist_inputs[:, :ACTION_DIM]


def load_rl_module_weights(checkpoint_path: str) -> dict:
    """Load RLModule state dict from an RLlib checkpoint directory."""
    ckpt_dir = Path(checkpoint_path)

    # RLlib new stack stores module weights in learner/
    # Try multiple known paths
    candidates = [
        ckpt_dir / "learner" / "learner_group" / "learner" / "rl_module" / "default_policy" / "module_state.pkl",
        ckpt_dir / "learner" / "module_state.pkl",
        ckpt_dir / "learner" / "learner_group" / "default_policy" / "module_state.pkl",
    ]

    # Also try finding any .pkl that contains state dict
    for pkl_path in ckpt_dir.rglob("*.pkl"):
        if "module_state" in pkl_path.name or "rl_module" in str(pkl_path):
            candidates.insert(0, pkl_path)

    for path in candidates:
        if path.exists():
            logger.info(f"Found module state at: {path}")
            with open(path, "rb") as f:
                state = pickle.load(f)
            return state

    # Try .pt files (PyTorch format)
    for pt_path in ckpt_dir.rglob("*.pt"):
        logger.info(f"Found .pt file at: {pt_path}")
        state = torch.load(pt_path, map_location="cpu", weights_only=False)
        return state

    raise FileNotFoundError(
        f"No module_state found in {checkpoint_path}. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def map_rllib_weights(rllib_state: dict, model: DeterministicPolicy) -> None:
    """Map RLlib weight names to our standalone model."""
    model_state = model.state_dict()

    # RLlib FCNet weight naming convention:
    #   encoder.net.0.weight → policy_net.0.weight (Linear)
    #   encoder.net.1.bias   → (skip activation layers)
    #   pi.0.weight          → action_head.weight
    #
    # Dump keys to understand structure
    logger.info(f"RLlib state keys: {list(rllib_state.keys())[:20]}")
    logger.info(f"Model state keys: {list(model_state.keys())}")

    # Try automatic mapping
    rllib_keys = list(rllib_state.keys())
    model_keys = list(model_state.keys())

    # Separate encoder and action head keys
    encoder_weights = []
    action_weights = []

    for k in rllib_keys:
        v = rllib_state[k]
        if not isinstance(v, (torch.Tensor, np.ndarray)):
            continue
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if "pi" in k or "action" in k or "_action" in k:
            action_weights.append((k, v))
        elif "vf" in k or "value" in k or "_value" in k:
            continue  # Skip value function
        else:
            encoder_weights.append((k, v))

    # Map encoder weights to policy_net
    policy_params = [(k, v) for k, v in model_state.items() if k.startswith("policy_net")]
    logger.info(f"Encoder weights found: {len(encoder_weights)}, policy params: {len(policy_params)}")

    if len(encoder_weights) == len(policy_params):
        for (rk, rv), (mk, _) in zip(encoder_weights, policy_params):
            model_state[mk] = rv.float()
            logger.info(f"  {rk} → {mk} ({rv.shape})")
    else:
        logger.warning(f"Weight count mismatch: {len(encoder_weights)} vs {len(policy_params)}")
        # Try matching by shape
        for mk, mv in list(model_state.items()):
            if mk.startswith("policy_net"):
                for rk, rv in encoder_weights:
                    if rv.shape == mv.shape:
                        model_state[mk] = rv.float()
                        encoder_weights.remove((rk, rv))
                        logger.info(f"  {rk} → {mk} (shape match: {rv.shape})")
                        break

    # Map action head
    action_params = [(k, v) for k, v in model_state.items() if k.startswith("action_head")]
    if len(action_weights) >= len(action_params):
        for (rk, rv), (mk, _) in zip(action_weights, action_params):
            model_state[mk] = rv.float()
            logger.info(f"  {rk} → {mk} ({rv.shape})")

    model.load_state_dict(model_state)
    logger.info("Weight mapping complete")


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
