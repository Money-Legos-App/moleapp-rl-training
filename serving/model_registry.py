"""
Model Registry — ONNX Export + Versioning
===========================================
Exports trained SB3 models to ONNX format for production inference.
Also exports VecNormalize statistics for feature scaling.
"""

from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from data.preprocessors.feature_engineer import FEATURE_HASH, FEATURE_VERSION, OBS_DIM

logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_dir: str,
    profile: str,
    version: str = "1.0.0",
    vecnorm_path: Optional[str] = None,
) -> dict[str, Path]:
    """
    Export a trained SB3 PPO model to ONNX format.

    Extracts the policy network (actor) and exports it so that
    production inference returns the deterministic action mean.

    Args:
        model_path: Path to saved SB3 model (e.g., "models/shield/shield_final.zip")
        output_dir: Directory for ONNX artifacts
        profile: "shield", "builder", or "hunter"
        version: Semantic version string
        vecnorm_path: Path to VecNormalize stats file

    Returns:
        Dict with paths to exported artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model = PPO.load(model_path)
    policy = model.policy

    # Create dummy input
    dummy_obs = torch.randn(1, OBS_DIM).float()

    # Export the actor (policy) network to ONNX
    # We need the deterministic output (action mean), not sampled
    class DeterministicPolicy(torch.nn.Module):
        def __init__(self, sb3_policy):
            super().__init__()
            self.features_extractor = sb3_policy.features_extractor
            self.mlp_extractor = sb3_policy.mlp_extractor
            self.action_net = sb3_policy.action_net

        def forward(self, obs):
            features = self.features_extractor(obs)
            latent_pi, _ = self.mlp_extractor(features)
            action_mean = self.action_net(latent_pi)
            return action_mean

    det_policy = DeterministicPolicy(policy)
    det_policy.eval()

    onnx_filename = f"{profile}-v{version}.onnx"
    onnx_path = output_path / onnx_filename

    torch.onnx.export(
        det_policy,
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

    # Validate ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX model exported and validated: {onnx_path}")

    artifacts = {"onnx": onnx_path}

    # Export VecNormalize stats
    if vecnorm_path:
        norm_stats = _extract_norm_stats(vecnorm_path)
        norm_filename = f"{profile}-v{version}.vecnorm.pkl"
        norm_path = output_path / norm_filename
        with open(norm_path, "wb") as f:
            pickle.dump(norm_stats, f)
        artifacts["vecnorm"] = norm_path
        logger.info(f"VecNormalize stats exported: {norm_path}")

    # Export metadata
    metadata = {
        "profile": profile,
        "version": version,
        "feature_version": FEATURE_VERSION,
        "feature_hash": FEATURE_HASH,
        "obs_dim": OBS_DIM,
        "action_dim": 5,
        "model_type": "PPO",
        "deterministic": True,
    }
    meta_filename = f"{profile}-v{version}.meta.pkl"
    meta_path = output_path / meta_filename
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    artifacts["metadata"] = meta_path
    logger.info(f"Metadata exported: {meta_path}")

    return artifacts


def _extract_norm_stats(vecnorm_path: str) -> dict:
    """Extract running mean/var from VecNormalize save file."""
    # Try loading as raw stats first
    with open(vecnorm_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "obs_rms_mean" in data:
        return data

    # If it's a VecNormalize object, extract stats
    if hasattr(data, "obs_rms"):
        return {
            "obs_rms_mean": data.obs_rms.mean.copy(),
            "obs_rms_var": data.obs_rms.var.copy(),
            "obs_rms_count": data.obs_rms.count,
            "clip_obs": data.clip_obs,
        }

    raise ValueError(f"Unrecognized VecNormalize format in {vecnorm_path}")


def verify_onnx_parity(
    sb3_model_path: str,
    onnx_path: str,
    n_samples: int = 100,
    atol: float = 1e-5,
) -> bool:
    """
    Verify that ONNX model output matches SB3 deterministic predictions.

    This is the critical parity test — if this fails, production inference
    will diverge from what was validated during training evaluation.
    """
    import onnxruntime as ort

    model = PPO.load(sb3_model_path)
    session = ort.InferenceSession(onnx_path)

    all_match = True
    for _ in range(n_samples):
        obs = np.random.randn(1, OBS_DIM).astype(np.float32)

        # SB3 deterministic prediction
        with torch.no_grad():
            sb3_action, _, _ = model.policy.forward(torch.from_numpy(obs), deterministic=True)
            sb3_action = sb3_action.numpy()

        # ONNX prediction
        onnx_action = session.run(["action_mean"], {"observation": obs})[0]

        if not np.allclose(sb3_action, onnx_action, atol=atol):
            logger.error(f"Parity mismatch: SB3={sb3_action}, ONNX={onnx_action}")
            all_match = False

    if all_match:
        logger.info(f"ONNX parity verified: {n_samples} samples match within atol={atol}")
    return all_match
