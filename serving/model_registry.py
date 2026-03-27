"""
Model Registry — ONNX Export + Versioning (RLlib)
===================================================
Exports trained RLlib PPO models to ONNX format for production inference.
Also exports MeanStdFilter statistics for feature scaling.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import torch
from ray.rllib.algorithms.algorithm import Algorithm

from data.preprocessors.feature_engineer import FEATURE_HASH, FEATURE_VERSION, OBS_DIM

logger = logging.getLogger(__name__)


class DeterministicPolicy(torch.nn.Module):
    """Wraps an RLlib RLModule to output deterministic action means via forward_inference."""

    def __init__(self, rl_module):
        super().__init__()
        self.rl_module = rl_module

    def forward(self, obs):
        fwd_out = self.rl_module.forward_inference({"obs": obs})
        # action_dist_inputs contains the mean for continuous actions
        action_mean = fwd_out["action_dist_inputs"]
        # Take only the first 5 dims (action means, excluding log_std if present)
        action_dim = 5
        if action_mean.shape[-1] > action_dim:
            action_mean = action_mean[:, :action_dim]
        return action_mean


def export_to_onnx(
    checkpoint_path: str,
    output_dir: str,
    profile: str,
    version: str = "1.0.0",
    norm_stats_path: Optional[str] = None,
) -> dict[str, Path]:
    """
    Export a trained RLlib PPO model to ONNX format.

    Args:
        checkpoint_path: Path to RLlib checkpoint directory
        output_dir: Directory for ONNX artifacts
        profile: "shield" or "builder"
        version: Semantic version string
        norm_stats_path: Path to norm_stats.pkl (from _save_norm_stats)

    Returns:
        Dict with paths to exported artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load RLlib algorithm from checkpoint
    algo = Algorithm.from_checkpoint(checkpoint_path)
    rl_module = algo.get_module()

    # Create dummy input
    dummy_obs = torch.randn(1, OBS_DIM).float()

    # Wrap for deterministic output
    det_policy = DeterministicPolicy(rl_module)
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

    # Export normalization stats
    if norm_stats_path:
        norm_stats = _load_norm_stats(norm_stats_path)
        norm_filename = f"{profile}-v{version}.vecnorm.pkl"
        norm_path = output_path / norm_filename
        with open(norm_path, "wb") as f:
            pickle.dump(norm_stats, f)
        artifacts["vecnorm"] = norm_path
        logger.info(f"Norm stats exported: {norm_path}")
    else:
        # Try extracting from the algorithm's MeanStdFilter
        norm_stats = _extract_filter_stats(algo)
        if norm_stats:
            norm_filename = f"{profile}-v{version}.vecnorm.pkl"
            norm_path = output_path / norm_filename
            with open(norm_path, "wb") as f:
                pickle.dump(norm_stats, f)
            artifacts["vecnorm"] = norm_path
            logger.info(f"Norm stats extracted from filter: {norm_path}")

    # Export metadata
    metadata = {
        "profile": profile,
        "version": version,
        "feature_version": FEATURE_VERSION,
        "feature_hash": FEATURE_HASH,
        "obs_dim": OBS_DIM,
        "action_dim": 5,
        "model_type": "PPO",
        "framework": "rllib",
        "deterministic": True,
    }
    meta_filename = f"{profile}-v{version}.meta.pkl"
    meta_path = output_path / meta_filename
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    artifacts["metadata"] = meta_path
    logger.info(f"Metadata exported: {meta_path}")

    algo.stop()
    return artifacts


def _load_norm_stats(path: str) -> dict:
    """Load normalization stats from pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "obs_rms_mean" in data:
        return data

    raise ValueError(f"Unrecognized norm stats format in {path}")


def _extract_filter_stats(algo) -> dict | None:
    """Extract MeanStdFilter stats from a running algorithm."""
    try:
        filters = algo.env_runner.filters
        if "default_policy" in filters:
            obs_filter = filters["default_policy"]
            return {
                "obs_rms_mean": np.array(obs_filter.rs.mean, dtype=np.float64),
                "obs_rms_var": np.array(obs_filter.rs.var, dtype=np.float64),
                "obs_rms_count": float(obs_filter.rs.count),
                "clip_obs": 10.0,
            }
    except Exception as e:
        logger.warning(f"Could not extract filter stats: {e}")
    return None


def verify_onnx_parity(
    checkpoint_path: str,
    onnx_path: str,
    norm_stats_path: Optional[str] = None,
    n_samples: int = 100,
    atol: float = 1e-5,
) -> bool:
    """
    Verify that ONNX model output matches RLlib deterministic predictions.

    This is the critical parity test — if this fails, production inference
    will diverge from what was validated during training evaluation.
    """
    import onnxruntime as ort

    algo = Algorithm.from_checkpoint(checkpoint_path)
    rl_module = algo.get_module()
    session = ort.InferenceSession(onnx_path)

    all_match = True
    for _ in range(n_samples):
        obs = np.random.randn(OBS_DIM).astype(np.float32)

        # RLlib deterministic prediction via RLModule
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        fwd_out = rl_module.forward_inference({"obs": obs_tensor})
        action_dist_inputs = fwd_out["action_dist_inputs"].squeeze(0).detach().numpy()
        rllib_action = action_dist_inputs[:5]  # action means only

        # ONNX prediction
        obs_batch = obs.reshape(1, -1)
        onnx_action = session.run(["action_mean"], {"observation": obs_batch})[0].flatten()

        if not np.allclose(rllib_action, onnx_action, atol=atol):
            logger.error(f"Parity mismatch: RLlib={rllib_action}, ONNX={onnx_action}")
            all_match = False

    if all_match:
        logger.info(f"ONNX parity verified: {n_samples} samples match within atol={atol}")

    algo.stop()
    return all_match
