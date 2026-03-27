from envs.base_trading_env import BaseTradingEnv
from envs.shield_env import ShieldTradingEnv
from envs.builder_env import BuilderTradingEnv

from gymnasium.wrappers import NormalizeObservation
from ray.tune.registry import register_env


def _make_env(env_cls, cfg):
    """Create env with NormalizeObservation wrapper (replaces MeanStdFilter)."""
    env = env_cls(**cfg)
    return NormalizeObservation(env)


# RLlib env registration — wrapped with observation normalization
register_env("ShieldTradingEnv", lambda cfg: _make_env(ShieldTradingEnv, cfg))
register_env("BuilderTradingEnv", lambda cfg: _make_env(BuilderTradingEnv, cfg))

ENV_MAP = {
    "shield": "ShieldTradingEnv",
    "builder": "BuilderTradingEnv",
}

__all__ = [
    "BaseTradingEnv",
    "ShieldTradingEnv",
    "BuilderTradingEnv",
    "ENV_MAP",
]
