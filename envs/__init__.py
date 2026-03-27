from envs.base_trading_env import BaseTradingEnv
from envs.shield_env import ShieldTradingEnv
from envs.builder_env import BuilderTradingEnv

from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from ray.tune.registry import register_env


def _make_env(env_cls, cfg):
    """Create env with observation + reward normalization wrappers."""
    env = env_cls(**cfg)
    env = NormalizeObservation(env)
    env = NormalizeReward(env, gamma=0.995)
    return env


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
