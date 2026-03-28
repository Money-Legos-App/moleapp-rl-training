from envs.base_trading_env import BaseTradingEnv
from envs.shield_env import ShieldTradingEnv
from envs.builder_env import BuilderTradingEnv

from gymnasium.wrappers import NormalizeObservation
from ray.tune.registry import register_env


def _make_env(env_cls, cfg):
    """Create env with observation normalization wrapper.

    Reward normalization is NOT used — the ×100 scaling in _calculate_reward()
    produces rewards in the right range (0.01-0.5). NormalizeReward with high
    gamma + long episodes (2880 steps) was undoing the scaling by adapting
    its running std, causing episode_reward_mean=0 throughout training.
    """
    env = env_cls(**cfg)
    env = NormalizeObservation(env)
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
