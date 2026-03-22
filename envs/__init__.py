from envs.base_trading_env import BaseTradingEnv
from envs.shield_env import ShieldTradingEnv
from envs.builder_env import BuilderTradingEnv
from envs.hunter_env import HunterTradingEnv

from ray.tune.registry import register_env

# RLlib env registration — env_config dict is passed as kwargs to constructors
register_env("ShieldTradingEnv", lambda cfg: ShieldTradingEnv(**cfg))
register_env("BuilderTradingEnv", lambda cfg: BuilderTradingEnv(**cfg))
register_env("HunterTradingEnv", lambda cfg: HunterTradingEnv(**cfg))

ENV_MAP = {
    "shield": "ShieldTradingEnv",
    "builder": "BuilderTradingEnv",
    "hunter": "HunterTradingEnv",
}

__all__ = [
    "BaseTradingEnv",
    "ShieldTradingEnv",
    "BuilderTradingEnv",
    "HunterTradingEnv",
    "ENV_MAP",
]
