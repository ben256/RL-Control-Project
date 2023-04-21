from environments.base_environment import BaseEnvironment
from environments.reward_shaping_environment import RewardShapingEnvironment


def select_env(env_name):
    if env_name == "BaseEnvironment":
        return BaseEnvironment()
    if env_name == "RewardShapingEnvironment":
        return RewardShapingEnvironment()
    else:
        raise ValueError("Unknown environment name")
