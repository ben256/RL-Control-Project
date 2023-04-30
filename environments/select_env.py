from environments.base_environment import BaseEnvironment
from environments.gaussian_reward_environment import GaussianRewardEnvironment
from environments.reward_machine_environment import RewardMachineEnvironment
from environments.reward_shaping_environment import RewardShapingEnvironment


def select_env(env_name, **kwargs):
    if env_name == "BaseEnvironment":
        return BaseEnvironment(**kwargs)
    if env_name == "RewardShapingEnvironment":
        return RewardShapingEnvironment(**kwargs)
    if env_name == "GaussianRewardEnvironment":
        return GaussianRewardEnvironment(**kwargs)
    if env_name == "RewardMachineEnvironment":
        return RewardMachineEnvironment(**kwargs)
    else:
        raise ValueError("Unknown environment name")
