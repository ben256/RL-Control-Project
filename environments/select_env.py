from environments.base_environment import BaseEnvironment
from environments.gaussian_reward_environment import GaussianRewardEnvironment
from environments.reward_machine_environment import RewardMachineEnvironment
from environments.reward_shaping_environment import RewardShapingEnvironment


def select_env(env_name):
    if env_name == "BaseEnvironment":
        return BaseEnvironment()
    if env_name == "RewardShapingEnvironment":
        return RewardShapingEnvironment()
    if env_name == "GaussianRewardEnvironment":
        return GaussianRewardEnvironment()
    if env_name == "RewardMachineEnvironment":
        return RewardMachineEnvironment()
    else:
        raise ValueError("Unknown environment name")
