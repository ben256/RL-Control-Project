from algorithms.DDPG import Agent as DDPGAgent
from algorithms.jit_compiler_DDPG import Agent as JIT_DDPGAgent


def select_algo(algo_name, **kwargs):
    if algo_name == "DDPG":
        return DDPGAgent(**kwargs)
    if algo_name == "JIT_DDPG":
        return JIT_DDPGAgent(**kwargs)
    else:
        raise ValueError("Unknown algorithm name")