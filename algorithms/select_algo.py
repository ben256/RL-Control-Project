from algorithms.DDPG import Agent


def select_algo(algo_name, **kwargs):
    if algo_name == "DDPG":
        return Agent(**kwargs)
    else:
        raise ValueError("Unknown algorithm name")