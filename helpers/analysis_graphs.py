import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_reward_history(scores):
    fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True, dpi=300)
    for key, value in scores.items():
        x = np.arange(len(value))
        ax.plot(x, value, label=key)

    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    # plt.show()
    plt.savefig(f"C:\\dev\\University\\MECH3890\\environment-model\\tests\\graphs\\base_env_reward_history2.png")
    plt.close()


base_env_ic1_results = pd.read_csv("C:\\dev\\University\\MECH3890\\environment-model\\models\\base_env_ic1\\score_history.csv")
base_env_ic2_results = pd.read_csv("C:\\dev\\University\\MECH3890\\environment-model\\models\\base_env_ic2\\score_history.csv")
base_env_ic3_results = pd.read_csv("C:\\dev\\University\\MECH3890\\environment-model\\models\\base_env_ic3\\score_history.csv")

base_results = {
    "base_env_ic1": base_env_ic1_results["Average Score"],
    "base_env_ic2": base_env_ic2_results["Average Score"],
    "base_env_ic3": base_env_ic3_results["Average Score"]
}

plot_reward_history(base_results)
