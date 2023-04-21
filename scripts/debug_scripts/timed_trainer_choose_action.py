import json
import os
import shutil
import time

import numpy as np
import torch
import cProfile

from environments.select_env import select_env
from algorithms.select_algo import select_algo
from helpers.graphs import plot_state_graph, plot_reward_graph

project_dir = "C:/dev/University/MECH3890/environment-model"
checkpoint_path = "../../models/training_9/model"
torch.manual_seed(42)  # What is the meaning of life the universe and everything?

training_name = "test"
env_name = "BaseEnvironment"
algorithm_name = "JIT_DDPG"
notes = "continuing training_9"

save_frequency = 100
num_epochs = 4001
epoch = 2401
alpha = 0.00008
beta = 0.0008
gamma = 0.95
sigma = 0.2
tau = 0.001
batch_size = 200
layer1_size = 400
layer2_size = 300


if __name__ == "__main__":
    env = select_env(env_name)
    agent = select_algo(algorithm_name, alpha=alpha, beta=beta, gamma=gamma, input_dims=env.observation_space.shape, tau=tau,
                        sigma=sigma, env=env, batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                        n_actions=env.action_space.shape[0], model_dir=checkpoint_path)

    agent.load_models()

    state = [
        np.random.randint(-450, 450),
        np.random.randint(-450, -50),
        np.random.randint(-10, 10),
        np.random.randint(-10, 10),
        np.random.randint(-2, 2),
        np.random.randint(-1, 1),
    ]
    agent.noise.reset()

    total_time = 0
    for i in range(100):
        start_time = time.time()
        agent.choose_action(state)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / 100
    print("Average time: {}".format(average_time))
