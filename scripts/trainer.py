import json
import os
import shutil
import time

import numpy as np
import torch

from environments.select_env import select_env
from algorithms.select_algo import select_algo
from helpers.graphs import plot_state_graph, plot_reward_graph

project_dir = "C:\\dev\\University\\MECH3890\\environment-model"
checkpoint_path = "C:\\dev\\University\\MECH3890\\environment-model\\models\\initial_force\\model"  # If loading from checkpoint set this to the checkpoint path
torch.manual_seed(42)  # What is the meaning of life the universe and everything?

training_name = "gr_env_ic1"
env_name = "GaussianRewardEnvironment"
algorithm_name = "JIT_DDPG"
notes = "JIT DDPG, gr_env_ic1"

load_from_checkpoint = False  # Whether to load from a checkpoint
average_reward = True  # Whether to average the reward over 100 episodes
initial_force = 0.0  # Initial force to apply to the lander
save_frequency = 100  # How often to save the model
num_epochs = 3001  # Number of epochs to train for
epoch = 0  # Current epoch
alpha = 0.00008  # Actor learning rate
beta = 0.0008  # Critic learning rate
gamma = 0.95  # Discount factor (closer to 1 = more future reward)
sigma = 0.20  # Noise factor
tau = 0.01  # Soft update factor
batch_size = 200  # Batch size
layer1_size = 400  # Size of first hidden layer
layer2_size = 300  # Size of second hidden layer

if __name__ == "__main__":
    print("-=| Starting training |=-")

    # Get the device
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda")
    print("Using {} device".format(device))

    # Create the model directory if it doesn't exist
    print("Creating training folder")
    training_dir = os.path.join(project_dir, "models", training_name)
    if os.path.exists(training_dir):
        print(f"Training folder '{training_name}' already exists")
        user_input = input("Enter a new training name (or 'n' to overwrite): ")
        if user_input == "n":
            shutil.rmtree(training_dir)
        else:
            training_name = user_input
        training_dir = os.path.join(project_dir, "models", training_name)
    os.mkdir(training_dir)

    # Create the checkpoint and final model directory
    model_checkpoint_dir = os.path.join(training_dir, "model")
    os.mkdir(model_checkpoint_dir)
    model_plots_dir = os.path.join(training_dir, "plots")
    os.mkdir(model_plots_dir)
    print(f"Created checkpoint and final directories")

    print("Creating environment and agent")
    env = select_env(env_name, initial_force=initial_force, initial_position=initial_position)

    agent = select_algo(algorithm_name, alpha=alpha, beta=beta, gamma=gamma, input_dims=env.observation_space.shape, tau=tau,
                        sigma=sigma, env=env, batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                        n_actions=env.action_space.shape[0], save_dir=model_checkpoint_dir, model_dir=checkpoint_path)

    if load_from_checkpoint:
        agent.load_models()

    # Save the parameters
    env_params = env.get_params()
    agent_params = agent.get_params()
    model_params = {
        "training_name": training_name,
        "env_name": env_name,
        "algorithm_name": algorithm_name,
        "env_params": env_params,
        "agent_params": agent_params,
        "notes": notes,
    }

    with open(os.path.join(training_dir, "parameters.json"), "w") as f:
        json.dump(model_params, f)

    best_score = env.reward_range[0]
    score_history = []
    state_history = []
    step_history = []

    start_time = time.time()

    for epoch in range(epoch, num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        state = env.reset()
        terminated = False
        truncated = False
        record = False
        if average_reward:
            score = []
        else:
            score = 0
        epoch_history = []
        agent.noise.reset()

        if epoch % save_frequency == 0 and epoch != 0:
            record = True

        while not (terminated or truncated):
            if record:
                epoch_history.append(state)

            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            agent.remember(state, action, reward, new_state, terminated)
            agent.learn()
            state = new_state
            if average_reward:
                score.append(reward)
            else:
                score += reward

        if average_reward:
            score = sum(score) / len(score)
            score_history.append(score)
        else:
            score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        step_history.append(env.env_step)

        if epoch > 30:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_best_models()

        if epoch % save_frequency == 0 and epoch != 0:
            # Plot Graphs
            plot_state_graph(epoch_history, epoch, model_plots_dir, env.observation_space.bounds())
            plot_reward_graph(score_history, model_plots_dir)

            # Save checkpoint
            agent.save_models()

            # Save score history to csv
            with open(os.path.join(training_dir, "score_history.csv"), "w") as f:
                f.write("Epoch,Score,Average Score,Steps\n")
                for i, score in enumerate(score_history):
                    f.write(f"{i},{score},{np.mean(score_history[max(0, i - 100):i + 1])},{step_history[i]}\n")

        print(f"Steps: {env.env_step}\tScore: {score:.5f}\tAverage Score: {avg_score:.5f}")

    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds")