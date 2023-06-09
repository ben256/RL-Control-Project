import json
import os
import shutil
import time

import numpy as np
import torch
import argparse

import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))

from environments.select_env import select_env
from algorithms.select_algo import select_algo
from helpers.graphs import plot_state_graph, plot_reward_graph

if __name__ == "__main__":
    start_time = time.time()
    torch.manual_seed(42)  # What is the meaning of life the universe and everything?

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_name', type=str, default='training_1')
    parser.add_argument('--env_name', type=str, default='BaseEnvironment')
    parser.add_argument('--algorithm_name', type=str, default='DDPG')
    parser.add_argument('--notes', type=str, default='baseline environment with DDPG')
    parser.add_argument('--rm', type=str, default='rm1')
    parser.add_argument('--initial_velocity', type=bool, default=False)
    parser.add_argument('--initial_position', type=bool, default=False)
    parser.add_argument('--save_frequency', type=int, default=100)
    parser.add_argument('--average_reward', type=bool, default=False)
    parser.add_argument('--num_epochs', type=int, default=1001)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--sigma', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--layer1_size', type=int, default=400)
    parser.add_argument('--layer2_size', type=int, default=300)
    args = parser.parse_args()

    training_name = args.training_name
    env_name = args.env_name
    algorithm_name = args.algorithm_name
    notes = args.notes
    rm = args.rm
    initial_velocity = args.initial_velocity
    initial_position = args.initial_position
    save_frequency = args.save_frequency
    average_reward = args.average_reward
    num_epochs = args.num_epochs
    epoch = args.epoch
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    sigma = args.sigma
    tau = args.tau
    batch_size = args.batch_size
    layer1_size = args.layer1_size
    layer2_size = args.layer2_size

    print("-=| Starting training |=-")

    # Get the device
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda")
    print("Using {} device".format(device))

    # Create the model directory if it doesn't exist
    print("Creating training folder")
    project_dir = os.path.abspath(os.getcwd())
    training_dir = os.path.join(project_dir, "models", training_name)
    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)
    os.mkdir(training_dir)

    model_checkpoint_dir = os.path.join(training_dir, "model")
    os.mkdir(model_checkpoint_dir)
    model_plots_dir = os.path.join(training_dir, "plots")
    os.mkdir(model_plots_dir)
    print("Created checkpoint and final directories")

    print("Creating environment and agent")
    env = select_env(env_name, initial_velocity=initial_velocity, initial_position=initial_position)

    agent = select_algo(algorithm_name, alpha=alpha, beta=beta, gamma=gamma, input_dims=env.observation_space.shape, tau=tau,
                        sigma=sigma, env=env, batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                        n_actions=env.action_space.shape[0], save_dir=model_checkpoint_dir, model_dir=model_checkpoint_dir)

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
