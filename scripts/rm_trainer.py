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
checkpoint_path = "C:\\dev\\University\\MECH3890\\environment-model\\models\\RO_RM_1\\model"  # If loading from checkpoint set this to the checkpoint path
torch.manual_seed(42)  # What is the meaning of life the universe and everything?

training_name = "rm_env_ic1"
env_name = "RewardMachineEnvironment"
algorithm_name = "DDPG"
notes = "rm_env_ic1"
rm = "rm7"

load_from_checkpoint = False  # Whether to load from a checkpoint
average_reward = False  # Whether to average the reward over 100 episodes
initial_position = False  # Whether to set the initial position
initial_velocity = False  # Whether to set the initial velocity
save_frequency = 100  # How often to save the model
num_epochs = 10001  # Number of epochs to train for
epoch = 0  # Current epoch
alpha = 0.0001  # Actor learning rate
beta = 0.001  # Critic learning rate
gamma = 0.9  # Discount factor (closer to 1 = more future reward)
sigma = 0.25  # Noise factor
tau = 0.003  # Soft update factor
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
    rm_filename = "{}.txt".format(rm)
    rm_filepath = os.path.join(project_dir, "helpers/reward_machines/txt_files", rm_filename)
    env = select_env(env_name, rm_filepath=rm_filepath, initial_position=initial_position, initial_velocity=initial_velocity)

    agent = select_algo(algorithm_name, alpha=alpha, beta=beta, gamma=gamma, input_dims=env.observation_space.shape, tau=tau,
                        sigma=sigma, env=env, batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                        n_actions=env.action_space.shape[0], save_dir=model_checkpoint_dir, model_dir=model_checkpoint_dir)

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
        state_dict = env.reset()
        state = state_dict['features']
        terminated = False
        truncated = False
        record = False
        score = 0
        epoch_history = []
        agent.noise.reset()

        if epoch % save_frequency == 0 and epoch != 0:
            record = True

        while not (terminated or truncated):
            if record:
                epoch_history.append(state)

            action = agent.choose_action(state)
            new_state_dict, reward, terminated, truncated, info = env.step(action)
            new_state = new_state_dict['features']

            if env.add_crm:
                experiences = info["crm-experience"]
            else:
                if env.add_rs:
                    reward = info["rs-reward"]
                experiences = [(state, action, reward, new_state, terminated)]

            for _state, _action, _reward, _new_state, _done in experiences:
                _state = _state['features']
                _state.shape = state.shape
                _action.shape = action.shape
                _new_state = _new_state['features']
                _new_state.shape = new_state.shape
                _reward = np.array([_reward])
                _done = np.array([_done])
                agent.remember(_state, _action, _reward, _new_state, _done)

            agent.learn()
            state = new_state
            score += reward

        score_history.append(score)
        step_history.append(env.env_step)
        if info:
            if "rm_state_id" in info:
                state_history.append(info["rm_state_id"])
        avg_score = np.mean(score_history[-100:])

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
                f.write("Epoch,Score,Average Score,Steps,RM State\n")
                for i, score in enumerate(score_history):
                    f.write(f"{i},{score},{np.mean(score_history[max(0, i - 100):i + 1])},{step_history[i]},{state_history[i]}\n")


        print("Steps: {}\tScore: {:.5f}\tAverage Score: {:.5f}".format(env.env_step, score, avg_score))

    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds")