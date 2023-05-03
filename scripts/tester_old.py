import json
import os
import shutil

import numpy as np
import torch

from environments.select_env import select_env
from algorithms.select_algo import select_algo
from helpers.graphs import plot_reward_graph_test, plot_state_graph_test

project_dir = "C:\\dev\\University\\MECH3890\\environment-model"
torch.manual_seed(42)  # What is the meaning of life the universe and everything?

run_name = "test"
env_name = "GaussianRewardEnvironment"
algorithm_name = "DDPG"
notes = "Gaussian Reward Environment with DDPG"
run_type = "test"
external_model = "RO_gaussian_3"

overwrite = True
load_from_checkpoint = True

num_tests = 100
epoch = 0
alpha = 0.00008
beta = 0.0008
gamma = 0.95
sigma = 0.2
tau = 0.001
batch_size = 200
layer1_size = 400
layer2_size = 300

if __name__ == "__main__":
    print("-=| Starting testing |=-")

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    # Create the model directory if it doesn't exist
    print("Creating training folder")
    training_dir = os.path.join(project_dir, "models", run_name)
    if os.path.exists(training_dir):
        print(f"Training folder '{run_name}' already exists")
        if overwrite:
            print("Overwriting existing training folder")
            shutil.rmtree(training_dir)
        else:
            user_input = input("Enter a new training name (or 'n' to overwrite): ")
            if user_input == "n":
                shutil.rmtree(training_dir)
            else:
                run_name = user_input
        training_dir = os.path.join(project_dir, "models", run_name)
    os.mkdir(training_dir)

    # Create the checkpoint and final model directory
    model_checkpoint_dir = os.path.join(training_dir, "model")
    os.mkdir(model_checkpoint_dir)
    model_plots_dir = os.path.join(training_dir, "plots")
    os.mkdir(model_plots_dir)
    print(f"Created checkpoint and final directories")

    # Access external model
    if run_type == "test":
        print("Loading external model")
        external_model_dir = os.path.join(project_dir, "models", external_model, "model")
    else:
        external_model_dir = None

    print("Creating environment and agent")
    env = select_env(env_name)

    agent = select_algo(algorithm_name, alpha=alpha, beta=beta, input_dims=env.observation_space.shape, tau=tau,
                        env=env, batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                        n_actions=env.action_space.shape[0], save_dir=model_checkpoint_dir, model_dir=external_model_dir)

    if run_type == "test":
        agent.load_models()

    # Save the parameters
    env_params = env.get_params()
    agent_params = agent.get_params()
    model_params = {
        "run_name": run_name,
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

    for test in range(num_tests):
        print('-' * 10)
        print('Epoch {}/{}'.format(test, num_tests - 1))

        state = env.reset()
        score = 0
        terminated = False
        truncated = False
        epoch_history = []
        sample = 0

        while not (terminated or truncated):
            epoch_history.append(state)
            action = agent.choose_action(state, eval=True)
            new_state, reward, terminated, truncated, info = env.step(action)
            state = new_state
            score += reward
            sample += 1

        state_history.append(epoch_history)
        score_history.append(score)

        print('score %.1f' % score, 'steps %d' % len(epoch_history))

    # plot_state_graph_test(state_history, epoch, model_plots_dir, env.observation_space.bounds())
    plot_reward_graph_test(score_history, model_plots_dir)

    average_score = np.mean(score_history)
    std_score = np.std(score_history)
    average_length = np.mean([len(x) for x in state_history])

    text_file = f"""Average score: {average_score}
Standard deviation: {std_score}
Average length: {average_length}"""

    with open(os.path.join(training_dir, "results.txt"), "w") as f:
        f.write(text_file)

    with open(os.path.join(training_dir, "score_history.txt"), "w") as f:
        f.write(str(score_history))

