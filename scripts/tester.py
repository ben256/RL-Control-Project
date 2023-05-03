import json
import math
import os
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from environments.select_env import select_env
from algorithms.select_algo import select_algo

project_dir = "C:\\dev\\University\\MECH3890\\environment-model"
torch.manual_seed(42)  # What is the meaning of life the universe and everything?

testing_name = "test2"
models = ["training_10"]
notes = ""

test_scenarios = [
    {
        "test_name": "Test 1",
        "x_position": 0,
        "y_position": -500,
        "x_velocity": 0,
        "y_velocity": 0,
        "angle": 0,
        "angular_velocity": 0,
        "initial_force": None
    },
    {
        "test_name": "Test 2",
        "x_position": "-100,100",
        "y_position": -500,
        "x_velocity": 0,
        "y_velocity": 0,
        "angle": 0,
        "angular_velocity": 0,
        "initial_force": None
    },
    {
        "test_name": "Test 3",
        "x_position": 0,
        "y_position": -500,
        "x_velocity": "-5,5",
        "y_velocity": "-5,5",
        "angle": 0,
        "angular_velocity": 0,
        "initial_force": None
    },
    {
        "test_name": "Test 4",
        "x_position": "-100,100",
        "y_position": -500,
        "x_velocity": "-5,5",
        "y_velocity": "-5,5",
        "angle": 0,
        "angular_velocity": 0,
        "initial_force": None
    },
    {
        "test_name": "Test 5",
        "x_position": 0,
        "y_position": -500,
        "x_velocity": 0,
        "y_velocity": 0,
        "angle": "-0.25,0.25",
        "angular_velocity": "-0.1,1",
        "initial_force": None
    },
    {
        "test_name": "Test 6",
        "x_position": "-100,100",
        "y_position": -500,
        "x_velocity": 0,
        "y_velocity": 0,
        "angle": "-0.25,0.25",
        "angular_velocity": "-0.1,1",
        "initial_force": None
    }
]


max_state = np.array([
    550,  # x_position
    -550,  # y_position
    20,  # x_velocity
    -20,  # y_velocity
    math.pi,  # angle
    2.0  # angular_velocity
])


num_tests = 100  # Runs per test


def create_initial_states(state_dict, num_initial_states):
    states = np.zeros((num_initial_states, 6))  # Initialize a states array with shape (num_initial_states, 6)

    # Set the values in the state array based on the dictionary
    for i, key in enumerate(["x_position", "y_position", "x_velocity", "y_velocity", "angle", "angular_velocity"]):
        value = state_dict[key]
        if isinstance(value, str):
            low, high = map(float, value.split(","))
            states[:, i] = np.random.uniform(low, high, num_initial_states)
        else:
            states[:, i] = value

    return states


def distance_from_goal(state):
    goal_state = np.array([0, 0, 0, 0, 0, 0])
    normalised_state = state / max_state
    distance = np.linalg.norm(normalised_state - goal_state)
    return distance


if __name__ == "__main__":
    print("-=| Starting testing |=-")

    # Get the device
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda")
    print("Using {} device".format(device))

    # Create the model directory if it doesn't exist
    print("Creating training folder")
    testing_dir = os.path.join(project_dir, "tests", testing_name)
    if os.path.exists(testing_dir):
        print(f"Training folder '{testing_name}' already exists")
        user_input = input("Enter a new training name (or 'n' to overwrite): ")
        if user_input == "n":
            shutil.rmtree(testing_dir)
        else:
            training_name = user_input
        testing_dir = os.path.join(project_dir, "tests", testing_name)
    os.mkdir(testing_dir)

    # Create data storage
    print("Creating data storage")
    # test_data_df = pd.DataFrame(columns=["model", "test_name", "episode", "success", "time_steps", "distance_from_goal"])
    test_data = []

    for model in models:
        # Load parameters
        print(f"Loading model '{model}'")
        model_dir = os.path.join(project_dir, "models", model)
        model_save_dir = os.path.join(model_dir, "model")
        with open(os.path.join(model_dir, "parameters.json"), "r") as f:
            model_parameters = json.load(f)
        agent_parameters = model_parameters["agent_params"]
        env = select_env(model_parameters["env_name"], test=True)
        agent = select_algo("test_DDPG", alpha=agent_parameters["actor_lr (alpha)"], beta=agent_parameters["critic_lr (beta)"],
                            gamma=agent_parameters["discount factor (gamma)"], input_dims=env.observation_space.shape, tau=agent_parameters["tau"],
                            sigma=agent_parameters["noise sigma"], env=env, batch_size=agent_parameters["batch_size"],
                            layer1_size=agent_parameters["actor fc1 dims"], layer2_size=agent_parameters["actor fc2 dims"],
                            n_actions=env.action_space.shape[0], save_dir=model_save_dir, model_dir=model_save_dir)
        # Save dir is where the checkpoints get saved to
        # Model dir is where the model is loaded from

        agent.load_models()

        for test in test_scenarios:
            print(f"Running test '{test['test_name']}'")
            initial_states = create_initial_states(test, num_tests)

            for index, initial_state in tqdm(enumerate(initial_states), total=num_tests):
                # Reset the environment
                state = env.reset(initial_state=initial_state)
                score = 0
                terminated = False
                truncated = False
                info = {"success": 0}
                epoch_history = []

                while not (terminated or truncated):
                    epoch_history.append(state)
                    action = agent.choose_action(state, eval=True)
                    new_state, reward, terminated, truncated, info = env.step(action)
                    state = new_state
                    score += reward

                if truncated:
                    success = 0
                    steps = 401
                else:
                    success = info["success"]
                    if success:
                        steps = env.env_step
                    else:
                        steps = 401

                # Save the data
                test_data.append({
                    "model": model,
                    "test_name": test["test_name"],
                    "episode": index + 1,
                    "success": success,
                    "time_steps": env.env_step,
                    "distance_from_goal": distance_from_goal(state),
                    "cumulative_reward": score,
                    # "epoch_history": epoch_history
                })

    results_df = pd.DataFrame(test_data)
    results_df.to_csv(os.path.join(testing_dir, "results.csv"), index=False)
