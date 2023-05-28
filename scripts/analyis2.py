import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], color=color)


project_dir = "C:\\dev\\University\\MECH3890\\environment-model"
testing_name = "ic3"
testing_dir = os.path.join(project_dir, "tests", testing_name)

results_df = pd.read_csv(os.path.join(testing_dir, "results.csv"))

# Data Processing
test_names = results_df["test_name"].unique()
models = results_df["model"].unique()

average_time_steps = np.zeros((len(models), len(test_names)))
success_rate = np.zeros((len(models), len(test_names)))
average_distance_from_goal = np.zeros((len(models), len(test_names)))
time_steps_to_reach_goal_dist = np.zeros((len(models), len(test_names)), dtype=np.ndarray)
distance_from_goal_dist = np.zeros((len(models), len(test_names)), dtype=np.ndarray)
for i, model in enumerate(models):
    for j, test_name in enumerate(test_names):
        test_results = results_df.loc[(results_df['model'] == model) & (results_df['test_name'] == test_name)]

        # Average Time Steps to Reach Goal State
        test_results['max_steps'] = 401
        test_results['corrected_time_steps'] = np.where(test_results['success'], test_results['time_steps'], test_results['max_steps'])
        avg_time_step = test_results['corrected_time_steps'].mean()
        average_time_steps[i, j] = avg_time_step

        # Success Rate
        success_rate[i, j] = test_results['success'].mean()

        # Average Distance from Goal
        average_distance_from_goal[i, j] = test_results['distance_from_goal'].mean()

        # Time Steps to Reach Goal
        time_steps_to_reach_goal_dist[i, j] = test_results['corrected_time_steps'].values

        # Distance from Goal
        distance_from_goal_dist[i, j] = test_results['distance_from_goal'].values

# Graphs
# Average Time Steps to Reach Goal State
# X-axis: Test Names
# Y-axis: Average time steps to reach the goal state
# Separate lines/bars for each model

fig, ax = plt.subplots(3, 1, tight_layout=True, figsize=(10, 10))
bar_width = 0.2
x = np.arange(len(test_names))

# Plot bars for each model
for i, model in enumerate(models):
    ax[0].bar(x + i * bar_width, average_time_steps[i], width=bar_width, label=model)

# Customize the plot
# ax[0].set_xlabel('Test Names')
ax[0].set_ylabel('Average Time Steps to Reach Goal State')
ax[0].set_xticks(x + bar_width * (len(models) - 1) / 2)
ax[0].set_xticklabels(test_names)
ax[0].legend()
# plt.title('Average Time Steps to Reach Goal State for Different Models')

# Plot bars for each model
for i, model in enumerate(models):
    ax[2].bar(x + i * bar_width, success_rate[i], width=bar_width, label=model)

# Customize the plot
# ax[2].set_xlabel('Test Names')
ax[2].set_ylabel('Success Rate')
ax[2].set_xticks(x + bar_width * (len(models) - 1) / 2)
ax[2].set_xticklabels(test_names)
ax[2].legend()

# Plot bars for each model
for i, model in enumerate(models):
    ax[1].bar(x + i * bar_width, average_distance_from_goal[i], width=bar_width, label=model)

# Customize the plot
ax[1].set_xlabel('Test Names')
ax[1].set_ylabel('Average Distance from Goal State')
ax[1].set_xticks(x + bar_width * (len(models) - 1) / 2)
ax[1].set_xticklabels(test_names)
ax[1].legend()
# plt.title('Average Distance from Goal State for Different Models')

# Show the plot
plt.savefig(os.path.join(testing_dir, "combined_bar_charts.png"))
plt.close(fig)
