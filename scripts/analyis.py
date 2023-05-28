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
testing_name = "ic1"
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

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
bar_width = 0.25
x = np.arange(len(test_names))

# Plot bars for each model
for i, model in enumerate(models):
    ax.bar(x + i * bar_width, average_time_steps[i], width=bar_width, label=model)

# Customize the plot
ax.set_xlabel('Test Names')
ax.set_ylabel('Average Time Steps to Reach Goal State')
ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
ax.set_xticklabels(test_names)
ax.legend()
# plt.title('Average Time Steps to Reach Goal State for Different Models')

# Show the plot
plt.savefig(os.path.join(testing_dir, "average_time_steps.png"))
plt.close(fig)

# Success Rate:
# X-axis: Test Names
# Y-axis: Success rate (percentage of successful episodes out of 100)
# Separate lines/bars for each model

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
bar_width = 0.25
x = np.arange(len(test_names))

# Plot bars for each model
for i, model in enumerate(models):
    ax.bar(x + i * bar_width, success_rate[i], width=bar_width, label=model)

# Customize the plot
ax.set_xlabel('Test Names')
ax.set_ylabel('Success Rate')
ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
ax.set_xticklabels(test_names)
ax.legend()
# plt.title('Success Rate for Different Models')

# Show the plot
plt.savefig(os.path.join(testing_dir, "success_rate.png"))
plt.close(fig)

# Average Distance from Goal State (for unsuccessful episodes):
# X-axis: Test Names
# Y-axis: Average distance from the goal state for unsuccessful episodes
# Separate lines/bars for each model

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
bar_width = 0.25
x = np.arange(len(test_names))

# Plot bars for each model
for i, model in enumerate(models):
    ax.bar(x + i * bar_width, average_distance_from_goal[i], width=bar_width, label=model)

# Customize the plot
ax.set_xlabel('Test Names')
ax.set_ylabel('Average Distance from Goal State')
ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
ax.set_xticklabels(test_names)
ax.legend()
# plt.title('Average Distance from Goal State for Different Models')

# Show the plot
plt.savefig(os.path.join(testing_dir, "average_distance_from_goal.png"))
plt.close(fig)

# Distribution of Time Steps to Reach Goal State (boxplot or violin plot):
# X-axis: Test Names
# Y-axis: Time steps to reach the goal state
# Separate boxplots/violin plots for each model

# Create the box plot
fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))

for i, test_name in enumerate(test_names):
    test_data = time_steps_to_reach_goal_dist[:, i]
    ax.boxplot(test_data, positions=[i + 1 + 0.25 * j for j in range(len(models))], widths=0.2)

ax.set_xticks(np.arange(len(test_names)) + 1)
ax.set_xticklabels(test_names)

# Customize the plot
ax.set_xlabel('Models')
ax.set_ylabel('Time Steps to Reach Goal State')
# plt.title('Distribution of Time Steps to Reach Goal State')

# Show the plot
plt.savefig(os.path.join(testing_dir, "time_steps_to_reach_goal.png"))
plt.close(fig)

# Distribution of Final Distance from Goal State (boxplot or violin plot):
# X-axis: Test Names
# Y-axis: Final distance from the goal state (successful and unsuccessful episodes)
# Separate boxplots/violin plots for each model

# Create the box plot
fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5), sharey=True)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
model_position_offset = np.linspace(0.2*len(models), -0.2*len(models), len(models))

for i, model in enumerate(models):
    test_data = distance_from_goal_dist[i]
    bp = ax.boxplot(test_data, positions=np.array(range(len(test_data)))*2.0-model_position_offset[i], widths=0.4, flierprops=dict(marker='x', markeredgecolor=colors[i], markersize=7))
    set_box_color(bp, colors[i])
    ax.plot([], c=colors[i], label=model)

ax.set_xticks(np.arange(len(test_names))*2.0)
ax.set_xticklabels(test_names)
ax.legend()

# Customize the plot
ax.set_xlabel('Test Names')
ax.set_ylabel('Final Distance from Goal State')
# plt.title('Distribution of Final Distance from Goal State')

# Show the plot
plt.savefig(os.path.join(testing_dir, "distance_from_goal.png"))
plt.close(fig)
