import numpy as np
import matplotlib
from matplotlib import pyplot as plt, patches
from matplotlib.collections import LineCollection
matplotlib.use('Agg')

state_dict = {
    0: "X Position",
    1: "Y Position",
    2: "X Velocity",
    3: "Y Velocity",
    4: "Angle",
    5: "Angular Velocity"
}


def plot_reward_graph(scores, save_dir):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    x = np.arange(len(scores))
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(f"{save_dir}/reward_history.png")
    plt.close()


def plot_reward_graph_test(scores, save_dir):
    # running_avg = np.zeros(len(scores))
    # for i in range(len(running_avg)):
    #     running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    x = np.arange(len(scores))
    plt.bar(x, scores)
    plt.title('Running average of previous 100 scores')
    plt.savefig(f"{save_dir}/reward_history.png")
    plt.close()

def plot_state_graph(state_history, epoch, save_dir, bounds=None):
    state_history = np.transpose(state_history)

    # Plot position and velocity graphs
    for index, (i1, i2) in enumerate([(0, 1), (2, 3)]):  # x, y, v_x, v_y
        x = state_history[i1]
        y = state_history[i2]
        t = np.arange(len(x))

        fig, ax = plt.subplots(tight_layout=True)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(min(t), max(t))
        lc = LineCollection(segments, cmap='viridis_r', norm=norm)

        lc.set_array(t)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax, label='Time')

        ax.set_title(f"State, Epoch: {epoch}")
        ax.set_xlabel(state_dict[i1])
        ax.set_ylabel(state_dict[i2])

        ax.grid(True)

        x_pad = abs(0.1 * (bounds[1][i1] - bounds[0][i1]))
        y_pad = abs(0.1 * (bounds[1][i2] - bounds[0][i2]))

        if bounds:
            ax.set_xlim(bounds[0][i1]-x_pad, bounds[1][i1]+x_pad)
            ax.set_ylim(bounds[0][i2]-y_pad, bounds[1][i2]+y_pad)

        bounds_box = patches.Rectangle((bounds[0][i1], bounds[0][i2]), bounds[1][i1] - bounds[0][i1], bounds[1][i2] - bounds[0][i2], linewidth=1, linestyle='--', edgecolor='#fde725', facecolor='none', label="Bounds")
        ax.add_patch(bounds_box)

        if index == 0:
            plt.gca().invert_yaxis()

        ax.plot(0, 0, 'x', label="Goal", c="#440154")
        ax.legend()

        plt.savefig(f"{save_dir}/Epoch {epoch} {state_dict[i1].split()[1]}.png", dpi=200)
        plt.close()

    # Plot angle and angular velocity graphs
    fig, ax = plt.subplots(2, 1, tight_layout=True)
    fig.suptitle(f"State, Epoch: {epoch}")

    for i in [4, 5]:  # theta, omega
        ax[i-4].plot(state_history[i], c="#21918c")
        if i == 5:
            ax[i-4].set_xlabel("Time")
        ax[i-4].set_ylabel(state_dict[i])

        y_pad = abs(0.1 * (bounds[1][i] - bounds[0][i]))
        ax[i-4].set_ylim(bounds[0][i]-y_pad, bounds[1][i]+y_pad)

        ax[i-4].grid(True)
        ax[i-4].axhline(y=bounds[0][i], color='#fde725', linewidth=1, linestyle='--', label="Bounds")
        ax[i-4].axhline(y=bounds[1][i], color='#fde725', linewidth=1, linestyle='--')
        ax[i-4].axhline(y=0, color='#440154', linewidth=1, linestyle='--', label="Goal")
        ax[i-4].legend()

    plt.savefig(f"{save_dir}/Epoch {epoch} Angle.png", dpi=200)
    plt.close()


def plot_state_graph_test(state_history, epoch, save_dir, bounds=None):

    # Plot position and velocity graphs
    for index, (i1, i2) in enumerate([(0, 1), (2, 3)]):  # x, y, v_x, v_y
        fig, ax = plt.subplots(tight_layout=True)

        for state_index, state in enumerate(state_history):
            state = np.transpose(state)
            x = state[i1]
            y = state[i2]
            t = np.arange(len(x))
            ax.plot(x, y, label=f"Episode {state_index}")

            # points = np.array([x, y]).T.reshape(-1, 1, 2)
            # segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # norm = plt.Normalize(min(t), max(t))
            # lc = LineCollection(segments, cmap='viridis_r', norm=norm)
            #
            # lc.set_array(t)
            # lc.set_linewidth(2)
            # line = ax.add_collection(lc)
            # fig.colorbar(line, ax=ax, label='Time')

        ax.set_title(f"State, Epoch: {epoch}")
        ax.set_xlabel(state_dict[i1])
        ax.set_ylabel(state_dict[i2])

        ax.grid(True)

        x_pad = abs(0.1 * (bounds[1][i1] - bounds[0][i1]))
        y_pad = abs(0.1 * (bounds[1][i2] - bounds[0][i2]))

        if bounds:
            ax.set_xlim(bounds[0][i1]-x_pad, bounds[1][i1]+x_pad)
            ax.set_ylim(bounds[0][i2]-y_pad, bounds[1][i2]+y_pad)

        bounds_box = patches.Rectangle((bounds[0][i1], bounds[0][i2]), bounds[1][i1] - bounds[0][i1], bounds[1][i2] - bounds[0][i2], linewidth=1, linestyle='--', edgecolor='#fde725', facecolor='none', label="Bounds")
        ax.add_patch(bounds_box)

        if index == 0:
            plt.gca().invert_yaxis()

        ax.plot(0, 0, 'x', label="Goal", c="#440154")
        # ax.legend()

        plt.savefig(f"{save_dir}/Epoch {epoch} {state_dict[i1].split()[1]}.png", dpi=200)
        plt.close()

    # Plot angle and angular velocity graphs
    fig, ax = plt.subplots(2, 1, tight_layout=True)
    fig.suptitle(f"State, Epoch: {epoch}")

    for i in [4, 5]:  # theta, omega

        for state_index, state in enumerate(state_history):
            state = np.transpose(state)
            ax[i-4].plot(state[i], label=f"Episode {state_index}")

        if i == 5:
            ax[i-4].set_xlabel("Time")
        ax[i-4].set_ylabel(state_dict[i])

        y_pad = abs(0.1 * (bounds[1][i] - bounds[0][i]))
        ax[i-4].set_ylim(bounds[0][i]-y_pad, bounds[1][i]+y_pad)

        ax[i-4].grid(True)
        ax[i-4].axhline(y=bounds[0][i], color='#fde725', linewidth=1, linestyle='--', label="Bounds")
        ax[i-4].axhline(y=bounds[1][i], color='#fde725', linewidth=1, linestyle='--')
        ax[i-4].axhline(y=0, color='#440154', linewidth=1, linestyle='--', label="Goal")
        # ax[i-4].legend()

    plt.savefig(f"{save_dir}/Epoch {epoch} Angle.png", dpi=200)
    plt.close()

