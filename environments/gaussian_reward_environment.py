"""
Environment Class
"""
import math

import numpy as np

from helpers.box import Box


class GaussianRewardEnvironment:
    def __init__(self, gravity=1.62, mass=15000.0):
        # Initialize the state, and other parameters
        self.gravity = gravity
        self.mass = mass  # kg
        self.moment_of_inertia = 80000.0  # kg m^2

        self.max_thrust = 30000.0  # N
        self.min_thrust = 0  # N

        self.max_thruster_angle = 0.5  # rad
        self.min_thruster_angle = -0.5  # rad

        # Observation space
        low = np.array(
            [
                -550,  # x (lander)
                -550,  # y (lander)
                -25.0,  # dx/dt (lander)
                -25.0,  # dy/dt (lander)
                -2 * math.pi,  # theta (lander)
                -1.0,  # dtheta/dt (lander)
            ]
        ).astype(np.float32)
        high = np.array(
            [
                550,  # x (lander)
                50,  # y (lander)
                25.0,  # dx/dt (lander)
                25.0,  # dy/dt (lander)
                2 * math.pi,  # theta (lander)
                1.0,  # dtheta/dt (lander)
            ]
        ).astype(np.float32)
        self.observation_space = Box(low, high)

        # Termination Space
        low = np.array(
            [
                -50,  # x (lander)
                -50,  # y (lander)
                -1.0,  # dx/dt (lander)
                -1.0,  # dy/dt (lander)
                -math.pi / 6,  # theta (lander)
                -1.0,  # dtheta/dt (lander)
            ]
        ).astype(np.float32)
        high = np.array(
            [
                50,  # x (lander)
                50,  # y (lander)
                1.0,  # dx/dt (lander)
                1.0,  # dy/dt (lander)
                math.pi / 6,  # theta (lander)
                1.0,  # dtheta/dt (lander)
            ]
        ).astype(np.float32)
        self.termination_space = Box(low, high)

        high = np.array([
            50,  # x (lander)
            50,  # y (lander)
            None,  # dx/dt (lander)
            None,  # dy/dt (lander)
            None,  # theta (lander)
            None,  # dtheta/dt (lander)
        ]).astype(np.float32)
        low = np.array([
            -50,  # x (lander)
            -100,  # y (lander)
            None,  # dx/dt (lander)
            None,  # dy/dt (lander)
            None,  # theta (lander)
            None,  # dtheta/dt (lander)
        ]).astype(np.float32)
        self.landing_space = Box(low, high)

        # Action is two floats [thruster angle, thruster power]
        # Thruster angle: -1..+1 angle from min_thruster_angle to max_thruster_angle
        # Thruster Power: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power
        self.action_space = Box(-1, +1, (2,))

        self.state = self.initial_state
        self.terminated = False
        self.truncated = False
        self.prev_shaping = None
        self.env_step = 0
        self.terminated_step = 0

        self.reward_range = (-float("inf"), float("inf"))

    def reset(self):
        # Reset the environment to its initial state
        self.state = self.initial_state()
        self.prev_shaping = None
        self.env_step = 0
        self.terminated = False
        self.truncated = False
        return self.step(np.array([0.0, 0.0]))[0]

    def step(self, action):
        # Update the state of the environment based on the actions taken by the agent
        new_state = self.update_state(self.state, action)

        self.terminated, extra_reward = self.check_termination(new_state)
        self.truncated = self.check_truncation()
        self.state = new_state
        reward = self.calculate_reward(new_state, extra_reward)
        self.env_step += 1

        return new_state, reward, self.terminated, self.truncated, {}

    def update_state(self, state, action, dt=1, l=1):
        # Update the state of the environment based on the actions taken by the agent
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        thruster_power = (np.clip(action[1], 0.0, 1.0) + 1.0) * 0.5 * self.max_thrust  # 0.5..1.0
        thruster_theta = (action[0] + 1) / 2 * (self.max_thruster_angle - self.min_thruster_angle) + self.min_thruster_angle

        x, y, v_x, v_y, theta, omega = state

        v_x_new = v_x + ((thruster_power * math.sin(thruster_theta)) / self.mass) * dt
        v_y_new = v_y + ((-thruster_power * math.cos(thruster_theta)) / self.mass + self.gravity) * dt

        x_new = x + v_x_new * dt
        y_new = y + v_y_new * dt

        omega_new = omega + ((l * thruster_power * math.sin(thruster_theta)) / (2 * self.moment_of_inertia)) * dt
        theta_new = theta + omega_new * dt

        return np.array([x_new, y_new, v_x_new, v_y_new, theta_new, omega_new])

    def calculate_reward(self, state, extra_reward=0):
        beta = 10  # Scaling factor
        sigma = 0.5  # Standard deviation
        goal_state = np.array([0, 0, 0, 0, 0, 0])  # Goal state for position, velocities, and angle

        x_position_reward_weight = 3
        y_position_reward_weight = 1
        x_velocity_reward_weight = 1
        y_velocity_reward_weight = 1
        theta_reward_weight = 1
        omega_reward_weight = 1

        x_position_normalised = x_position_reward_weight * abs(state[0] / self.observation_space.high[0])
        y_position_normalised = y_position_reward_weight * abs(state[1] / self.observation_space.low[1])
        x_velocity_normalised = x_velocity_reward_weight * abs(state[2] / self.observation_space.high[2])
        y_velocity_normalised = y_velocity_reward_weight * abs(state[3] / self.observation_space.low[3])
        theta_normalised = theta_reward_weight * abs(state[4] / self.observation_space.high[4])
        omega_normalised = omega_reward_weight * abs(state[5] / self.observation_space.low[5])

        # Extract the relevant state components: x_position, y_position, x_velocity, y_velocity, and angle
        current_state = np.array([x_position_normalised, y_position_normalised, x_velocity_normalised, y_velocity_normalised, theta_normalised, omega_normalised])

        # Calculate the Euclidean distance between the current state and the goal state
        distance = np.linalg.norm(current_state[:5] - goal_state[:5])

        # if self.landing_space.is_bounded(state):
        #     distance = np.linalg.norm(current_state - goal_state)
        # else:
        #     distance = np.linalg.norm(current_state[:2] - goal_state[:2])

        # Compute the Gaussian-inspired reward function
        exponent_term = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        reward = beta * exponent_term + extra_reward

        return reward

    def check_termination(self, state):
        # Check if lander outside the observation space
        if not self.observation_space.is_bounded(state):
            # steps_reward = -3 * abs(min(0, self.env_step - 30))

            return True, 0  # + steps_reward

        # Check if lander inside the termination space
        if self.termination_space.is_bounded(state):
            self.terminated_step += 1

            if self.terminated_step > 10:
                print("Landed successfully!")
                return True, 0
            else:
                print("In termination space!")
                return False, 0

        else:
            self.terminated_step = 0
            return False, 0

    def check_truncation(self):
        # Check if the episode ended due to truncation
        if self.env_step >= 400:
            return True

    def initial_state(self):
        initial_x = np.random.choice([-400, -300, -200, 200, 300, 400])
        # initial_x = 200
        # initial_y = -500.0
        initial_y = -500.0
        return np.array([initial_x, initial_y, 0.0, 0.0, 0.0, 0.0])

    def render(self):
        # Render the environment (optional)
        pass

    def get_params(self):
        # Export the parameters of the environment (optional)
        return {
            "gravity": self.gravity,
            "mass": self.mass,
            "moment_of_inertia": self.moment_of_inertia,
            "max_thrust": self.max_thrust,
            "min_thruster_angle": self.min_thruster_angle,
            "max_thruster_angle": self.max_thruster_angle,
            "observation_space_low": self.observation_space.low.tolist(),
            "observation_space_high": self.observation_space.high.tolist(),
            "termination_space_low": self.termination_space.low.tolist(),
            "termination_space_high": self.termination_space.high.tolist(),
            "action_space_low": self.action_space.low,
            "action_space_high": self.action_space.high,
        }
