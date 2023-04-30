"""
Environment Class
"""
import math

import numpy as np

from helpers.box import Box


class BaseEnvironment:
    def __init__(self, gravity=1.62, mass=15000.0, initial_force=0.0):
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
                -35.0,  # dx/dt (lander)
                -35.0,  # dy/dt (lander)
                -2 * math.pi,  # theta (lander)
                -1.0,  # dtheta/dt (lander)
            ]
        ).astype(np.float32)
        high = np.array(
            [
                550,  # x (lander)
                50,  # y (lander)
                35.0,  # dx/dt (lander)
                35.0,  # dy/dt (lander)
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

        if initial_force == 0.0:
            self.enable_initial_force = False
        else:
            self.enable_initial_force = True
            self.initial_force = initial_force

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

        thruster_theta += theta

        v_x_new = v_x + ((thruster_power * math.sin(thruster_theta)) / self.mass) * dt
        v_y_new = v_y + ((-thruster_power * math.cos(thruster_theta)) / self.mass + self.gravity) * dt

        x_new = x + v_x_new * dt
        y_new = y + v_y_new * dt

        omega_new = omega + ((l * thruster_power * math.sin(thruster_theta)) / (2 * self.moment_of_inertia)) * dt
        theta_new = theta + omega_new * dt

        return np.array([x_new, y_new, v_x_new, v_y_new, theta_new, omega_new])

    def calculate_reward(self, state, extra_reward=0):
        reward = 0

        x_position_reward_weight = 10
        y_position_reward_weight = 10
        x_velocity_reward_weight = 10
        y_velocity_reward_weight = 10
        angle_reward_weight = 10

        # Compute reward based on current state
        x_position_reward = -x_position_reward_weight * abs(state[0] / self.observation_space.high[0])
        y_position_reward = -y_position_reward_weight * abs(state[1] / self.observation_space.low[1])
        x_velocity_reward = -x_velocity_reward_weight * abs(state[2] / self.observation_space.high[2])
        y_velocity_reward = -y_velocity_reward_weight * abs(state[3] / self.observation_space.low[3])
        angle_reward = -angle_reward_weight * abs(state[4] / self.observation_space.high[4])

        return x_position_reward + y_position_reward + x_velocity_reward + y_velocity_reward + angle_reward

    def check_termination(self, state):
        # Check if lander outside the observation space
        if not self.observation_space.is_bounded(state):
            return True, 0

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
        if self.enable_initial_force:
            force_angle = np.random.uniform(-math.pi / 2, math.pi / 2)
            force_magnitude = np.random.uniform(0, self.initial_force)
            initial_x_velocity = force_magnitude * math.cos(force_angle) / self.mass
            initial_y_velocity = force_magnitude * math.sin(force_angle) / self.mass
        else:
            initial_x_velocity = 0.0
            initial_y_velocity = 0.0

        initial_x_position = 0
        initial_y_position = -500.0
        return np.array([initial_x_position, initial_y_position, initial_x_velocity, initial_y_velocity, 0.0, 0.0])

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
