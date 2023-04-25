"""

Environment Class
"""
import math
from random import randint

import numpy as np

from helpers.box import Box
from helpers.reward_machines.reward_machine import RewardMachine


class BaseRewardMachineEnvironment:
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
        # self.terminated_step = 0

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
        return 0  # Reward comes from the Reward Machine

    def check_termination(self, state):
        # Check if lander outside the observation space
        if not self.observation_space.is_bounded(state):
            return True, 0  # + steps_reward
        else:
            return False, 0

        # # Check if lander inside the termination space
        # if self.termination_space.is_bounded(state):
        #     self.terminated_step += 1
        #
        #     if self.terminated_step > 50:
        #         print("Landed successfully!")
        #         return True, 0
        #     else:
        #         return False, 0
        #
        # else:
        #     self.terminated_step = 0
        #     return False, 0

    def check_truncation(self):
        # Check if the episode ended due to truncation
        if self.env_step >= 400:
            return True

    def get_events(self):
        x_position = self.state[0]
        y_position = self.state[1]
        x_velocity = self.state[2]
        y_velocity = self.state[3]
        angle = self.state[4]
        angular_velocity = self.state[5]
        events = ''

        # Check if lander is within a 100x100 box around (x=0, y=500)
        if -25 <= x_position <= 25 and -475 >= y_position >= -525 and -0.2 <= angle <= 0.2:
            events += 'b'

        # Check if lander is within a 100x100 box around (x=0, y=0)
        if -25 <= x_position <= 25 and -25 <= y_position <= 25 and -0.2 <= angle <= 0.2 and -1 <= y_velocity <= 1:
            events += 'c'

        return events

    def initial_state(self):
        initial_x = randint(-100, 100)
        initial_y = -500.0
        return np.array([initial_x, initial_y, 0.0, 0.0, 0.0, 0.0])

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


class RewardMachineEnvironmentWrapper:
    def __init__(self, env, rm_file, add_crm=True, add_rs=True, gamma=0.9, rs_gamma=0.9):
        self.env = env

        # Loading the reward machine
        self.rm_file = rm_file
        self.reward_machine = RewardMachine(rm_file)
        self.num_rm_states = len(self.reward_machine.get_states())

        # Computing one-hot encodings for the non-terminal RM states
        self.rm_state_features = {}
        for state_id in self.reward_machine.get_states():
            state_features = np.zeros(self.num_rm_states)
            state_features[len(self.rm_state_features)] = 1
            self.rm_state_features[state_id] = state_features
        self.rm_done_features = np.zeros(self.num_rm_states)  # for terminal RM states, we give as features an array of zeros

        # Initialize parameters for RewardMachineWrapper functionality
        self.add_crm = add_crm
        self.add_rs = add_rs
        if add_rs:
            self.reward_machine.add_reward_shaping(gamma, rs_gamma)
        self.valid_states = None

    def reset(self):
        self.state = self.env.reset()
        self.current_state_id = self.reward_machine.reset()
        self.valid_states = None

        # Adding the RM state to the observation
        return self.get_observation(self.state, self.current_state_id, False)

    def step(self, action):
        # RM state before executing the action
        state_id = self.current_state_id
        rm = self.reward_machine  # Needed to calculate CRM and RS without affecting the RM state

        # Executing the action in the base environment
        new_state, original_reward, terminated, truncated, info = self.env.step(action)

        # Getting the output of the detectors and saving information for generating counterfactual experiences
        true_propositions = self.env.get_events()
        self.crm_params = rm, self.state, action, new_state, terminated, true_propositions, info
        self.state = new_state

        # Update the RM state
        self.current_state_id, rm_reward, rm_done = self.reward_machine.step(self.current_state_id, true_propositions, info)
        if rm_done:
            print("RM done! Houston, we have landed!")
        # Returning the result of this action
        done = rm_done or terminated or truncated
        if done:
            info['done_reason'] = ['rm_done', 'env_done', 'truncated'][[rm_done, terminated, truncated].index(True)]
        rm_state = self.get_observation(new_state, self.current_state_id, done)

        # Adding crm if needed
        if self.add_crm:
            crm_experience = self._get_crm_experience(*self.crm_params)
            info["crm-experience"] = crm_experience
        elif self.add_rs:
            # Computing reward using reward shaping
            _, _, _, rs_terminated, rs_true_props, rs_info = self.crm_params
            _, rs_rm_rew, _ = rm.step(state_id, rs_true_props, rs_info, self.add_rs, rs_terminated)
            info["rs-reward"] = rs_rm_rew

        return rm_state, rm_reward, terminated or rm_done, truncated, info

    def get_observation(self, next_obs, state_id, done):
        rm_features = self.rm_done_features if done else self.rm_state_features[state_id]
        rm_state = {'features': next_obs, 'rm-state': rm_features}
        return rm_state

    def _get_crm_experience(self, rm, state, action, next_state, env_done, true_propositions, info):
        """
        Returns a list of counterfactual experiences generated per each RM state.
        Format: [..., (state, action, r, new_state, done), ...]
        """
        reachable_states = set()
        experiences = []
        for state_id in rm.get_states():
            experience, next_rm_state = self._get_rm_experience(rm, state_id, state, action, next_state, env_done, true_propositions, info)
            reachable_states.add(next_rm_state)
            if self.valid_states is None or state_id in self.valid_states:
                # We only add experience that are possible (i.e., it is possible to reach state state_id given the previous experience)
                experiences.append(experience)

        self.valid_states = reachable_states
        return experiences

    def _get_rm_experience(self, rm, state_id, state, action, next_state, env_done, true_propositions, info):
        rm_state = self.get_observation(state, state_id, False)
        next_state_id, rm_rew, rm_done = rm.step(state_id, true_propositions, info, self.add_rs, env_done)
        done = rm_done or env_done
        rm_next_state = self.get_observation(next_state, next_state_id, done)
        return (rm_state, action, rm_rew, rm_next_state, done), next_state_id

    def __getattr__(self, name):
        return getattr(self.env, name)


# class RewardMachineEnvironmentWrapper:
#     def __init__(self, base_env, rm_filepath, add_crm, add_rs, gamma, rs_gamma):
#         self.env = base_env
#         self.rm = RewardMachine(rm_filepath)
#         self.rm_state = self.rm.u0
#         self.add_crm = add_crm
#         self.add_rs = add_rs
#         if add_rs:
#             self.rm.add_reward_shaping(gamma, rs_gamma)
#
#     def reset(self):
#         self.valid_states = None
#         return self.env.reset()
#
#
#     def step(self, action):
#         new_state, original_reward, terminated, truncated, info = self.env.step(action)
#         self.state = new_state
#         events = self.get_events()
#         next_rm_state, rm_reward, rm_done = self.rm.step(self.rm_state, events, info, add_rs=self.add_rs, env_done=terminated)
#         self.rm_state = next_rm_state
#
#         if self.add_crm:
#             crm_experience = self._get_crm_experience(action, terminated, events, info)
#             info["crm-experience"] = crm_experience
#         elif self.add_rs:
#             info["rs-reward"] = rm_reward
#
#         total_reward = original_reward + rm_reward
#         return new_state, total_reward, terminated, truncated, info

    # def step(self, action):
    #     u_id = self.rm.get_current_state()
    #     next_obs, original_reward, env_done, info = self.env.step(action)
    #     events = self.get_events()
    #     next_u_id, rm_rew, rm_done = self.rm.step(u_id, events, info, self.add_rs, env_done)
    #     self.rm.set_current_state(next_u_id)
    #
    #     if self.add_crm:
    #         crm_experience = self._get_crm_experience(obs, action, next_obs, env_done, events, info)
    #         info["crm-experience"] = crm_experience
    #     elif self.add_rs:
    #         _, rs_rm_rew, _ = self.rm.step(u_id, events, info, self.add_rs, env_done)
    #         info["rs-reward"] = rs_rm_rew
    #
    #     done = rm_done or env_done
    #     return next_obs, rm_rew, done, info
    #
    # def _get_crm_experience(self, obs, action, next_obs, env_done, events, info):
    #     exp, next_u = self._get_rm_experience(self.rm, u_id, obs, action, next_obs, env_done, events, info)
    #     return exp
    #
    # def _get_rm_experience(self, rm, u_id, obs, action, next_obs, env_done, events, info):
    #     rm_obs = self.get_observation(obs, u_id, False)
    #     next_u_id, rm_rew, rm_done = rm.step(u_id, events, info, self.add_rs, env_done)
    #     done = rm_done or env_done
    #     rm_next_obs = self.get_observation(next_obs, next_u_id, done)
    #     return (rm_obs, action, rm_rew, rm_next_obs, done), next_u_id
    #
    # def get_observation(self, obs, u_id, done):
    #     # Adjust this method according to your specific environment and reward machine requirements
    #     rm_obs = np.concatenate((obs, [u_id]))
    #
    #     if done:
    #         # If the episode is done, you can return a specific terminal observation or perform any
    #         # necessary processing here.
    #         pass
    #
    #     return rm_obs
    #
    # # Include the `get_events()` method here, as provided in a previous response
    # def get_events(self):
    #     x_position = self.env.state[0]
    #     y_position = self.env.state[1]
    #     x_velocity = self.env.state[2]
    #     y_velocity = self.env.state[3]
    #     angle = self.env.state[4]
    #     angular_velocity = self.env.state[5]
    #     events = ''
    #
    #     # Check if lander is within a 100x100 box around (x=0, y=500)
    #     if -50 <= x_position <= 50 and 450 <= y_position <= 550 and -0.2 <= angle <= 0.2:
    #         events += 'b'
    #
    #     # Check if lander is within a 100x100 box around (x=0, y=0)
    #     if -50 <= x_position <= 50 and -50 <= y_position <= 50 and -0.2 <= angle <= 0.2 and -1 <= y_velocity <= 1:
    #         events += 'c'
    #
    #     return events


class RewardMachineEnvironment(RewardMachineEnvironmentWrapper):
    def __init__(self, **kwargs):
        base_env = BaseRewardMachineEnvironment(**kwargs)
        rm_files = ['../helpers/reward_machines/txt_files/t1.txt']
        super().__init__(base_env, rm_files[0])
