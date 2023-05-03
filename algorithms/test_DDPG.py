import os
import time

import torch as T
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class OUActionNoise(object):
    # Used in actor network to add noise to actions

    def __init__(self, mu, sigma, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # allows for OUActionNoise()() syntax
        x = (
                self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt
                + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        # checks to see if x0 is None, if so, sets x_prev to mu
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    # Stores memory of past experiences for use in training
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size  # first available memory slot, wraps around if mem_cntr > mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)  # 1 if done, 0 if not done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


class CriticNetwork(nn.Module):  # nn.module allows access to train/eval methods and parameters to be optimized
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, model_dir, save_dir='model/'):
        super(CriticNetwork, self).__init__()  # calls nn.Module's __init__ method
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims  # number of nodes in first hidden layer
        self.fc2_dims = fc2_dims  # number of nodes in second hidden layer
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = save_dir
        self.model_dir = model_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')
        self.model_file = os.path.join(self.model_dir, name+'_best')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  #input dims, output dims
        # Constrains the initial weights of the network to narrow range of values to improve convergence
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])  # value to initialize weights of fc1
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)  # initialize weights of fc1 (tensor, min, max)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)  # initialize biases of fc1 (tensor, min, max)
        self.bn1 = nn.LayerNorm(self.fc1_dims)  # batch normalization layer, helps with convergence of model

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)  # eval and train methods need to be called to switch between modes

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)  # action value function
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)  # output layer, scaler value hence 1 output
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)  # parameters() from nn.Module
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # forward pass of the network
        # state is a tensor of shape (batch_size, 3, 96, 96)
        # action is a tensor of shape (batch_size, n_actions)
        # returns a tensor of shape (batch_size, 1)
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)  # Can be done before or after relu
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))  # Double relu function not the best
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)  # Creates state dict where keys are parameter names and values are parameter tensors

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.model_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)


class ActorNetwork(nn.Module):
    # Similar to critic network but with different structure, no action value function
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, model_dir, save_dir='model/'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = save_dir
        self.model_dir = model_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')
        self.model_file = os.path.join(self.model_dir, name+'_best')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)  # representation of the policy, real vector of shape n_actions, actual actions
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # state is a tensor of shape (batch_size, 3, 96, 96)
        # returns a tensor of shape (batch_size, n_actions)
        prob = self.fc1(state)
        prob = self.bn1(prob)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = self.bn2(prob)
        prob = F.relu(prob)
        mu = T.tanh(self.mu(prob))  # tanh squashes the output between -1 and 1, which is needed for the action space

        return mu

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.model_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, model_dir, sigma=0.15, gamma=0.99, n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64, save_dir='model/'):
        # Layer sizes are the same as in the paper
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # discount factor (how much less is a future reward valued), agent will prefer immediate reward over future reward
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        if model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = save_dir

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='actor', save_dir=save_dir, model_dir=model_dir)
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='target_actor', save_dir=save_dir, model_dir=model_dir)  # offpolicy hence target network
        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='critic', save_dir=save_dir, model_dir=model_dir)
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='target_critic', save_dir=save_dir, model_dir=model_dir)

        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=sigma)  # Mean of rewards over time, 0 for now

        self.update_network_parameters(tau=1)  # Solves moving target problem, copies weights from actor to target actor
        # tau is 1 so that all networks have the same weights at the beginning

    def choose_action(self, observation, eval=False):
        self.actor.eval()  # doesn't perform evaluation on the network, just tells torch not to calculate stats for batch norm and dropout
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)  # convert to tensor
        mu = self.actor.forward(observation).to(self.actor.device)  # forward pass through actor network
        if eval:
            self.actor.train()
            return mu.cpu().detach().numpy()
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)  # add noise to the action
        self.actor.train()  # back to training mode
        return mu_prime.cpu().detach().numpy()  # can't return tensor, convert to numpy array

    def remember(self, state, action, reward, new_state, done):
        # interface with replay buffer
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:  # if memory is not full, don't learn
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic.device)  # doesn't matter which device, all networks are on the same device
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)  # calculate target actions
        critic_value_ = self.target_critic.forward(new_state, target_actions)  # get estimates of actions from target actor network and forward to target critic network
        critic_value = self.critic.forward(state, action)  # what is estimate of states and actions actually encountered in replay buffer

        target = []
        for j in range(self.batch_size):
            # loop easier than vectorising
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])  # target is reward + gamma * critic value * done
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)  # reshape to (batch_size, 1)

        self.critic.train()  # back to training mode
        self.critic.optimizer.zero_grad()  # zero gradients so gradients from previous steps don't accumulate
        critic_loss = F.mse_loss(target, critic_value)  # mean squared error loss between target and critic value
        critic_loss.backward()  # backpropagate
        self.critic.optimizer.step()  # update weights

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)  # forward pass through actor network
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)  # loss is negative of critic value
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        # Tau allows the update of the target network to be more gradual, important for slow convergence
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()  # returns a generator
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)  # turning into dictionary makes iteration easier
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in critic_state_dict:  # iterate through network parameters and update target network parameters
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def save_best_models(self):
        print('... saving best models ...')
        self.actor.save_best()
        self.critic.save_best()
        self.target_actor.save_best()
        self.target_critic.save_best()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

    def get_params(self):
        return {
            'batch_size': self.batch_size,
            'actor_lr (alpha)': self.alpha,
            'critic_lr (beta)': self.beta,
            'discount factor (gamma)': self.gamma,
            'tau': self.tau,
            'noise sigma': self.noise.sigma,
            'noise theta': self.noise.theta,
            'noise dt': self.noise.dt,
            'actor input dims': self.actor.input_dims[0],
            'critic input dims': self.critic.input_dims[0],
            'actor fc1 dims': self.actor.fc1_dims,
            'actor fc2 dims': self.actor.fc2_dims,
            'critic fc1 dims': self.critic.fc1_dims,
            'critic fc2 dims': self.critic.fc2_dims,
            'memory max size': self.memory.mem_size,
        }

