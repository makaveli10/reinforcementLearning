import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

from q_networks import DQNEstimator
from replay_memory import ReplayMemory


class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # visualising training
        self.render = False
        self.load_model = False
        
        # hyperparams for estimator
        self.gamma = 0.95
        self.lr = 0.001
        self.replay_memory_size = 50000
        self.epsilon = 1.0
        self.min_epsilon = 0.000001
        self.explore_step = 3000
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / self.explore_step
        self.batch_size = 32
        self.replay_memory_init_size = 500
        self.update_target_model_every = 1000
        
        # Replay Memory
        self.memory = ReplayMemory(self.replay_memory_size)
        
        # create estimator and target estimators
        self.q_estimator = DQNEstimator(state_size, action_size)
        self.target_estimator = DQNEstimator(state_size, action_size)
        self.optimizer = optim.Adam(self.q_estimator.parameters(), lr=self.lr)
        
        # initialize target estimator
        # TODO: copy q_estimator weights to target model
        
        if self.load_model:
            # TODO: Load saved Q estimator
            pass
        
    def update_target_estimator(self):
        self.target_estimator.load_state_dict(self.q_estimator.state_dict())
        
    def get_action(self, state):
        # random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:   # greedy action
            state = Variable(torch.from_numpy(state)).float()
            q_values = self.q_estimator(state)
            _, best_action = torch.max(q_values, dim=1)
            return int(best_action)
    
    def train_network(self):
        # epsilon decay
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay

        # fetch samples
        samples = self.memory.sample(self.batch_size)
        samples = np.array(samples).transpose()
        
        # create batches of states, actions, rewards, next_states, done
        # stack all the states
        states = np.vstack(samples[0])
        actions = torch.LongTensor(list(samples[1]))
        rewards = torch.FloatTensor(list(samples[2]))
        next_states = Variable(torch.FloatTensor(np.vstack(samples[3])))
        is_dones = samples[4]
        
        is_dones = torch.FloatTensor(is_dones.astype(int))
        
        # forward propagation Q_network for current states
        states = torch.Tensor(states)
        states = Variable(states).float()
        preds = self.q_estimator(states)
        
        # onehot encoding actions
        actions_one_hot = F.one_hot(actions, num_classes=self.action_size)
        actions_one_hot = torch.FloatTensor(actions_one_hot.float())
        actions_one_hot = Variable(actions_one_hot)
        
        # get current actions' action value
        preds = torch.sum(torch.mul(preds, actions_one_hot), dim=1)
        
        # Q function of next state
        nex_state_preds = self.target_estimator(next_states).data
        
        # calculate Q-Learning target
        target = rewards + (1 - is_dones) * self.gamma * torch.max(nex_state_preds, dim=1)[0]
        target = Variable(target)
        
        # calculate mse loss (preds and targets)
        loss = F.mse_loss(preds, target).mean()
        
        # backward propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

class DoubleDQNAgent():
    def __init__(self, state_size, action_size):
        # for rendering
        self.render = False
        
        self.state_size = state_size
        self.action_size = action_size
        
        # hyperparams for estimator
        self.gamma = 0.95
        self.lr = 0.001
        self.replay_memory_size = 50000
        self.epsilon = 1.0
        self.epsilon_min = 0.000001
        self.explore_steps = 3000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_steps
        self.batch_size = 32
        self.replay_memory_init_size = 1000
        
        # Estimators
        self.q_estimator = DQNEstimator(state_size, action_size)
        self.target_estimator = DQNEstimator(state_size, action_size)
        self.optimizer = optim.SGD(self.q_estimator.parameters(), lr=self.lr)
        
        # memory 
        self.memory = ReplayMemory(self.replay_memory_size)
        
    def update_target_estimator(self):
        self.target_estimator.load_state_dict(self.q_estimator.state_dict())
    
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = Variable(torch.from_numpy(state)).float()
            q_values = self.q_estimator(state)
            _, best_action = torch.max(q_values, dim=1)
            return int(best_action)
    
    def train(self):
        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        # fetch samples from memory
        batch = self.memory.sample(self.batch_size)
        batch = np.array(batch).transpose()
        
        # stack all the states
        states = np.vstack(batch[0])
        actions = torch.LongTensor(list(batch[1]))
        rewards = torch.FloatTensor(list(batch[2]))
        
        # stack all the next states
        next_states = np.vstack(batch[3])
        dones = batch[4]
        dones = dones.astype(int)
        
        # actions one hot encoding
        actions_one_hot = F.one_hot(actions, num_classes=self.action_size)
        actions_one_hot = torch.FloatTensor(actions_one_hot.float())
        actions_one_hot = Variable(actions_one_hot)
        
        # Forward prop
        states = torch.FloatTensor(states)
        states = Variable(states)
        preds = self.q_estimator(states)
        
        # get current action value
        preds = torch.sum(torch.mul(preds, actions_one_hot), dim=1)
        
        # Double DQN
        next_states = torch.FloatTensor(next_states)
        next_states = Variable(next_states)
        next_action_values = self.q_estimator(next_states)
        best_actions = torch.argmax(next_action_values, dim=1)
        q_values_next_target = self.target_estimator(next_states)

        dones = torch.FloatTensor(dones)
        target = rewards + (1 - dones) * self.gamma * q_values_next_target[np.arange(self.batch_size), 
                                                                           best_actions]
        target = Variable(target)
        
        loss = F.mse_loss(preds, target).mean()
        
        # zero out accumulated grads
        self.optimizer.zero_grad()
        
        # back prop
        loss.backward()
        self.optimizer.step()
        return loss.item()
