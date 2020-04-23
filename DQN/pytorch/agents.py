import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

from replay_memory import ReplayMemory
from q_networks import DQNEstimator



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
        states = torch.Tensor(states).float()
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