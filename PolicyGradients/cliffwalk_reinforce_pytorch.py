#!/usr/bin/env python
# coding: utf-8


import sys
import os
import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
from collections import namedtuple
from torch.autograd import Variable
if '../' not in sys.path:
    sys.path.append('../')
    
from lib.envs.cliffwalking import CliffWalkingEnv
from lib import plots
matplotlib.style.use('ggplot')

env = CliffWalkingEnv()
action_size = env.action_space.n
action_size


class PolicyEstimator(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyEstimator, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc = nn.Linear(in_features=state_size, out_features=action_size)
    
    def forward(self, state):
        return self.fc(state)
    
    def predict(self, state):
        state_tensor = torch.tensor(state)
        state_one_hot = F.one_hot(state_tensor, int(self.state_size))
        state_variable = Variable(torch.unsqueeze(state_one_hot, 0)).float()
        output = self.forward(state_variable)
        return torch.squeeze(F.softmax(output, dim=1))
    
    def update(self, state, target, action, optimizer):
        action_probs = self.predict(state)
        picked_action_prob = torch.gather(action_probs, 0, action)
        loss = -torch.log(picked_action_prob) * target
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class ValueEstimator(nn.Module):
    def __init__(self, state_size):
        super(ValueEstimator, self).__init__()
        self.state_size = state_size
        self.fc = nn.Linear(in_features=state_size, out_features=1)
    
    def forward(self, state):
        return self.fc(state)
    
    def predict(self, state):
        state_tensor = torch.tensor(state)
        state_one_hot = F.one_hot(state_tensor, int(self.state_size))
        state_variable = Variable(torch.unsqueeze(state_one_hot, 0)).float()
        return torch.squeeze(self.forward(state_variable))
    
    def update(self, state, target, optimizer):
        value_estimate = self.predict(state)
        value_loss = sum((value_estimate - target)**2)
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()


class ReinforceBaselineAgent():
    def __init__(self, state_size, action_size, num_episodes):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_estimator = PolicyEstimator(state_size=state_size, action_size=action_size)
        self.value_estimator = ValueEstimator(state_size=state_size)
        self.lr = 0.01
        self.policy_optimizer = torch.optim.Adam(self.policy_estimator.parameters(), lr=self.lr)
        self.value_optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=self.lr)
        self.num_episodes = num_episodes
        self.gamma = 1.0
    
    def train(self):
        stats = plots.EpisodeStats(
        episode_lengths=np.zeros(self.num_episodes),
        episode_rewards=np.zeros(self.num_episodes))
    
        Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        for i_episode in range(self.num_episodes):
            state = env.reset()
            trajectory = list()
            for t in itertools.count():
                # get action prediction
                action_probs = self.policy_estimator.predict(state).detach().numpy()
                
                # get action
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                
                # take step
                next_state, reward, done, _ = env.step(action)
                
                trajectory.append(Transition(state=state,
                                             action=action,
                                             reward=reward,
                                             next_state=next_state,
                                             done=done))
                
                stats.episode_lengths[i_episode] = t
                stats.episode_rewards[i_episode] += reward
                
                # Print out which step we're on, useful for debugging.
                print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, self.num_episodes, stats.episode_rewards[i_episode - 1]), end="")
                sys.stdout.flush()
                
                if done:
                    break
                
                state = next_state
                
            for t, transition in enumerate(trajectory):
                # get total reward
                total_return = sum(self.gamma**i * tr.reward for i, tr in enumerate(trajectory[t:]))
                
                # get value_estimate
                value_estimate = self.value_estimator.predict(transition.state).detach()
                
                advantage = torch.FloatTensor([total_return]) - value_estimate
                advantage = torch.FloatTensor([advantage])
                
                # update value estimator
                self.value_estimator.update(transition.state, 
                                            torch.FloatTensor([total_return]), 
                                            self.value_optimizer)
                
                # update policy estimator
                action = torch.LongTensor([transition.action])
                self.policy_estimator.update(transition.state, 
                                             advantage, 
                                             action, 
                                             self.policy_optimizer)
                
        return stats

if __name__=="__main__":
    agent = ReinforceBaselineAgent(env.observation_space.n, action_size, 2000)
    stats = agent.train()
    plots.plot_episode_stats(stats, smoothing_window=25, file='results/pytorch_reinforce/')
