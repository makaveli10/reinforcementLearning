#!/usr/bin/env python
# coding: utf-8
import gym
import itertools
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

if "../" not in sys.path:
    sys.path.append("../") 

from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plots

matplotlib.style.use('ggplot')


env = WindyGridworldEnv()


def create_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA) * (epsilon/nA)
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def sarsa_lambd(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, lambd=0.9):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    nA = env.action_space.n
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Keeps track of useful statistics
    stats = plots.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = create_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            
        E = defaultdict(lambda: np.zeros(env.action_space.n))
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(env.action_space.n, p=action_probs)
        for t in itertools.count():
            # environment efforts after taking action
            next_state, reward, done, _ = env.step(action)
            
            next_action_probs = policy(next_state)
            
            next_action = np.random.choice(env.action_space.n, p=next_action_probs)
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            td_error = reward + (discount_factor * Q[next_state][next_action]) - Q[state][action]
            E[state][action] += 1
            
            for s, _ in Q.items():
                for a_ in range(nA):
                    Q[s][a_] += alpha * td_error * E[s][a_]
                    E[s][a_] *= discount_factor * lambd
            
            if done:
                break
            
            state = next_state
            action = next_action
    
    return Q, stats


if __name__=='__main__':
    Q, stats = sarsa_lambd(env, 300)
    plots.plot_episode_stats(stats, file='results/sarsa_lambda/')