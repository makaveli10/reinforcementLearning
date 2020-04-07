#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
import itertools

if '../' not in sys.path:
    sys.path.append('../')
    
from collections import defaultdict
from lib.envs.cliffwalking import CliffWalkingEnv
from lib import plots

matplotlib.style.use('ggplot')


env = CliffWalkingEnv()


def create_epsilon_greedy_policy(Q,  nA, epsilon=0.1):
    """
    Behavior Policy.
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        # create action prob numpy array
        A = np.ones(nA, dtype=np.float) * (epsilon/nA)
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def expected_sarsa_off_policy(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    nA = env.action_space.n
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(nA))

    # Keeps track of useful statistics
    stats = plots.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    behavior_policy = create_epsilon_greedy_policy(Q, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
            
        state = env.reset()
        
        for t in itertools.count():
            # sample action from behavior policy
            action_probs = behavior_policy(state)
            action = np.random.choice(env.action_space.n, p=action_probs)
            
            # take action and observe environment's effects
            next_state, reward, done, _ = env.step(action)
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # sample next action from target policy
            next_action = np.argmax(Q[next_state])
            
            td_target = reward + discount_factor * ((1-epsilon)*Q[next_state][next_action]+(epsilon/nA)*np.sum([Q[next_state][a] for a in range(nA)]))
            
            # update Q value
            Q[state][action] += alpha * (td_target - Q[state][action])
            
            
            if done: 
                break
            
            state = next_state
    
    return Q, stats


if __name__=='__main__':
    Q, stats = expected_sarsa_off_policy(env, num_episodes=300)
    plots.plot_episode_stats(stats, file='results/expected_sarsa_off_policy/')



