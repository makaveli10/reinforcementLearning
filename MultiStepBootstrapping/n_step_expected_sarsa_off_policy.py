import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

if '../' not in sys.path:
    sys.path.append('../')
    
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plots
matplotlib.style.use('ggplot')


# initialize windy gridworld env
env = WindyGridworldEnv()


# create behavior policy
def create_behavior_policy(Q, nA, epsilon=0.3):
    """
    Creates a behavior policy. Epsilon greedy policy.
    Args:
        Q: A dictionary that maps states to action probabilities. Each 
           value is a numpy array of length on nA.
        epsilon: The probability to select a random action. Float 0 and 1
        nA: Number of actions in the env's action space.
    Returns:
        A function that takes observations as argument and returns
        action probabilities i.e a numpy array of length nA.
    """
    def policy_fn(observations):
        """
        Takes random action with probability epsilon/nA and best
        action with probability 1 - epsilon + epsilon/nA
        """
        A = np.ones(nA, dtype=np.float) * (epsilon/nA)
        best_action = np.argmax(Q[observations])
        A[best_action] += 1.0 - epsilon
        return A
    return policy_fn


def create_target_policy(Q, nA, epsilon=0.1):
    """
    Creates a target policy. Can be another epsilon greedy policy
    or greedy policy with respect to Q action values.
    Args:
        Q: A dictionary that maps states to action probabilities. Each 
           value is a numpy array of length on nA.
        epsilon: The probability to select a random action. Float 0 and 1
        nA: Number of actions in the env's action space.
    Returns:
        A function that takes observations as argument and returns
        action probabilities i.e a numpy array of length nA.
    """
    def policy_fn(observations):
        A = np.ones(nA, dtype=np.float) * (epsilon/nA)
        best_action = np.argmax(Q[observations])
        A[best_action] += 1.0 - epsilon
        return A
    return policy_fn


def n_step_expected_sarsa(env, num_episodes, n=10, gamma=0.9, alpha=0.1, epsilon=0.3):
    """
    n step Expected SARSA algorithm: Off-policy TD control. Finds the optimal target policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        n: future time steps to look ahead and calculate return for.
        gamma: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    NOTE: some parts taken from https://github.com/Breakend/MultiStepBootstrappingInRL/blob/master/n_step_sarsa.py
    """
    # initializations
    # number of actions
    nA = env.action_space.n
    
    # create Q dict. A nested dict that maps state-> action values
    Q = defaultdict(lambda: np.zeros(nA))
    
    # policy we are following which is more 
    # exploratory and less greedy
    behavior_policy = create_behavior_policy(Q, nA)
    
    # Policy we are learning. Less exploratory
    # than behavior policy -> means more greedy.
    target_policy = create_target_policy(Q, nA)
    
    # Keeps track of useful statistics
    stats = plots.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    max_reward = 0
    total_reward = 0
    rewards_per_episode = []
    q_variance = []
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        T = sys.maxsize
        tau = 0
        t = -1
        
        stored_actions = {}
        stored_rewards = {}
        stored_states = {}
        
        # reset env to get initial state
        state = env.reset()
    
        # get action probs from behavior policy
        action_probs = behavior_policy(state)
        action = np.random.choice(np.arange(nA), p=action_probs)
        
        stored_actions[0] = action
        stored_states[0] = state
        
        while tau < (T - 1):
            t += 1
            if t < T:
                state, reward, done, _ = env.step(action)
                
                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t
                
                stored_rewards[(t+1) % (n+1)] = reward
                stored_states[(t+1) % (n+1)] = state
                
                if done:
                    T = t + 1
                else:
                    action_probs = behavior_policy(state)
                    action = np.random.choice(np.arange(nA), p=action_probs)
                    stored_actions[(t+1) % (n+1)] = action
            tau = t - n + 1
            if tau >= 0:
                # calculate rho
                rho = np.prod(
                    [target_policy(stored_states[i%(n+1)])[stored_actions[i%(n+1)]]/behavior_policy(stored_states[i%(n+1)])[stored_actions[i%(n+1)]] for i in range(tau+1, min(tau+n-1, T-1)+1)]
                    )
                
                # calculate return
                G = np.sum([(gamma**(i-tau-1))*stored_rewards[i%(n+1)] for i in range(tau+1, min(tau+n, T)+1)])
                
                
                if tau+n < T:
                    expected_sarsa_update = np.sum(
                        [target_policy(stored_states[(tau+n) % (n+1)])[a] * Q[stored_states[(tau+n) % (n+1)]][a] for a in range(nA)]
                    )
                    G += (gamma**n) * expected_sarsa_update
                    
                s_tau, a_tau = stored_states[tau % (n+1)], stored_actions[tau % (n+1)]
                
                td_error = G - Q[s_tau][a_tau]
                Q[s_tau][a_tau] += alpha * rho * td_error
    return Q, stats


if __name__=='__main__':
    Q, stats = n_step_expected_sarsa(env, num_episodes=300)
    plots.plot_episode_stats(stats, file='results/n_step_off_policy_expected_sarsa/')