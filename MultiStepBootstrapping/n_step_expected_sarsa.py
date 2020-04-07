import sys
import itertools
import matplotlib
matplotlib.use('Agg')
import numpy as np

if '../' not in sys.path:
    sys.path.append('../')
    
from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plots

matplotlib.style.use('ggplot')


# initialize windy gridworld env
env = WindyGridworldEnv()


def create_epsilon_greedy_policy(Q, nA, epsilon=0.3):
    """
    Creates a epsilon greed policy based on the Q values.
    
    Args:
        Q: A dictionary that maps from state -> action-values. Each 
           value is a numpy array of length nA (see below)
        nA: Number of actions in the environments' action space.
        epsilon: Probablity to select a random action. Float between 0 and 1.
        
    Returns:
        A numpy array of length nA specifying the probability of selecting each 
        action given the observations.
    """
    def policy_fn(observation):
        # initialize action values
        A = np.ones(nA, dtype=np.float) * (epsilon/nA)
        
        # get the best action for current state
        best_action = np.argmax(Q[observation])
        
        # update probability of current action
        A[best_action] += 1.0 - epsilon
        return A
    return policy_fn



def n_step_expected_sarsa(env, num_episodes, n=5, gamma=0.9, epsilon=0.1, alpha=0.1):
    """
    (n step)Expected SARSA: On policy TD control. Finds the optimal epsilon greedy policy. The 
    algorithm is same as n step SARSA except that its last element is the branch over all action 
    possibilities weighted by their probabilities under pi(policy we are following).
    
    Args:
        env: The OpenAI environment.
        num_episodes: Number of episodes to run for.
        n: future time steps to look ahead and calculate return for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float betwen 0 and 1.
        
    Returns:
        A tuple (Q, state)
        Q: A dict mapping state -> action values. Q is the optimal action-value 
           function, a dictionary mapping state -> action values.
        stats: An EpisodeStats object with two numpy arrays for episode_lengths 
               and episode_rewards.
    """

    nA = env.action_space.n
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(nA))
    
    # policy we are following
    policy = create_epsilon_greedy_policy(Q, nA, epsilon)
    
    # track useful stats to plot
    stats = plots.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
    )
    
    
    for i_episode in range(num_episodes):
        # print the current episode for debugging
        if (i_episode + 1) % 100 == 0:    
            print("\rEpisode {}/{}.".format(i_episode+1, num_episodes),  end="")
            sys.stdout.flush()
        
        # initializations
        stored_rewards = {}
        stored_states = {}
        stored_actions = {}
        
        T = sys.maxsize
        t = -1
        tau = 0
        
        # reset OpenAI env to get the initial state
        state = env.reset()
        
        # get action probabilities from the policy function
        action_probs = policy(state)

        # sample action according to the action probabilities
        action = np.random.choice(np.arange(nA), p=action_probs)
        
        # store current action and state
        stored_actions[0] = action
        stored_states[0] = state
        
        while tau < (T - 1):
            t += 1
            if t < T:
                # observe environments effects after taking sampled action
                next_state, reward, done, _ = env.step(action)
                
                # assign next_state to current state
                state = next_state
                
                stored_states[(t+1) % (n+1)] = state 
                stored_rewards[(t+1) % (n+1)] = reward
                
                stats.episode_lengths[i_episode] = t
                stats.episode_rewards[i_episode] += reward   
                
                if done:
                    T = t + 1
                else:
                    # select and store action A[t+1]
                    action_probs = policy(state)
                    action = np.random.choice(np.arange(nA), p=action_probs)
                    stored_actions[(t+1) % (n+1)] = action
            
            tau = t - n + 1
            if tau >= 0:
                # caluclate return
                G = np.sum([(gamma**(i-tau-1)) * stored_rewards[i % (n+1)] for i  in range(tau+1, min(tau+n, T)+1)])
                
                # this step we calculate value of all action possibilities weighted 
                # by their probabilities under pi(policy we are following).
                if tau + n < T:
                    exp_sarsa_update = np.sum(
                        [policy(stored_states[(tau+n) % (n+1)])[a] * Q[stored_states[(tau+n) % (n+1)]][a] for a in range(nA)]
                    )
                    G += (gamma ** n) * exp_sarsa_update
                
                # update Q value here
                s_tau, a_tau = stored_states[tau % (n+1)], stored_actions[tau % (n+1)]
                Q[s_tau][a_tau] += alpha * (G - Q[s_tau][a_tau])
            
    return Q, stats


if __name__=='__main__':
    Q, stats = n_step_expected_sarsa(env, num_episodes=300, n=10)
    plots.plot_episode_stats(stats, file='results/n_step_expected_sarsa/')