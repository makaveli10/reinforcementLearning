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


def get_sigma(nA, a):
    """
    Choosing sigma randomly for now.
    """
    # TODO: replace this with a function so 
    # that sigma can be chosen as a function 
    # of variables
    return np.random.randint(2, size=nA)[a]


def q_sigma(env, num_episodes, n=10, gamma=0.9, alpha=0.1):
    """
    n step q sigma algorithm: Off policy TD control.
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
    """
    nA = env.action_space.n
    
    # Create dict Q. A mapping from state to action values
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
    
    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode+1, num_episodes), end="")
            sys.stdout.flush()
            
        T = sys.maxsize
        t = -1
        tau = 0
        
        stored_actions = {}
        stored_states = {}
        stored_rewards = {}
        stored_rho = {}
        stored_sigma = {}
        
        state = env.reset()
        action_probs = behavior_policy(state)
        action = np.random.choice(np.arange(nA), p=action_probs)
        sigma = get_sigma(nA, action)
        rho = target_policy(state)[action]/action_probs[action]
        
        # store selected params
        stored_states[0] = state
        stored_actions[0] = action
        stored_rho[0] = rho
        stored_sigma[0] = sigma
        
        while tau < (T - 1):
            t += 1
            if t < T:
                # take action and observe envronment's effect
                state, reward, done, _ = env.step(action)
                
                stored_states[(t+1) % (n+1)] = state
                stored_rewards[(t+1) % (n+1)] = reward
                
                stats.episode_lengths[i_episode] = t
                stats.episode_rewards[i_episode] += reward
                
                if done:
                    T = t + 1
                else:
                    action_probs = behavior_policy(state)
                    action = np.random.choice(np.arange(nA), p=action_probs)
                    sigma = get_sigma(nA, action)
                    rho = target_policy(state)[action]/action_probs[action]
                    
                    stored_actions[(t+1) % (n+1)] = action
                    stored_sigma[(t+1) % (n+1)] = sigma
                    stored_rho[(t+1) % (n+1)] = rho

            # tau is the time whose estimate is being updated
            tau = t - n + 1
            if tau >= 0:
                if t + 1 < T:
                    G = Q[stored_states[(t+1) % (n+1)]][stored_actions[(t+1) % (n+1)]]
                
                for k in range(min(t+1, T), tau, -1):
                    if k == T:
                        G = stored_rewards[T % (n+1)]
                    else:
                        s_k = stored_states[k % (n+1)]
                        a_k = stored_actions[k % (n+1)]
                        r_k = stored_rewards[k % (n+1)]
                        sigma_k = stored_sigma[k % (n+1)]
                        rho_k = stored_rho[k % (n+1)]
                        v_ = np.sum([(target_policy(s_k)[a]) * Q[s_k][a] for a in range(nA)])
                        G = r_k + gamma * ((sigma_k * rho_k) + ((1 - sigma_k) * (target_policy(s_k)[a_k]))) * (G - Q[s_k][a_k]) + gamma * v_
                
                s_tau, a_tau =  stored_states[tau % (n+1)], stored_actions[tau % (n+1)]
                td_error = G - Q[s_tau][a_tau]
                Q[s_tau][a_tau] += alpha * td_error
                
    return Q, stats

if __name__=='__main__':
    Q, stats = q_sigma(env, num_episodes=300)
    plots.plot_episode_stats(stats, file='results/n_step_q_sigma/')   
            
            