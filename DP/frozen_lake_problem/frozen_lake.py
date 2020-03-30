import gym
import numpy as np
from dp_solvers import policy_iteration,value_iteration

action_mapping = {
    0: '\u2191', # UP
	1: '\u2192', # RIGHT
	2: '\u2193', # DOWN
	3: '\u2190', # LEFT
}

def play_episodes(environment, n_episodes, policy):
    # get total number of wins
    wins = 0
    total_rewards = 0
    
    # play episodes
    for episode in range(n_episodes):
        terminated = False
        
        # get random initial state
        state = environment.reset()
        
        # play until terminated
        while not terminated:
            
            # get the best action
            best_action = np.argmax(policy[state])
            
            # take step in the environment
            next_state, reward, terminated, p = environment.step(best_action)
            
            # update next state
            state = next_state
            
            total_rewards += reward
            if terminated and reward==1.0:
                wins +=  1
                
    avg_reward = total_rewards / n_episodes
    return wins, total_rewards, avg_reward


# Number of episodes to play
n_episodes = 10000

# Functions to find best policy
solvers = [
    ('Policy Iteration', policy_iteration),
    ('Value Iteration', value_iteration)
]

for iteration_name, iteration_func in solvers:

    # Load a Frozen Lake environment
    environment = gym.make('FrozenLake-v0')

    # Search for an optimal policy using policy iteration
    policy, V = iteration_func(environment.env)

    print(f'\n Final policy derived using {iteration_name}:')
    print(' '.join([action_mapping[action] for action in np.argmax(policy, axis=1)]))

    # Apply best policy to the real environment
    wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)

    print(f'{iteration_name} :: number of wins over {n_episodes} episodes = {wins}')
    print(f'{iteration_name} :: average reward over {n_episodes} episodes = {average_reward} \n\n')