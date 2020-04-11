# Some parts of this are taken from 
# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter12/random_walk.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
matplotlib.style.use('ggplot')

# Total number of states excluding the terminal states
nS = 19

STATES = np.arange(1, nS + 1)

# Start state is always the middle one
START_STATE = 10

# Terminal state
# -1 reward on state 0 and 1 on state 20
TERMINAL_STATES = [0, nS + 1]

# True value function
TRUE_VALUES = np.arange(-20, 22, 2) / 20.
TRUE_VALUES[0] = TRUE_VALUES[nS + 1] = 0.


def offline_lambda_return(weights, rate, alpha, rate_truncate=1e-3):
    trajectory = [START_STATE]
    reward = 0.0
    state = START_STATE
    
    while state not in TERMINAL_STATES:
        # print("in while")
        next_state = state + np.random.choice([-1, 1])
        if next_state == 0:
            reward = -1
        elif next_state == nS + 1:
            reward = 1
        else:
            reward = 0
        
        # learn(next_state, reward)
        trajectory.append(next_state)
        if next_state in TERMINAL_STATES:
            # print("in if")
            trajectory_len = len(trajectory) - 1
            
            # offline learning
            for time_step in range(trajectory_len):
                #update for each state in trajectory
                state = trajectory[time_step]
                
                # lambda return from time_step
                returns = 0.0
                plambda = 1
                for n in range(1, trajectory_len - time_step):
                    # calculate n_step_return ( n, time_step)
                    end_time = min(time_step + n, trajectory_len)
                    nreturns = weights[trajectory[end_time]]
                    if end_time == trajectory_len:
                        nreturns += reward
                    returns += plambda * nreturns
                    plambda *= rate
                    
                    if plambda < rate_truncate:
                        break
                
                returns *= 1 - rate
                    
                if plambda >= rate_truncate:
                    returns += plambda * reward
            
                delta = returns - weights[state]
                delta *= alpha
            
                weights[state] += delta
                # print(weights)
        
        state = next_state
        
        
def random_walk(runs, lambdas, alphas):
    # play for 10 episodes for each run
    episodes = 10
    # track the rms errors
    errors = [np.zeros(len(alphas_)) for alphas_ in alphas]
    for run in tqdm(range(runs)):
        for lambdaIndex, rate in enumerate(lambdas):
            for alphaIndex, alpha in enumerate(alphas[lambdaIndex]):
                weights = np.zeros(nS + 2)
                for episode in range(episodes):
                    offline_lambda_return(weights, rate, alpha)
                    # print(weights)
                    stateValues = [weights[state] for state in STATES]
                    errors[lambdaIndex][alphaIndex] += np.sqrt(np.mean(np.power(stateValues - TRUE_VALUES[1: -1], 2)))

    # average over runs and episodes
    for error in errors:
        error /= episodes * runs

    for i in range(len(lambdas)):
        plt.plot(alphas[i], errors[i], label='lambda = ' + str(lambdas[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.legend()
    
    
if __name__=="__main__":
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01)]
    random_walk(50, lambdas, alphas)

    plt.savefig('results/offline_mc_lambda/figure_12_3.png')
    plt.close()
    

