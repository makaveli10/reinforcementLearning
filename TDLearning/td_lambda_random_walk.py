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


def td_lambda(weights, rate, alpha):
    eligibility = np.zeros(nS + 2)

    state = START_STATE
    
    while state not in TERMINAL_STATES:
        # take action move left or right
        next_state = state + np.random.choice([-1, 1])
        if next_state == 0:
            reward = -1
        elif next_state == nS + 1:
            reward = 1
        else:
            reward = 0
        
        eligibility *= rate
        eligibility[state] += 1
        td_error = reward + weights[next_state] - weights[state]
        td_error *= alpha
        weights += td_error * eligibility
        
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
                    td_lambda(weights, rate, alpha)
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
              np.arange(0, 0.99, 0.09),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.33, 0.03),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01),
              np.arange(0, 0.044, 0.004)]
    random_walk(50, lambdas, alphas)

    plt.savefig('results/td_lambda/figure_12_6.png')
    plt.close()