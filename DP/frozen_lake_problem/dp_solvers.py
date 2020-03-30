import numpy as np
import sys
if '../' not in sys.path:
    sys.path.append('../')


def policy_evaluation(policy, 
                      env, 
                      theta=1e-9, 
                      discount_factor=1.0,
                      max_iterations=1e9):
    """
    Evaluates a policy given the environments' full dynamics
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: The OpenAI environment where
            env.P represents the transition probs
            env.P[s][a] represents the list of transition tuples (prob, next_state, reward, terminated)
            env.nS represents no of states
            env.nA represents no of Actions
        theta: threshold. Stop evaluation when change is less than theta
        discount_factor: Gamma
    Returns:
        Value functions. Vector of length env.nS
    """
    # random value function with all zeros
    V = np.zeros(env.nS)
    
    iterations = 1
    while iterations < max_iterations:
        # stopping criteria
        delta = 0
        # evaluate each state
        for state in range(env.nS):
            v = 0
            
            # evaluate all possible actions
            for a, action_prob in enumerate(policy[state]):
                for (prob, next_state, reward, _) in env.P[state][a]:
                    # expected value 
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            
            # update delta
            delta = max(delta, np.abs(v - V[state]))
            
            V[state] = v
        if delta < theta:
            print("Evaluation iterations: {}".format(iterations))
            break 
        
        iterations += 1
    
    return np.array(V)


def one_step_lookahead(state, V, env, gamma):
    """
    Helper function to calculate value of all actions in a state
    Args:
        state: current state
        V: value functions. A vector of length env.nS
        env: OpenAI environment of FrozenLake
        gamma: discount factor. Relative weightage of future rewards
    Returns:
        A vector of length env.nA represention the value taking an action
    """
    A = np.zeros((env.nA))
    
    for a in range(env.nA):
        for prob, next_state, reward, _ in env.P[state][a]:
            A[a] += prob * (reward + gamma * V[next_state])
    return A


def policy_iteration(env, theta=1e-9, discount_factor=1.0, max_iterations=1e9):
    """
    Iterative policy improvement. Iteratively evlauates a policy and improves it
    until an optimal policy is found.
    Args:
        env: The OpenAI environment where
            env.P represents the transition probs
            env.P[s][a] represents the list of transition tuples (prob, next_state, reward, terminated)
            env.nS represents no of states
            env.nA represents no of Actions
        theta: Threshold value. Stop policy evaluation if delta is less than theta ( Early Stopping ).
        discount_factor: Relative weightage of future rewards.
    Returns:
    A tuple (policy, V)
        policy: An optimal Policy to behave in the env of shape [nS, nA].
        V: A vector of length nS denoting how good it is to be in a state. 
    """
    # create random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    iteration = 1
    while iteration < max_iterations:
        # evaluation
        V = policy_evaluation(policy, env)
        policy_stable = True

        # iterate through each state and improve policy
        for state in range(env.nS):
            old_action = np.argmax(policy[state])
            action_values = one_step_lookahead(state, V, env, discount_factor)
            best_action = np.argmax(action_values)
            
            if old_action != best_action:
                policy_stable = False

            # deterministic
            policy[state] = np.eye(env.nA)[best_action]
            
            # stochastic policy
            # best_actions = np.argwhere(action_values==np.max(action_values)).flatten()
            # policy[state] = np.sum([np.eye(env.nA)[i] for i in best_actions], axis=0) / len(best_actions)
            
            
        if policy_stable:
            print("Evaluated policies {}".format(iteration))
            return policy, V
        
        iteration += 1
    

def value_iteration(env, theta=1e-9, discount_factor=1.0, max_iteration=1e9):
    """
    Value iteration to solve MDP
    Args:
        env: Open AI environment already initialized.
        theta: thershold for stopping 
        discount factor: relative weightage for future rewards
        max_iterations: number of iterations to evaluate for
    """
    V = np.zeros(env.nS)
    iteration =1
    while iteration < max_iteration:
        delta = 0
        for state in range(env.nS):
            v = 0
            action_values = one_step_lookahead(state, V, env, discount_factor)
            best_value = np.max(action_values)
            delta = max(delta, abs(V[state] - best_value))
            V[state] = best_value
        if delta < theta:
            print("Value iteration converger after {} iterations".format(iteration))
            break
        iteration += 1
    
    policy = np.zeros([env.nS, env.nA])
    for state in range(env.nS):
        action_values = one_step_lookahead(state, V, env, discount_factor)
        
        # select best action
        best_action = np.argmax(action_values)
        policy[state][best_action] = 1.0
    
    return policy, V

# [[0 3 3 3]
#  [0 0 0 0]
#  [3 1 0 0]
#  [0 2 1 0]]

# Value Function
# [[0.8235294  0.82352939 0.82352939 0.82352938]
#  [0.8235294  0.         0.52941175 0.        ]
#  [0.8235294  0.8235294  0.76470587 0.        ]
#  [0.         0.88235293 0.94117647 0.        ]]

    