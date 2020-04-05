## Model free prediction and control

### What this folder has
- Understand TD(0) and apply the prediction algorithm to Blackjack env
- Read random walk example from the book ex 6.2
- Algorithm for on-policy TD control methods such as SARSA and n-step-SARSA
- TD Target for SARSA: R[t+1] + gamma * Q[next_state][next_action]
- TD target for n-step-SARSA: 
    G[t:t+n] = R[t+1]+ gamma * R[t+2]+···+(gamma ** (n-1))* R[t+n] + (gamma ** n) * Q[t+n-1](S[t+n], A[t+n]),
- Algorithm for off-policy TD control methods such as Q-learning
- TD Target for Q-Learning: R[t+1] + gamma * max(Q[next_state])


### Important points
- If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be temporal-di↵erence (TD) learning. TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).

- Q-leanring is considered as an off-policy method becuase its updatingoesn't bootstrap from the off policy method but on a target policy i.e. max over all actions from the next state. Target policy will always be a greedy policy no matter what the behavior policy is.
