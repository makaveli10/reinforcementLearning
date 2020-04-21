# DEEP Q-LEARNING

 - The goal of the agent is to select actions in a fashion that maximizes cumulative 
 future reward. More formally, we use a deep convolutional neural network to approximate
 the optimal action-value function.

- DQN uses Deep CNN as a function approximator as we know is a powerful function approximator but if not applied properly then training can diverge and is also unstable.

- So, there are several tricks mentioned in [Human-Level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) like Experience Replay. Storing the agents experience and then
selecting randomly from that experience for training the function approximator. This randomizes over the data, thereby removing correlations in the observation sequence and smoothing over changes in the data distribution 

- The other trick mentioned in the paper is the use of a target network. They used an iterative update that adjusts the action-values (Q) towards target values that are only periodically updated, thereby reducing correlations with the target.


## Double DQN

- The max operator in Deep Q-Learning uses the same values to both select and to evaluate an action which is more likely to select overestimated values resulting in overoptimistic value estimates.
- DQN update => w[t+1] = w[t] + lr(Y(Q)t - Q(S[t],a;w(t)))*grad(Q(S[t],a;w(t)))
Y(Q)t = R[t+1] + gamma * max(a)Q(S[t+1], a; w-(t))
- Decoupling the selection from the evaluation is the idea behind Double Q-learning.
- For Double DQN Y(Q)t = R[t+1] + gamma * Q(S[t+1], argmax(a)Q(S[t+1], a; w(t)); w-(t)).
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)


