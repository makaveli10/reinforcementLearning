import gym
import numpy as np
import itertools
from agents import DoubleDQNPrioritizedReplayAgent

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DoubleDQNPrioritizedReplayAgent(state_size, action_size)
for i_episode in range(100):
    observation = env.reset()
    observation = np.reshape(observation, [1, state_size])
    for t in itertools.count():
        env.render()
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        observation = np.reshape(observation, [1, state_size])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()