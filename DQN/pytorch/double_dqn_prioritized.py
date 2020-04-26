import sys
import gym
import math
import random
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image
from agents import DoubleDQNPrioritizedReplayAgent
matplotlib.use('Agg')
matplotlib.style.use('ggplot')


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DoubleDQNPrioritizedReplayAgent(state_size, action_size)
    scores, episodes = [], []
    saved = False
    for e in range(500):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for t in count():
            if agent.render:
                env.render()
            # get action for the current state and go one step in environment
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 200 else -10
            # save the sample <s, a, r, s'> to the replay memory
            agent.push_sample(state, action, reward, next_state, done)
            # every time step do the training
            if agent.memory.tree.n_transitions >= agent.replay_memory_init_size:
                agent.train()
#                 break
            score += reward
            state = next_state
            if done:
                # every episode update the target model to be same with model
                agent.update_target_estimator()
                # every episode, plot the play time
                score = score if score == 200 else score + 10
                scores.append(score)
                episodes.append(e)
                plt.plot(episodes, scores, 'b')
                plt.xlabel("Episodes")
                plt.ylabel("Scores")
                plt.savefig("plots/cartpole_prioritized.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      agent.memory.tree.n_transitions, "  epsilon:", agent.epsilon)
                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(100, len(scores)):]) > 195:
                    print(np.mean(scores[-min(100, len(scores)):]))
                    torch.save(agent.q_estimator.state_dict(), 'saved_model/cartpole_doubledqn_prioritized')
                    sys.exit()
                break