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
from agents import DoubleDQNAgent


env = gym.make('CartPole-v0')
if __name__=="__main__":
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DoubleDQNAgent(state_size, action_size)
    scores, episodes = list(), list()
    num_episodes = 500
    total_steps = 0
    flag = False
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0

    # Populate replay memory
    print("Populating replay memory")
    for i in range(agent.replay_memory_init_size):
        action = agent.get_action(state)
        next_state, reward, is_done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        reward = reward if not is_done or score == 200 else -10
        agent.memory.push(state, action, reward, next_state, is_done)
        score += reward
        if is_done:
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            score = 0
        else:
            state = next_state

    for i_episode in range(1, num_episodes + 1):
        score = 0
        # update target network
        agent.update_target_estimator()
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for t in count():
            if agent.render:
                env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 199 else -10
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train()
            score += reward
            state = next_state
            if done:
                score = score if score==200 else score + 10
                scores.append(score)
                episodes.append(i_episode)
                plt.plot(episodes, scores, label="Scores vs Episode")
                plt.xlabel("Episodes")
                plt.ylabel("Score")
                plt.savefig("plots/cartpole_double_dqn.png")
                if np.mean(scores[-min(100, len(scores)):]) >=195:
                    torch.save(agent.q_estimator.state_dict(), "saved_model/cartpole_double_dqn")
                    print("Saved Model")
                    print("Average Score of last 100 episodes {}".format(np.mean(scores[-min(100, len(scores)):])))
                    flag = True
                    sys.exit()
                break
            total_steps += 1
        print("\rTotal Steps: {}  Episode {}/{} Loss: {:.4f}  epsilon: {}".format(total_steps, i_episode, num_episodes, loss, agent.epsilon), end="")
        sys.stdout.flush()