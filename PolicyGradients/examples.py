import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
matplotlib.use('Agg')
matplotlib.style.use('ggplot')


# Reference - https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/
# Create Short Corridoor environment
class Environment:
    """
    Reward is -1 per step. With only two actions
    left and right. Only three states, left causes no
    movement in first state. But in the middle state
    i.e. second state movements are reversed left goes
    right and right goes left. All episodes start in 
    state 0.
    """
    def __init__(self):
        # reset env
        self.reset()
    
    def reset(self):
        self.state = 0
    
    def step(self, right):
        """
        In state 0 left stays in 0 right behaves normally
        and goes to 1 but in state 1 right and left behave
        opposite. In state 2 left goes to middle state
        and right terminates the episode.
        Args:
            action: 0 for left and 1 for right.
        Returns:
            A tuple of (reward, is_terminal_state)
        """
        is_done = False
        reward = -1
        if self.state == 1:
            # if go right
            if right:
                self.state -= 1
            else:
                self.state += 1
        else:
            if right:
                self.state += 1
            else:
                self.state = max(0, self.state - 1)
        if self.state == 3:
            is_done = True
            reward = 0
        return reward, is_done
    
    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
    

# Create Agent that follows Reinforce MC Control
class ReinforceMCAgent:
    def __init__(self, gamma, alpha):
        self.gamma = gamma
        self.alpha = alpha
        self.params = np.array([-1.5, 1.5])
        self.features = np.array([[0,1], [1,0]])
        self.actions = list()
        self.rewards = list()
        
    def get_policy(self):
        prod = np.dot(self.params, self.features)
        probs = softmax(prod)
        eps = 0.05
        indexmin = np.argmin(probs)
        if probs[indexmin] < eps:
            probs[:] = 1 - eps
            probs[indexmin] = eps
        return probs

    def get_prob_right(self):
        return self.get_policy()[1]
    
    def get_action(self):
        probs = self.get_policy()
        right = np.random.uniform() <= probs[1]
        self.actions.append(right)
        return right
    
    def update_params(self, latest_reward):
        self.rewards.append(latest_reward)
        
        # calculate G for each time step
        G = np.zeros(len(self.rewards))
        G[-1] = latest_reward
        for i in range(2, len(G)+1):
            G[-i] = self.gamma * G[-i+1] + self.rewards[-i]
        
        # update params
        for i in range(len(G)):
            j = 1 if self.actions[i] else 0
            probs = self.get_policy()
            grad_ln = self.features[:, j] - np.dot(self.features, probs)
            self.params += self.alpha * (self.gamma**i) * G[i] * grad_ln
            
        self.actions = list()
        self.rewards = list()
       

def agent_run(episodes, run_agent):
    env = Environment()
    agent = run_agent()
    
    rewards = np.zeros(episodes)
    for i in range(episodes):
        reward = None
        total_rewards = 0
        env.reset()
        
        for t in itertools.count():
            if reward is not None:
                agent.rewards.append(reward)
            action = agent.get_action()
            reward, is_done = env.step(action)
            total_rewards += reward
            if is_done:
                agent.update_params(reward)
                break
        
        rewards[i] = total_rewards
    return rewards
 

class ReinforceBaseline(ReinforceMCAgent):
    def __init__(self, gamma, alpha, alpha_w):
        super(ReinforceBaseline, self).__init__(gamma, alpha)
        self.alpha_w = alpha_w 
        self.w = 0
        
    def update_params(self, latest_reward):
        self.rewards.append(latest_reward)
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]
        
        for i in range(2, len(G)+1):
            G[-i] = self.gamma * G[-i+1] + self.rewards[-i]
            
        # update params
        for i in range(len(G)):
            delta = G[i] - self.w
            self.w += self.alpha_w * delta
            
            j = 1 if self.actions[i] else 0
            probs = self.get_policy()
            grad_ln = self.features[:, j] - np.dot(self.features, probs)
            self.params += self.alpha * delta * (self.gamma**i) * grad_ln
        self.actions = list()
        self.rewards = list()
 
 
# Create driver method to plot results
def plot_13_1():
    runs = 100
    episodes = 1000
    gamma = 1
    agents = [lambda: ReinforceMCAgent(gamma, alpha=2e-4), 
              lambda: ReinforceMCAgent(gamma, alpha=2e-5),
              lambda: ReinforceMCAgent(gamma, alpha=2e-3)]
    labels = ['2e-4',
              '2e-5',
              '2e-3']
    rewards = np.zeros((len(agents), runs, episodes))
    for idx, agent in  enumerate(agents):
        for run in tqdm(range(runs)):
            r = agent_run(episodes, agent)
            rewards[idx, run, :] = r
    plt.plot(np.arange(episodes) + 1, -11.6 * np.ones(episodes), ls='dashed', color='red', label='-11.6')
    for idx, label in enumerate(labels):
        plt.plot(np.arange(episodes) + 1, np.mean(rewards[idx], axis=0), ls='dashed', label=label)
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('figure_13_1.png')
    plt.close()
    

def plot_13_2():
    n_runs = 100
    episodes = 1000
    gamma = 1
    alpha = 2e-4
    agents = [lambda: ReinforceMCAgent(gamma, alpha),
              lambda: ReinforceBaseline(gamma, 2e-3, 2e-2)]
    labels = ['Reinforce without baseline',
              'Reinforce with baseline']
    
    rewards = np.zeros((len(agents), n_runs, episodes))
    for idx, agent in enumerate(agents):
        for i in tqdm(range(n_runs)):
            r = agent_run(episodes, agent)
            rewards[idx, i, :] = r
            
    plt.plot(np.arange(episodes) + 1, -11.6 * np.ones(episodes), ls='dashed', color='red', label='-11.6')
  
    for idx, label in enumerate(labels):
        plt.plot(np.arange(episodes) + 1, np.mean(rewards[idx], axis=0), label=label)

    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('figure_13_2.png')
    plt.close()
        

if __name__=="__main__":
    plot_13_2()
    
