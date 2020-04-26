import random
import numpy as np
from collections import namedtuple
from utils import BinarySumTree


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayMemory(object):
    def __init__(self, maxlen):
        self.capacity = maxlen
        self.memory = list()
        self.position = 0
    
    def push(self, *args):
        """Saves Transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

# https://arxiv.org/pdf/1511.05952.pdf
class PrioritizedReplayMemory(object):
    """
    Stochastic Proportional Prioritized Experience Replay Memory. Adds
    transitions to memory uses a SumTree to keep track or errors and 
    fetch transition from each segment of errorrs.
    Fetches Transitions from replay memory according to the magnitude 
    of error. Read paper for more Details.
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = BinarySumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        """
        Samples a segment, and then sample uniformly among the transitions 
        within it. This works particularly well in conjunction with a 
        minibatch based learning algorithm: choose k to be the size of the 
        minibatch, and sample exactly one transition from each segment – this 
        is a form of stratified sampling that has the added advantage of 
        balancing out the minibatch
        """
        batch = []
        idxs = []
        segment = self.tree.cumulative_error() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.fetch_transition(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.cumulative_error()
        is_weight = np.power(self.tree.n_transitions * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)