import random
from collections import namedtuple


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