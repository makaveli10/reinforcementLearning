import numpy as np


class BinarySumTree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*self.capacity - 1)
        self.transitions = np.zeros(self.capacity, dtype=object)
        self.n_transitions = 0
        self.transition_index = 0
    
    # propagate the change up the tree to the node
    def propagate_error(self, index, delta):
        # get the parent of current node
        parent = (index - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self.propagate_error(parent, delta) 
    
    # Retrieve sample from the tree according to given priority
    def retrieve_transition(self, index, priority):
        # get children 
        left_child_index = 2 * index + 1
        right_child_index = left_child_index + 1
        
        if left_child_index >= len(self.tree):
            return index
        
        if priority <= self.tree[left_child_index]:
            return self.retrieve_transition(left_child_index, priority)
        else:
            return self.retrieve_transition(right_child_index, priority - self.tree[left_child_index])
    
    # append error to tree and transition to transitions
    def add(self, error, transition):
        # get the tree index to store error
        index = self.transition_index + self.capacity - 1
        
        # append transition to data
        self.transitions[self.transition_index] = transition
        
        # update the tree with the error
        self.update(index, error)
        
        # update transition_index
        self.transition_index += 1
        if self.transition_index >= self.capacity:
            self.transition_index = 0  
        if self.n_transitions < self.capacity:
            self.n_transitions += 1
    
    def update(self, index, error):
        delta = error - self.tree[index]
        self.tree[index] = error
        
        self.propagate_error(index, delta)
    
    def fetch_transition(self, sample_priority):
        index = self.retrieve_transition(0, sample_priority)
        transition_index = index - self.capacity + 1
        transition = self.transitions[transition_index]
        return (index, self.tree[index], transition)
    
    # get total error in the tree
    def cumulative_error(self):
        return self.tree[0]

    