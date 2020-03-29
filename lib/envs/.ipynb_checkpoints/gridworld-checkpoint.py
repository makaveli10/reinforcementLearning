import io
import sys
import numpy as np
from gym.envs.toy_text import discrete

# TODO: define actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    BaseClass: 'https://github.com/openai/gym/blob/master/gym/envs/toy_text/discrete.py'
    
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, shape=[4, 4]):
        """
        Base class has the following members
        - nS: number of states
        - nA: number actions
        - P: transitions
        - isd: initial state distribution
        
        (*) dictionary dict of dicts of lists, where
            P[s][a] == [(probability, nextstate, reward, done), ...]
        (**) list or array of length nS
        """
        # TODO create and initialize env params
        self.shape = shape
        nS = np.prod(shape)
        nA = 4
        
        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        
        # create grid
        grid = np.arange(nS).reshape(shape)
        iterator = np.nditer(grid, flags=['multi_index'])
        
        # iterate through grid
        while not iterator.finished:
            # fetch state and block
            s = iterator.iterindex
            y, x = iterator.multi_index
            
            # create P[s], reward and if in terminal state
            # P[s][a] = (prob, next_state, reward, is_done)
            is_done = lambda s: s==0 or s==(nS-1)
            P[s] = {a: [] for a in range(nA)}
            reward = 0.0 if is_done(s) else -1.0
            
            # if stuck in terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                # not terminal state
                ns_up = s if y == 0 else s-MAX_X
                ns_right = s if x==(MAX_X-1) else s+1
                ns_down = s if y==(MAX_Y-1) else s+MAX_X
                ns_left = s if x==0 else s-1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]
            
            iterator.iternext()
            
        #initial state distribution
        isd = np.ones(nS)/ nS
        
        self.P = P
        
        super(GridworldEnv, self).__init__(nS, nA, P, isd)
    
    def _render(self, mode='human', close=False):
        """
        Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return
        
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")
            
            it.iternext()
