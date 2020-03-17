import numpy as np
import math
import matplotlib.pyplot as plt

class Params():
    def __init__(self):
        # define parameters
        # max car per location
        self.max_car = 20
    
        # maximum cars moved from one location
        self.max_move_loc = 5
    
        # reward
        self.reward_per_car = 10
    
        # cost
        self.cost_per_car = 2
    
        # discount rate
        self.gamma = 0.9
    
        # Cost to keep more car than half the maximum overnight, for the
        # modified version of Jack's Car Rental problem independent of number of cars
        self.cost_per_slot_additional = 4
    
        # Small number determining the accuracy of policy evaluation's estimation
        self.theta = 0.01
    
        # Expectation for rental requests in first location
        self.lambda_request_first = 3

        # Expectation for rental requests in second location
        self.lambda_request_second = 4

        # Expectation for returns in first location
        self.lambda_return_first = 3

        # Expectation for returns in second location
        self.lambda_return_second = 2

        # Possible versions of the problem
        self.problem_types = ['original_problem', 'modified_problem']

    
class PolicyIteration():
    def __init__(self, params, problem_type):
        self.params = params
        self.problem_type = problem_type
        # all possible states
        self.states = [(x, y) for x in range(self.params.max_car + 1) for y in range(self.params.max_car + 1)]
        
        # all value functions
        self.values = np.zeros((self.params.max_car + 1, self.params.max_car + 1))
        
        # all possible actions
        self.pi = np.zeros((self.params.max_car + 1, self.params.max_car + 1))
        
        # list of policy func
        self.pis = []
        
    def solve(self):
        """
        """
        i = 0
        while True:
            print('Iteration: ', i + 1)
            
            # policy evaluation
            while True:
                delta = 0
                for s in self.states:
                    v = self.values[s]
                    self.values[s] = self.V_eval(s, self.pi[s])
                    delta = np.maximum(delta, abs(v - self.values[s]))
                if delta < self.params.theta:
                    break
                print('Delta:', delta)
                                       
            # policy iteration               
            policy_stable = True
            for s in self.states:
                old_act = self.pi[s]
                values = {a: self.V_eval(s, a) for a in self.A(s)}
                self.pi[s] = np.random.choice([a for a, value in values.items()
                                                   if a == np.max(list(values.values()))])
                if old_act != self.pi[s]:
                    policy_stable = False
            if policy_stable:
                break
            i += 1
                                       
    def A(self, s):
        """
        Get all possible actions given a state
        :param s: state
        :return: possible actions
        """
        values = []
        actions = [x for x in range(-self.params.max_move_loc, self.params.max_move_loc + 1)]
        s_first, s_second = s
        
        # Discard actions that would make the number of car negative or 
        # higher than max in at least one of the locations
        for a in actions:
            if s_first - a < 0 or s_first - a > self.params.max_car:
                continue
            if s_second + a < 0 or s_second + a > self.params.max_car:
                continue
            values.append(a)
        return values
    
    def V_eval(self, s, a):
        """
        Compute value given a state and an action for the state following the formula:
        sum over all s',r of p(s',r|s, a)[r + gamma*V(s')]
        :param s: state
        :param a: action
        :return: value
        """
        value = 0
        s_first, s_second = s
        
        # action
        s_first -= int(a)
        s_second += int(a)
        
        # compute cost of action
        if self.problem_type == 'original_problem':
            cost = self.params.cost_per_car * abs(a)
        else:
            if a > 0:
                a -= 1
            cost = self.params.cost_per_car * abs(a) + self.params.cost_per_slot_additional(
                1 if s_first > self.params.max_car / 2 else 0 + 1 if s_second > self.params.max_car / 2 else 0)
            
        # compute for each possible new state:prob, reward, value of the new state
        sum_prob_i = 0
        for i in range(s_first + 1):
            if i == s_first:
                p_i = 1 - sum_prob_i
            else:
                p_i = PolicyIteration.poisson(self.params.lambda_request_first, i)
                sum_prob_i += p_i
            r_i = i * self.params.reward_per_car
            sum_prob_j = 0
            for j in range(s_second + 1):
                if j == s_second:
                    p_j = 1 - sum_prob_j
                else:
                    p_j = PolicyIteration.poisson(self.params.lambda_request_second, j)
                    sum_prob_j += p_j
                r_j = j * self.params.reward_per_car
                sum_prob_k = 0
                for k in range(self.params.max_car + i - s_first + 1):
                    if k == self.params.max_car + i - s_first + 1:
                        p_k = 1 - sum_prob_k
                    else:
                        p_k = PolicyIteration.poisson(self.params.lambda_return_first, k)
                        sum_prob_k += p_k
                    sum_prob_l = 0
                    for l in range(self.params.max_car + j - s_second + 1):
                        if l == self.params.max_car + j - s_second + 1:
                            p_l = 1 - sum_prob_l
                        else:
                            p_l = PolicyIteration.poisson(self.params.lambda_return_second, l)
                            sum_prob_l += p_l
                        
                        value += p_i * p_j * p_k * p_l * (
                            r_i + r_j - cost + self.params.gamma * self.values[s_first - i + k, s_second - j + l])
        return value
    
    def print_pis(self):
        """
        Print policies
        """
        for idx, pi in enumerate(self.pis):
            plt.figure()
            plt.imshow(pi, origin='lower', interpolation='none', vmin=-self.params.max_move_loc, vmax=self.params.max_move_loc)
            plt.xlabel('#Cars at second location')
            plt.ylabel('#Cars at first location')
            plt.title('pi{:d} {:s}'.format(idx, self.problem_type))
            plt.colorbar()

    def print_V(self):
        """
        Print values
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(0, self.params.max_car + 1)
        Y = np.arange(0, self.params.max_car + 1)
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, self.values)
        plt.title('V {:s}'.format(self.problem_type))
                        
    
    @staticmethod
    def poisson(l, n):
        """
        :param l: lambda parameter of poisson distribution, rate
        :param n: n variable of poisson distribution, number of occurrences
        :return: probability of the event
        """
        return ((l**n)*np.exp(-n)) / math.factorial(n)
        
        
def exercise4_7():
    print('Exercise 4.7')

    # Set up parameters
    params = Params()

    for problem_type in params.problem_types:
        print('Problem type:', problem_type)

        # Set up the algorithm
        policy_iteration = PolicyIteration(params, problem_type)

        # Solve the problem
        policy_iteration.solve()

        # Show results
        policy_iteration.print_pis()
        policy_iteration.print_V()

exercise4_7()
plt.show()    
    

    