import pdb
from dist import DDist, uniform_dist, delta_dist, mixture_dist
from util import *
import random

class MDP:
    # Needs the following attributes:
    # states: list or set of states
    # actions: list or set of actions
    # discount_factor: real, greater than 0, less than or equal to 1
    # start: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    # These are functions:
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward

    def __init__(self, states, actions, transition_model, reward_fn, 
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

# Perform value iteration on an MDP, also given an instance of a q
# function.  Terminate when the max-norm distance between two
# successive value function estimates is less than eps.
# interactive_fn is an optional function that takes the q function as
# argument; if it is not None, it will be called once per iteration,
# for visuzalization

# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values This must be
# initialized before interactive_fn is called the first time.

def value_iteration(mdp, q, eps = 0.01, interactive_fn = None,
                    max_iters = 10000):
    # Your code here
    for it in range(max_iters):
        q_new = q.copy()
        for s in q_new.states:
            for a in q_new.actions:
                t_q = float(np.dot(rv([mdp.transition_model(s, a).prob(elt) for elt in q.states]), cv([value(q, s) for s in q.states])))
                v = mdp.reward_fn(s, a) + mdp.discount_factor * t_q
                q_new.set(s, a, v)
        if max([abs(q_new.get(s, a) - q.get(s, a)) for s in q_new.states for a in q_new.actions]) < eps:
            return q_new
        q = q_new
    return q
''' 
def v(s):
        return value(q,s)
    for it in range(max_iters):
        new_q = q.copy()
        delta = 0
        for s in mdp.states:
            for a in mdp.actions:
                new_q.set(s, a, mdp.reward_fn(s, a) + mdp.discount_factor * \
                          mdp.transition_model(s, a).expectation(v))
                delta = max(delta, abs(new_q.get(s, a) - q.get(s, a)))
        if delta < eps:
            return new_q
        q = new_q
    return q
'''

# Compute the q value of action a in state s with horizon h, using
# expectimax
def q_em(mdp, s, a, h):
    # Your code here
    if h == 0 :
        return 0
    else:
        return mdp.reward_fn(s, a) + mdp.discount_factor * \
            sum([p*max([q_em(mdp, sp, ap, h-1) for ap in mdp.actions]) for (sp, p) in mdp.transition_model(s, a).d.items()])

# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    """ Return Q*(s,a) based on current Q

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q_star = value(q,0)
    >>> q_star
    10
    """
    # Your code here
    Q_s = [q.get(s, a) for a in q.actions]
    index_max_a = Q_s.index(max(Q_s))
    a = q.actions[index_max_a]
    return q.get(s, a)
    #return max(q.get(s, a) for a in q.actions)

# Given a state, return the action that is greedy with reespect to the
# current definition of the q function
def greedy(q, s):
    """ Return pi*(s) based on a greedy strategy.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> greedy(q, 0)
    'c'
    >>> greedy(q, 1)
    'b'
    """
    # Your code here
    Q_s = [q.get(s, a) for a in q.actions]
    index_max = Q_s.index(max(Q_s))
    a = q.actions[index_max]
    return a
    #return argmax(q.actions, lambda a: q.get(s, a))
    
def epsilon_greedy(q, s, eps = 0.5):
    """ Return an action.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> eps = 0.
    >>> epsilon_greedy(q, 0, eps) #greedy
    'c'
    >>> epsilon_greedy(q, 1, eps) #greedy
    'b'
    """
    if random.random() < eps:  # True with prob eps, random action
        # Your code here
        return uniform_dist(q.actions).draw() # we explore
    else:
        # Your code here
        return greedy(q, s) # we choose a greedy action

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]
