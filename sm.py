from re import S
from unicodedata import digit

from numpy import append
from pyparsing import Word
from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        state = self.start_state
        output_seq = []
        for input in input_seq:
            state = self.transition_fn(state, input)
            output_seq.append(self.output_fn(state))
        
        return output_seq
            


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = (0, 0) # Change

    def transition_fn(self, s, x):
        # Your code here
        (carry, digit) = s 
        (d_0, d_1) = x 
        
        '''if (d_0 + d_1)  == 2:
            if carry == 1:
                return (1, 1)
            else:
                return (1, 0)
        elif (d_0 + d_0) == 1:
            if carry == 1:
                return (1, 0)
            else:
                return (0, 1)
        else:
            if carry == 1:
                return (0, 1)
            else:
                return (0, 0)'''
        total = d_0 + d_1 + carry
        return 1 if total > 1 else 0, total % 2

    def output_fn(self, s):
        # Your code here
        (carry, digit) = s 
        return digit

    def transduce(self, input_seq):
        (d_0, d_1) = input_seq[-1]
        if (d_0 + d_1) > 0:
            input_seq.append((0, 0))
        
        return super().transduce(input_seq)

class Reverser(SM):
    start_state = (-1, [])

    def transition_fn(self, s, x):
        # Your code here
        (pos, seq) = s
        if pos == -1 and x!= 'end':
            seq.append(x)
            return (-1, seq)
        if pos == -1 and x == 'end':
            return (1, seq)
        if pos == 1 and x!= 'end':
            return (1, seq)
    
    ''' 
    start_state = ([], 'input')

    def transition_fn(self, s, x):
        (symbols, mode) = s
        if x == 'end':
            return symbols, 'output'
        elif mode == 'input':
            return symbols + [x], mode
        else:
            return symbols[:-1], mode
    '''        

    ''' 
    def output_fn(self, s):
        (symbols, mode) = s
        if mode == 'output' and len(symbols) > 0:
            return symbols[-1]
        else:
            return None
    '''
    
    def output_fn(self, s):
        # Your code here
        (pos, seq) = s 
        if pos == -1:
            return None
        if pos == 1:
            if len(seq) != 0:
                return seq.pop()
            else:
                return None

class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        self.Wsx = Wsx 
        self.Wss = Wss 
        self.Wo = Wo 
        self.Wss_0 = Wss_0 
        self.Wo_0 = Wo_0 
        self.f1 = f1 
        self.f2 = f2 
        self.m = Wss.shape[1]
        self.l = Wsx.shape[0]
        self.n = Wss_0.shape[0]
        self.start_state = np.zeros([self.m, 1])

    def transition_fn(self, s, i):
        # Your code here
        return self.f1(np.dot(self.Wss, s) + np.dot(self.Wsx, i) + self.Wss_0)

    def output_fn(self, s):
        # Your code here
        return self.f2(self.Wo @ s + self.Wo_0)
    
    