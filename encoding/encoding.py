import re
import tensorflow as tf

class Memory:
    def __init__(self):
        self.Q = {}
    def pop(self, cls = None):
        out = self.Q[cls][len(self.Q[cls])-1]
        if len(self.Q[cls]) > 2:
            self.Q[cls] = self.Q[cls][:len(self.Q[cls])-2]
        return out
    def push(self, token, cls = None):
        self.Q[cls].append(token)
        return

class Encoder(tf.Module):
    def __init__(self, *cats, n_limit = None):
        self.max_combos = len(clss)**n_limit
        self.TransitionMatix = None
        self.TransitionStates = {}
        self.cats = list(cats)
        self.cat_idxs = [i for i in range(0, len(cats)-1)]

    def __call__(self, seq):
        pass
