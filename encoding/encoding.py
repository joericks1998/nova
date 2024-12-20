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
    def __init__(self, tags, n_limit = None, predefinitions = {}):
        self.max_combos = (len(tags)-1)**n_limit
        self.TransitionMatrix = tf.zeros(shape = (1, len(tags)))
        self.TransitionStates = {}
        self.tags = {}
        i = 0
        for tag in tags[:len(tags)-1]:
            self.tags[i] = tag
            i+=1
        self.predefinitions = predefinitions
        self.tag_map = {}
    def __call__(self, sequence):
        #pretag sequences
        tagged_data = []
        for token in sequence:
            tagged = False
            for k in self.predefinitions.keys():
                if token in self.predefinitions[k]:
                    tagged_data.append((token, k))
                    tagged = True
            if not tagged:
                tagged_data.append((token, None))
        return dict(tagged_data)

    def addTransition(self, tag_seq, target):
        if target not in self.tags.values():
            msg = "Invalid Target Tag"
            raise ValueError(msg)
        self.TransitionStates[self.TransitionMatrix.shape[0]] = tag_seq
