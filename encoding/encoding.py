import re
import tensorflow as tf
import numpy as np
import os
import json

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

class SemanticParser(tf.Module):
    def __init__(self, tags = None, n_limit = None, predefinitions = None,
                transition_matrix = None, transition_states = None):
        self.max_combos = (len(tags)-1)**n_limit
        if not transition_matrix:
            self.TransitionMatrix = tf.zeros(shape = (1, len(tags.values())))
        else:
            self.TransitionMatrix = transition_matrix
        if not transition_states:
            self.TransitionStates = {}
        else:
            self.TransitionStates = transition_states
        self.tags = tags
        self.predefinitions = predefinitions
    def __call__(self, sequence, memory = None):
        #pretag tokens in sequence
        tagged_data = []
        for token in sequence:
            tagged = False
            for k in self.predefinitions.keys():
                if token in self.predefinitions[k]:
                    tagged_data.append((token, k))
                    tagged = True
            if not tagged:
                tagged_data.append((token, None))
        tag_dict = dict(tagged_data)
        i = 0
        for k in tag_dict.keys():
            tag_seq = list(tag_dict.values())
            if not tag_dict[k]:
                if i == 0:
                    idx = self.TransitionStates[""]
                else:
                    idx = self.TransitionStates[" -> ".join([tkn for tkn in tag_seq[:i]])]
                vec = tf.nn.embedding_lookup(self.TransitionMatrix, idx)
                tag_tensor = tf.Variable([list(map(float, self.tags.values()))])
                dot = tf.tensordot(tag_tensor,vec, axes=1)[0]
                tag_dict[k] = [k for k in self.tags.keys() if self.tags[k] == dot][0]
            i+=1
        encoded_arr = []
        for k,v in tag_dict.items():
            if v == '~pad~':
                continue
            elif v == '~relation~':
                encoded_arr.append(k)
            else:
                if v=='~value~':
                    if k.isdigit():
                        v = '~value.int~'
                    elif k.isdecimal():
                        v = '~value.float~'
                    elif k.lower() in ("null", "none"):
                        v = '~value.null~'
                    else:
                        v = 'value.string~'
                encoded_arr.append(v)
        return " ".join(encoded_arr)
    def addTransition(self, tag_seq, targets):
        if isinstance(targets, str):
            targets = [targets]
        print(targets)
        for tkn in targets:
            if not tkn in self.tags.keys():
                msg = "Invalid Target Tag"
                raise ValueError(msg)
        self.TransitionStates[tag_seq] = self.TransitionMatrix.shape[0]
        depth = len(self.tags.keys())
        transition = tf.one_hot([self.tags[t] for t in targets], depth)
        self.TransitionMatrix = tf.concat([self.TransitionMatrix, transition], axis = 0)
        return
    def save(self, path = None):
        os.makedirs(path, exist_ok = True)
        with open(os.path.join(path, "predefined_tags.json"),  "w") as f:
            json.dump(self.predefinitions, f)
        with open(os.path.join(path, "tags.json"),  "w") as f:
            json.dump(self.tags, f)
        with open(os.path.join(path, "transition_states.json"),  "w") as f:
            json.dump(self.TransitionStates, f)
        np.save(os.path.join(path,"transition_matrix.npy"), self.TransitionMatrix.numpy())
        return
    @classmethod
    def load(cls):
        return
