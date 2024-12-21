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
        self.TransitionMatrix = tf.zeros(shape = (1, len(tags)-1))
        self.TransitionStates = {}
        self.tags = {}
        i = 0
        for tag in tags[:len(tags)-1]:
            self.tags[tag] = i
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
