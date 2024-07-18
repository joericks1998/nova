import numpy as np
import local_math
from .constants import embd_constants

class Embedding:
    def __init__(self, v_dim):
        self.v_dim = v_dim
        self.hmap = {}
        self.M = None
        self.Z = None
        self.dz = None
        self.dM = None
    
    def setParams(self, word):
        if word in self.hmap.keys():
            pass
        else:
            vec = local_math.heInit((1,self.v_dim))
            if self.M is None:
                self.hmap[word] = 0
                self.M = vec
            else:
                self.M = np.hstack((self.M, vec))
                self.hmap[word] = self.M.shape[1]-1
        # will need this snippet in order to one hot the inputs for each neural network
        # self.a = np.array(([1 if i == self.hmap[word] else 0 for i in range(0, self.M.shape[0])],)).T
        return 