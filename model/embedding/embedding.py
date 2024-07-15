import numpy as np
import math

class Embedding:
    def __init__(self, v_dim):
        self.v_dim = v_dim
        self.hmap = {}
        self.M = None
        self.a = None
        self.Z = None
        self.dz = None
        self.dM = None
    
    def setParams(self, word):
        if word in self.hmap.keys():
            pass
        else:
            vec = math.heInit((1,self.v_dim))
            print(vec)
            if self.M is None:
                self.hmap[word] = 0
                self.M = vec
            else:
                self.M = np.hstack((self.M, vec))
                self.hmap[word] = self.M.shape[1]-1
        self.a = np.array(([1 if i == self.hmap[word] else 0 for i in range(0, self.M.shape[1])],)).T
        return 