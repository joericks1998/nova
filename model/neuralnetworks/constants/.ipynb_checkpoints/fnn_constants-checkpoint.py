import numpy as np
import rnn_static


class Layer:
    def __init__(self, W=None, a=None, b=None):
        self.W = W
        self.a = a
        self.b = b
        self.Z = None
        self.dz = None
        self.dW = None
        self.db = None