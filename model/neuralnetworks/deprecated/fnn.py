import numpy as np
import local_math
from .constants import fnn_constants



class Layer:
    def __init__(self, W=None, a=None, b=None):
        self.W = W
        self.a = a
        if np.any(W):
            self.b = np.zeros((1, W.shape[0]))
        else:
            self.b = None
        self.Z = None
        self.dz = None
        self.dW = None
        self.db = None

class Neural:
    def __init__(self, output_dim = None, embedding_instance = None, hidden_dim = None, network_dim = 5):
        self.token = None
        self.embedding = embedding_instance
        self.layers = []
        self.hidden_dim = hidden_dim
        self.network_dim = network_dim
        self.Y_hat = None
        self.Y = None
        for i in range(network_dim-1):
            lyr = Layer(W=local_math.heInit((hidden_dim, hidden_dim)))
            self.layers.append(lyr)
        lyr = Layer(W=local_math.heInit((output_dim, hidden_dim)))
        self.layers.append(lyr)

    # getting the answer from the network
    @property
    def Answer(self):
        i_max = np.argmax(self.Y_hat)
        return self.options[i_max]

    # getting and setting the correct answer for training
    @property
    def CorrectAnswer(self):
        return self.correct_answer

    #setter
    @CorrectAnswer.setter
    def CorrectAnswer(self, answer):
        if any([answer == i for i in self.options]):
            self.correct_answer = answer
            self.Y = np.array(([1 if answer == i else 0 for i in self.options],)).T
        else:
            msg = "answer not in options"
            raise ValueError(msg)

    #feed forward
    def feedForward(self):
        if not self.token:
            msg = "Please set token before feeding forward!"
            raise ValueError(msg)
        plyr = Layer()
        ebd = self.embedding
        ebd.setParams(self.token)
        ebd.Z = np.dot(ebd.M, ebd.a)
        plyr.a = ReLU(ebd.Z)
        i = 1
        for lyr in self.layers:
            lyr.Z = np.dot(lyr.W, plyr.a) - lyr.b
            if i == self.network_dim:
                self.Y_hat = softmax(lyr.Z)
            else:
                lyr.a = ReLU(lyr.Z)
                plyr = lyr
                i += 1
        return

    def backPropagate(self, learning_rate):
        i = len(self.layers)-1
        while i >= 0:
            prev = self.layers[i-1]
            lyr = self.layers[i]
            if i == len(self.layers)-1:
                dA = self.Y_hat - self.Y
                if all(i == 0 for i in dA):
                    print("loss is zero")
                    return
                # clip gradient
                dA = clip(dA, 1)
                lyr.dz = dA
            else:
                dA = np.dot(prev.W.T , prev.Z)
                # clip gradient
                dA = clip(dA, 1)
                lyr.dz = dA * heaviside(lyr.Z)
            lyr.dW = np.dot(lyr.W.T, lyr.dz)
            lyr.db = np.sum(lyr.dz, axis = 1, keepdims = True)
            lyr.W -= learning_rate*lyr.dW.T
            lyr.b -= learning_rate*lyr.db
            i -= 1
        prev = self.layers[0]
        ebd = self.embedding
        dA = np.dot(prev.W , prev.Z)
        # clip gradient
        dA = clip(dA, 1)
        ebd.dz = dA * heaviside(ebd.Z)
        ebd.dM = np.dot(ebd.M.T, ebd.dz)
        ebd.M -= learning_rate*ebd.dM.T
        self.embedding = ebd
        return
    # training
    def train(self, epochs = 10, learning_rate = 0.2):
        if not self.CorrectAnswer:
            msg = "No training data has been added"
            raise ValueError(msg)
        print("Training network...")
        for i in range(epochs):
            self.feedForward()
            self.backPropagate(learning_rate)
        print("Done.")
        
    
