import numpy as np

# Math functions required

def ReLU(v):
    return np.array([np.maximum(e, 0) for e in v])

def softmax(M):
    exp_values = np.exp(M - np.max(M))  # for numerical stability
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)

def heaviside(M):
    return np.array([1 if val >= 0 else 0 for val in M]).T

# the gradient of the softmax function is a little trickier, as this is a vector function and not a scalar

def softmax_gradient(s):
    """
    Compute the Jacobian matrix of the softmax function.

    Parameters:
    softmax_output (numpy.ndarray): The output of the softmax function, a 1D array.

    Returns:
    numpy.ndarray: The Jacobian matrix of the softmax function.
    """
    s = s.reshape(1, -1)
    print(s)
    jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
    return jacobian_matrix

# calculate loss
def loss(Y, Y_hat):
    entropy = -np.sum(Y * np.log(Y_hat)) 
    return entropy

def gradientLoss(v_prev, v_curr):
    dMSE = 2*(v_prev - v_curr)
    return dMSE

def clip(gradient, n):
    return np.clip(gradient, n, -n)

def vectorInit(dim):
    return np.random.random((self.v_dim, 1)) - 1

def heInit(shape):
    fan_in = shape[1]
    scale = np.sqrt(2/fan_in)
    return np.random.randn(*shape) * scale