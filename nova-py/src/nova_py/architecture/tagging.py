import tensorflow as tf

class Layer(tf.keras.layers.Layer):
    def __init__(self, num_features = None, num_groups = None, temperature = None):
        self.num_features = num_features
        self.num_groups = num_groups
        self.projection = tf.keras.layers.Dense(self.num_features*self.num_groups)
        self.temperature = temperature

    def __call__(self, batch):
        flat_batch = tf.reshape(batch, shape = (1,-1))
        logits = self.projection(flat_batch)
        logits = self.temperature * logits

    @property
    def Parameters(self):
        # Return the kernel (weights) and bias from the dense projection layer.
        return [
            self.projection.kernel,  # The weight matrix of the projection layer.
            self.projection.bias     # The bias vector of the projection layer.
        ]
