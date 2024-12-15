import tensorflow as tf

class Layer(tf.Module):
    def __init__(self, d_model, dff, name=None):
        super(Layer, self).__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
    def __call__(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    @property
    def Parameters(self):
        return [self.dense1.kernel,
                self.dense1.bias,
                self.dense2.kernel,
                self.dense2.bias]
    
    def save(self, io_dir = None):
