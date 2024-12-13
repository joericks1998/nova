import tensorflow as tf

class Layer(tf.Module):
    def __init__(self, d_model, dff, name=None):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)

    def __call__(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x
