import tensorflow as tf

class Layer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(Layer, self).__init__()
        self.projection = tf.keras.layers.Dense(vocab_size)

    def __call__(self, inputs):
        logits = self.projection(inputs)  # Project to vocab_size
        probabilities = tf.nn.softmax(logits, axis=-1)  # Convert to probabilities
        return probabilities
