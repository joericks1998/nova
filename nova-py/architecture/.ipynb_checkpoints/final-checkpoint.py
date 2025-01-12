import tensorflow as tf
from static import constants

# Define a custom layer class, inheriting from `tf.keras.layers.Layer`.
class Layer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        # Initialize the parent `tf.keras.layers.Layer` class.
        super(Layer, self).__init__()
        # Define a dense layer to project inputs to `vocab_size` dimensions.
        # This is typically used as the output layer of a generative model.
        self.projection = tf.keras.layers.Dense(vocab_size)

    # Define the forward pass logic for the layer.
    def __call__(self, inputs):
        # Apply the dense layer to project inputs to `vocab_size` dimensions.
        logits = self.projection(inputs)
        # Apply temperature scaling for randomness
        scaled_logits = logits / constants.temperature
        # Use softmax to convert logits into probabilities across the vocabulary.
        probabilities = tf.nn.softmax(logits, axis=-1)
        return probabilities  # Return the probabilities as the output.

    # Serialize the layer's configuration into a dictionary.
    def get_config(self):
        return master_config(Layer.__init__) # Return the stored configuration.

    # Reconstruct the layer from a serialized configuration.
    @classmethod
    def from_config(cls, config):
        return cls(**config)  # Use the configuration dictionary to instantiate the layer.

    # Define a property to retrieve the trainable parameters (weights and biases).
    @property
    def Parameters(self):
        # Return the kernel (weights) and bias from the dense projection layer.
        return [
            self.projection.kernel,  # The weight matrix of the projection layer.
            self.projection.bias     # The bias vector of the projection layer.
        ]
