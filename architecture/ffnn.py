import tensorflow as tf

# Define a custom layer class, inheriting from `tf.Module`.
class Layer(tf.Module):
    def __init__(self, d_model, dff, name=None):
        # Initialize the parent `tf.Module` class with an optional name.
        super(Layer, self).__init__(name=name)

        # Store the configuration of the layer for serialization.
        self.config = {
            "d_model": d_model,  # Output dimensionality.
            "dff": dff           # Intermediate feedforward dimensionality.
        }

        # Define the first dense layer with a ReLU activation function.
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')

        # Define the second dense layer without activation to project back to `d_model` dimensions.
        self.dense2 = tf.keras.layers.Dense(d_model)

    # Define the forward pass logic for the layer.
    def __call__(self, x):
        # Pass the input `x` through the first dense layer.
        x = self.dense1(x)
        # Pass the output of the first dense layer through the second dense layer.
        x = self.dense2(x)
        return x  # Return the final output.

    # Serialize the layer's configuration into a dictionary.
    def get_config(self):
        return self.config  # Return the stored configuration.

    # Reconstruct the layer from a serialized configuration.
    @classmethod
    def from_config(cls, config):
        return cls(**config)  # Use the configuration to instantiate the layer.

    # Define a property to retrieve all trainable parameters (weights and biases) in the layer.
    @property
    def Parameters(self):
        # Return the kernel (weights) and bias for both dense layers in a list.
        return [
            self.dense1.kernel,  # Weights of the first dense layer.
            self.dense1.bias,    # Biases of the first dense layer.
            self.dense2.kernel,  # Weights of the second dense layer.
            self.dense2.bias     # Biases of the second dense layer.
        ]
