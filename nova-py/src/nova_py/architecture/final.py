import tensorflow as tf
from . import masking

# Define a custom layer class, inheriting from `tf.keras.layers.Layer`.
class Layer(tf.keras.layers.Layer):
    def __init__(self, d_model ,vocab_len, temperature):
        # Initialize the parent `tf.keras.layers.Layer` class.
        super(Layer, self).__init__()
        # Define a dense layer to project inputs to `vocab_size` dimensions.
        self.vocab_len = vocab_len
        # This is typically used as the output layer of a generative model.
        self.projection = tf.keras.layers.Dense(vocab_len, input_dim= d_model)
        # Define the model temperature
        self.temperature = temperature
        # Manually build layers
        self.projection.build((None, d_model))

    # Define the forward pass logic for the layer.
    # @tf.function(reduce_retracing=True)
    def __call__(self, inputs, top_p = 0.9, num_samples=1, training = False, hard_mask = [162]):
        # Apply the dense layer to project inputs to `vocab_size` dimensions.
        logits = self.projection(inputs)
        # Apply temperature scaling for randomness (if not training)
        if not training:
            scaled_logits = logits * self.temperature
            # Apply hard mask on tokens
            logits = masking.simple_mask(logits, hard_mask)
        # Use softmax to convert logits into probabilities across the vocabulary.
        probabilities = tf.nn.softmax(logits, axis=-1)
        # If training, stop here and return raw probabilities
        if training:
            return probabilities[:,probabilities.shape[1]-1,:]
        # Sort the probabilities in descending order
        sorted_probs, sorted_indices = tf.sort(probabilities, direction='DESCENDING'), tf.argsort(probabilities, direction='DESCENDING')
        # Compute the cumulative probabilities
        cumulative_probs = tf.math.cumsum(sorted_probs, axis=-1)
        # Create a mask for tokens where cumulative probability <= p
        mask = cumulative_probs <= top_p
        # Ensure at least one token is included
        # mask = tf.concat([[True], mask[:-1]], axis=0)
        # Filter out tokens not in the top-p set
        top_p_probs = tf.boolean_mask(sorted_probs, mask)
        top_p_indices = tf.boolean_mask(sorted_indices, mask)
        # Normalize the probabilities of the top-p tokens
        top_p_probs /= tf.reduce_sum(top_p_probs)
        # Sample from the top-p tokens
        sampled_index = tf.random.categorical(tf.math.log([top_p_probs]), num_samples=num_samples)[0,0]
        # Map back to the original token IDs
        sampled_token = tf.gather(top_p_indices, sampled_index)
        return sampled_token  # Return the probabilities as the output.

    # Serialize the layer's configuration into a dictionary.
    def get_config(self):
        return {
            "vocab_len": self.vocab_len,
            "temperature": self.temperature
        }# Return the stored configuration.

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
