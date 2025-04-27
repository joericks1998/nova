import tensorflow as tf
from . import masking, attention
import functools

# Define a custom layer class, inheriting from `tf.keras.layers.Layer`.
class Layer(tf.keras.layers.Layer):
    def __init__(self, d_model ,vocab_len, temperature, **kwargs):
        # Initialize the parent `tf.keras.layers.Layer` class.
        super(Layer, self).__init__(**kwargs)
        # Define a dense layer to project inputs to `vocab_size` dimensions.
        self.vocab_len = vocab_len
        # This is typically used as the output layer of a generative model.
        self.projection = tf.keras.layers.Dense(vocab_len, input_dim= d_model)
        # Define the model temperature
        self.temperature = temperature
        # Manually build layers
        self.projection.build((None, d_model))
        self.attention_pool = attention.Pool()

    # function for running top p sampling on a sequence
    @tf.function(reduce_retracing=True)
    def sample_top_p(self, p_sequence, p=None, num_samples=1):
        # Sort the probabilities in descending order
        sorted_probs = tf.sort(p_sequence, direction='DESCENDING'),
        sorted_indices = tf.expand_dims(tf.argsort(p_sequence, direction='DESCENDING'), axis = 0)
        # Compute the cumulative probabilities
        cumulative_probs = tf.math.cumsum(sorted_probs, axis=-1)
        # force p to be bigger than the smallest p in cumulative probabilities
        # this essentially so we force sampling at least one token
        p = tf.maximum(cumulative_probs[0, 0], p)
        # Create a mask for tokens where cumulative probability <= ps
        # print(p)
        p_mask = cumulative_probs <= p
        # print(p_mask)
        # print(cumulative_probs)
        # Force at least one True if all are False
        # Filter out tokens not in the top-p set
        top_p_probs = tf.boolean_mask(sorted_probs, p_mask)
        top_p_indices = tf.boolean_mask(sorted_indices, p_mask)
        # Normalize the probabilities of the top-p tokens
        top_p_probs /= tf.reduce_sum(top_p_probs)
        # Sample from the top-p tokens
        sampled_index = tf.random.categorical(tf.math.log([top_p_probs]), num_samples=num_samples)[0,0]
        # Map back to the original token IDs
        sampled_token = tf.gather(top_p_indices, sampled_index)
        return tf.cast(sampled_token, dtype=self.compute_dtype)
    # Define the forward pass logic for the layer.
    @tf.function(reduce_retracing=True)
    def call(self, batch, top_p = 0.9, num_samples=1, training = False, hard_mask = [162]):
        # Apply the dense layer to project inputs to `vocab_size` dimensions.
        logits = self.projection(batch)
        batch_size = tf.shape(logits)[0]
        sequence_len = tf.shape(logits)[1]
        final_logits = tf.gather(logits, indices=sequence_len-1, axis = 1)
        # print(logits)
        # Apply temperature scaling for randomness (if not training)
        if not training:
            scaled_logits = logits * self.temperature
            # Apply hard mask on tokens
            # logits = masking.simple_mask(logits, hard_mask, dtype=self.compute_dtype)
        # Use softmax to convert logits into probabilities across the vocabulary.
        probabilities = tf.nn.softmax(final_logits, axis=1)
        # If training, stop here and return raw probabilities
        if training:
            return probabilities[:,probabilities.shape[1]-1,:]
        else:
            # else call the sampler for top p sampling
            # print(probabilities)
            sampler = functools.partial(self.sample_top_p, p=top_p, num_samples=num_samples)
            sampled_tokens = tf.map_fn(sampler, probabilities)
            return tf.cast(sampled_tokens, dtype=tf.int32)

    # Serialize the layer's configuration into a dictionary.
    def get_config(self):
        config = super().get_config()
        config.update({
                "vocab_len": self.vocab_len,
                "temperature": self.temperature
        })
        return config # Return the stored configuration.

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
