import tensorflow as tf
from . import attention
import functools

# final layer in autoregression
class Layer(tf.keras.layers.Layer):
    #constructer
    def __init__(self, d_model, num_features, temperature, **kwargs):
        super().__init__(**kwargs)
        assert num_features is not None
        assert temperature is not None
        self.d_model = d_model
        self.num_features = num_features
        self.temperature = temperature

    def build(self, input_shape):
        # This is typically used as the output layer of a generative model.
        self.projection = tf.keras.layers.Dense(self.num_features, input_dim=self.d_model)

    # function for running top p sampling on a sequence
    # @tf.function(reduce_retracing=True)
    def rowwise_mode(self, tensor):
        def get_mode(row):
            unique_vals, _, counts = tf.unique_with_counts(row)
            mode_index = tf.argmax(counts)
            return unique_vals[mode_index]
        return tf.map_fn(get_mode, tensor, fn_output_signature=self.compute_dtype)
    #nucleus sampling function
    @tf.function(reduce_retracing=True)
    def sample_top_p(self, p_sequence, p=None, num_samples=1):
        # Sort the probabilities in descending order
        sorted_probs = tf.sort(p_sequence, direction='DESCENDING')
        sorted_indices = tf.argsort(p_sequence, direction='DESCENDING')
        # Compute the cumulative probabilities
        cumulative_probs = tf.math.cumsum(sorted_probs, axis=-1)
        # force p to be bigger than the smallest p in cumulative probabilities
        # this essentially so we force sampling at least one token
        p = tf.maximum(cumulative_probs[0, 0], p)
        # Create a mask for tokens where cumulative probability <= ps
        p_mask = cumulative_probs <= p
        # Force at least one True if all are False
        # Filter out tokens not in the top-p set
        top_p_probs = tf.ragged.boolean_mask(sorted_probs, p_mask).to_tensor(default_value=0.0)
        top_p_indices = tf.ragged.boolean_mask(sorted_indices, p_mask).to_tensor(default_value=0)
        # Normalize the probabilities of the top-p tokens
        top_p_probs /= tf.reduce_sum(top_p_probs, axis=1, keepdims=True)
        # Sample from the top-p tokens
        sampled_indecies = tf.random.categorical(tf.math.log(top_p_probs), num_samples=num_samples)
        # Map back to the original token IDs
        sampled_tokens = tf.gather(top_p_indices, sampled_indecies, batch_dims=1)
        return tf.cast(sampled_tokens, dtype=self.compute_dtype)
    # call the model
    @tf.function(reduce_retracing=True)
    def call(self, batch, top_p=None, num_samples=1, training = False):
        pad_mask = tf.one_hot([0], depth=self.num_features) * -1e4
        if training:
            num_samples=1
        # apply attention pool
        logits = self.projection(batch)
        if not training:
            logits = self.temperature * logits
        # mask the first logit (reserved for padding)
        pad_mask = tf.one_hot([0], depth=self.num_features) * -1e4
        logits += tf.cast(pad_mask, self.compute_dtype)
        # apply softmax
        probabilities = tf.nn.softmax(logits)
        sampler = functools.partial(self.sample_top_p, p=top_p, num_samples=num_samples)
        sampled_tokens = tf.map_fn(sampler, probabilities)
        modes = self.rowwise_mode(tf.reshape(sampled_tokens, (tf.shape(sampled_tokens)[1], -1)))
        return tf.cast(modes, dtype=tf.int32)


    @property
    def Parameters(self):
        # Return the kernel (weights) and bias from the dense projection layer.
        return [
            self.projection.kernel,  # The weight matrix of the projection layer.
            self.projection.bias    # The bias vector of the projection layer.
        ] # get attention pool parameters

    #parameters getter for model training
    def get_config(self):
        config = super().get_config()
        config.update({
                "num_features": self.num_features,
                "num_groups": self.num_groups,
                "temperature": self.temperature
        })
        return config

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls,config):
        return cls(**config)
