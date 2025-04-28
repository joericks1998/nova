import tensorflow as tf
from . import attention
import functools

class Layer(tf.keras.layers.Layer):
    def __init__(self, num_features = None, temperature = None, name = None, **kwargs):
        super().__init__(name=name, **kwargs)
        assert num_features is not None
        assert temperature is not None
        self.num_features = num_features
        self.temperature = temperature
        self.projection = tf.keras.layers.Dense(self.num_features)
        self.attention_pool = attention.Pool()
        self.pad_mask = tf.one_hot([0], depth=self.num_features) * -1e4
    # function for running top p sampling on a sequence
    # @tf.function(reduce_retracing=True)
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
        p_mask = cumulative_probs <= p
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
    # call the model
    # @tf.function(reduce_retracing=True)
    def call(self, batch, top_p=None, num_samples=1, training = False):
        if training:
            num_samples=1
        # apply attention pool
        pooled_batch = self.attention_pool(batch)
        logits = self.projection(pooled_batch)
        if not training:
            logits = self.temperature * logits
        # mask the first logit (reserved for padding)
        logits += tf.cast(self.pad_mask, self.compute_dtype)
        # apply softmax
        probabilities = tf.nn.softmax(logits)
        # If training, stop here and return raw probabilities
        if training:
            return probabilities[:,probabilities.shape[1]-1,:]
        else:
            # else call the sampler for top p sampling
            sampler = functools.partial(self.sample_top_p, p=top_p, num_samples=num_samples)
            sampled_tokens = tf.map_fn(sampler, probabilities)
            return tf.cast(sampled_tokens, dtype=tf.int32)
        return sampled_token  # Return the probabilities as the output.


    @property
    def Parameters(self):
        # Return the kernel (weights) and bias from the dense projection layer.
        return [
            self.projection.kernel,  # The weight matrix of the projection layer.
            self.projection.bias    # The bias vector of the projection layer.
        ] + self.attention_pool.Parameters # get attention pool parameters

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
