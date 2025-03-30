import tensorflow as tf
from . import attention

class Layer(tf.keras.layers.Layer):
    def __init__(self, num_features = None, num_groups = None, temperature = None, name = None):
        super().__init__(name=name)
        assert num_features is not None
        assert num_groups is not None
        assert temperature is not None
        self.num_features = num_features
        self.num_groups = num_groups
        self.temperature = temperature
        self.projection = tf.keras.layers.Dense(self.num_features*self.num_groups)
        self.attention_pool = attention.Pool()

    # translate output token to tag format
    def _tag_translate(self, sampled_token):
        group = 0
        for i in range(sampled_token):
            if i % self.num_features==0:
                group+=1
        return tf.constant([sampled_token%self.num_features, group])

    @tf.function(reduce_retracing=True)
    def __call__(self, batch, top_p=0.5, num_samples=1, training = False):
        if training:
            num_samples=1
        else:
            num_samples=num_samples
        pooled_batch = self.attention_pool(batch)
        logits = self.projection(pooled_batch)
        if not training:
            logits = self.temperature * logits
        probabilities = tf.nn.softmax(logits)
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
