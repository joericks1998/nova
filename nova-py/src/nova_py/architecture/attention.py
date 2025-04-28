import tensorflow as tf
from . import masking

class PerformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, kernel_transformation=None, autoregressive = False, name=None, **kwargs):
        super(PerformerLayer, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.kernel_transformation = kernel_transformation or self.default_kernel_transformation
        self.autoregressive = autoregressive

        assert d_model % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.depth = d_model // num_heads
        if self.autoregressive:
            self.initializer = tf.keras.initializers.LecunNormal()
        else:
            self.initializer = tf.keras.initializers.LecunNormal()
        # Initialize weights
        self.dense = tf.keras.layers.Dense(d_model)
        # layer normalization
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
    # build layer and add weights
    def build(self, input_shape):
        self.wq = self.add_weight(
            name="wQ",
            shape=(self.d_model, self.d_model),
            initializer=self.initializer,
            trainable=True,
            dtype=self.dtype
        )
        self.wk = self.add_weight(
            name="wK",
            shape=(self.d_model, self.d_model),
            initializer=self.initializer,
            trainable=True,
            dtype=self.dtype
        )
        self.wv = self.add_weight(
            name="wV",
            shape=(self.d_model, self.d_model),
            initializer=self.initializer,
            trainable=True,
            dtype=self.dtype
        )
    #define kernel transformation rule
    @tf.function(reduce_retracing=True)
    def default_kernel_transformation(self, x):
        return tf.nn.relu(x) + 1e-6

    @tf.function(reduce_retracing=True)
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result to shape (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # main call
    # @tf.function(reduce_retracing=True)
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]
        # creating lookahead mask
        if self.autoregressive:
            lookahead_mask = masking.create_look_ahead_mask(seq_len)
        # expanded padding mask
        expanded_mask = tf.expand_dims(mask, axis=1)
        #dotting q,k,v
        q = tf.tensordot(q, self.wq, axes=[[2], [0]])  # (batch_size, seq_len, d_model)
        k = tf.tensordot(k, self.wk, axes=[[2], [0]])  # (batch_size, seq_len, d_model)
        v = tf.tensordot(v, self.wv, axes=[[2], [0]])  # (batch_size, seq_len, d_model)
        # normalize layers
        q = self.layernorm(q)
        k = self.layernorm(k)
        v = self.layernorm(v)
        # split_heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len, depth)
        # Apply kernel transformation
        q_prime = self.kernel_transformation(q)  # Apply kernel transformation
        k_prime = self.kernel_transformation(k)
        # Apply pad mask
        k_prime = k_prime * expanded_mask[..., None]
        v = v * expanded_mask[..., None]
        # FAVOR+ Mechanism
        kv = tf.einsum('...nd,...ne->...de', k_prime, v)
        z = 1.0 / (tf.einsum('...nd,...d->...n', q_prime, tf.reduce_sum(k_prime, axis=-2)) + 1e-6)
        attention_output = tf.einsum('...nd,...de,...n->...ne', q_prime, kv, z)
        # apply mask
        if self.autoregressive:
            attention_output = masking.masked_attention(q_prime, k_prime, v, dtype=self.compute_dtype, mask=lookahead_mask)

        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        output = self.dense(attention_output)
        #normalize outputs
        return self.layernorm(output)
        # apply pad mask


    #get config for serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "kernel_transformation": self.kernel_transformation
        })
        return config

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    #parameters getter for model training
    @property
    def Parameters(self):
        return [self.wq, self.wk, self.wv] + self.layernorm.trainable_variables

# attention pooling for tagging
class Pool(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.query_layer = tf.keras.layers.Dense(1)  # Score for each timestep

    @tf.function(reduce_retracing=True)
    def call(self, inputs, mask=None):
        # inputs: (seq_len, activation_dim) or (batch_size, seq_len, activation_dim)
        scores = self.query_layer(inputs)  # (batch_size, seq_len, 1) or (seq_len, 1)
        scores = tf.squeeze(scores, axis=-1)  # Remove last dim: (batch_size, seq_len) or (seq_len,)
        weights = tf.nn.softmax(scores, axis=-1)  # Softmax over time
        if mask is not None:
            weights *= tf.cast(mask, self.dtype)
            weights /= tf.reduce_sum(weights, axis=-1, keepdims=True)
        pooled = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=-2)  # Weighted sum over seq_len
        return pooled  # (batch_size, activation_dim) or (activation_dim,)

    @property
    def Parameters(self):
        return [self.query_layer.kernel, self.query_layer.bias]
