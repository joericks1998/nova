import tensorflow as tf
import numpy as np

class Layer(tf.keras.layers.Layer):
    # Initialize the EmbeddingLayer with the given embedding dimension and optional name
    def __init__(self, d_model=None, N=None, name=None, **kwargs):
        if not d_model or not N:
            msg = '''
                Values for N and d_model must be passed.
            '''
            raise ValueError(msg)
        super(Layer, self).__init__(name=name, **kwargs)
        self.d_model = d_model # Store the dimension of the embeddings for serialization
        self.N = N
        self.initializer = tf.keras.initializers.GlorotUniform()
    # overriding build for mixed precision
    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embedding_matrix",
            shape=(self.N, self.d_model),
            initializer=self.initializer,
            trainable=True,
            dtype=self.dtype
        )
    # Method to retrieve or create the embedding for a given word
    @tf.function(reduce_retracing=True)
    def call(self, tokens):
        return tf.nn.embedding_lookup(self.embeddings, tokens)
        # Retrieve the embedding for the given word using its index

    #get config for serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "N": self.N
        })
        return config

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    #parameters getter for model training
    @property
    def Parameters(self):
        return [self.embeddings]
