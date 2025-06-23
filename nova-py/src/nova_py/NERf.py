import tensorflow as tf
import numpy as np
from .architecture import embedding, transformer, tagging

class Model(tf.keras.Model):
    def __init__(self, d_model, num_transformers, num_features, vocab_size,
                    layerdrop, num_heads, dff, dropout_rate, temperature, top_p,
                    num_samples, **kwargs):
        super().__init__(name="NERf")
        self.d_model = d_model
        self.num_transformers = num_transformers
        self.num_features = num_features
        self.vocabulary_size = vocab_size
        self.num_heads = num_heads
        self.dff = dff
        self.layerdrop = layerdrop
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.top_p = top_p
        self.num_samples = num_samples
        self.warm = False
        return
    # build function
    def build(self, input_shape):
        self.embedder = embedding.Layer(d_model = self.d_model,
            N = self.vocabulary_size, name = "NERfembed")
        self.transformers = [transformer.Layer(d_model = self.d_model,
            num_heads = self.num_heads, dff = self.dff, dropout_rate=self.dropout_rate, autoregressive=False, name =f"NERformer{i}")
            for i in range(self.num_transformers)]
        self.tagger = tagging.Layer(self.d_model, self.num_features, self.temperature, name="NER_tagging")
        return
    # embedding forward pass
    # @tf.function(reduce_retracing=True)
    def _embedPass(self, batch, mask=None):
        """
        Forward pass for embedding batch...
        """
        # runtime tensor shape
        batch_shape = tf.shape(batch)
        # flatten batch
        flat_batch = tf.reshape(batch, [-1])
        # embedding tokenized batch
        embeddings = self.embedder(flat_batch)
        # return embedding batch in the shape it was recieved in (with added dimension for logits)
        embed_dim = tf.shape(embeddings)[-1]  # gives int tensor
        # Now construct the shape properly
        target_shape = tf.concat([batch_shape, [embed_dim]], axis=0)
        # expand mask
        expanded_mask = tf.expand_dims(mask, axis=-1)
        # return reshaped tensor
        return tf.reshape(embeddings, target_shape) * expanded_mask

    # @tf.function(reduce_retracing=True)
    def _transformPass(self, embed_batch, training=False, mask=None):
        """
        Forward pass through transformers
        """
        # set forward pass batch
        fpass_batch = embed_batch
        # set increment to zero
        i = 0
        # loop through transformers
        for tfmr in self.transformers:
            # require at least one forward pass
            if self.warm == True or tf.random.uniform(shape = ()) < self.layerdrop:
                continue
            fpass_batch = tfmr(fpass_batch, training=training, mask=mask)
        # return forward pass batch after processed through transformers
        return fpass_batch

    # @tf.function(reduce_retracing=True)
    def tag(self, tokens, training=False, mask=None):
        # forward pass on embeddings
        embeddings = self._embedPass(tokens, mask=mask)
        # forward pass through transformers
        transforms = self._transformPass(embeddings, training=training, mask=mask)
        # define while loop break condition
        tags = self.tagger(transforms, top_p=self.top_p, num_samples=self.num_samples)
        # return transposed looped output
        return tags

    # @tf.function(reduce_retracing=True)
    def call(self, tokens, mask, training=False):
        inference_batch = self.tag(tokens, training=training, mask=mask)
        self.warm = True
        return inference_batch

    @property
    def Parameters(self):
        params = self.embedder.Parameters
        for layer in self.transformers:
            params += layer.Parameters
        params += self.tagger.Parameters
        return params
    # model size (important for training)
    @property
    def Size(self):
        """
        Calculate model size from parameters
        """
        # get parameters
        parameters = self.Parameters
        # set size = 0
        s = 0
        # perform size calculation on each parameter and sum them
        for p in parameters:
            n_p = 1
            # for dimension in the parameter shape, multiply values together
            for d in p.shape:
                n_p *= d
            # add layer size to overall size
            s += n_p
        # return size
        return s
    #parameters getter for model training
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_transformers": self.num_transformers,
            "num_features": self.num_features,
            "num_groups": self.num_groups,
            "vocabulary_size": self.vocabulary_size,
            "layerdrop": self.layerdrop,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "temperature": self.temperature
        })
        return config

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls,config):
        return cls(**config)
