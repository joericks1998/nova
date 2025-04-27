import tensorflow as tf
import numpy as np
from .architecture import embedding, transformer, tagging

class Model(tf.keras.Model):
    def __init__(self, d_model=None, num_transformers=None, num_features=None,
                    vocabulary_size=None, layerdrop=None, num_heads=None,
                    dff=None, dropout_rate=None, temperature=None, top_p=None,
                    num_samples=None):
        super().__init__(name="NERf")
        self.d_model = d_model
        self.num_transformers = num_transformers
        self.num_features = num_features
        self.vocabulary_size = vocabulary_size
        self.num_heads = num_heads
        self.dff = dff
        self.layerdrop = layerdrop
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.top_p = top_p
        self.num_samples = num_samples
        self.embedder = embedding.Layer(d_model = self.d_model,
            N = self.vocabulary_size, name = "NERfembed")
        self.embedder.build(input_shape=tf.TensorShape([None]))
        self.transformers = [transformer.Layer(d_model = self.d_model,
            num_heads = self.num_heads, dff = self.dff, dropout_rate=dropout_rate, autoregressive=False, name =f"NERformer{i}")
            for i in range(self.num_transformers)]
        self.tagger = tagging.Layer(num_features=self.num_features,
            temperature=self.temperature, name="NER_tagging")
        self._spans = None
        return

    @tf.function(reduce_retracing=True)
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

    @tf.function(reduce_retracing=True)
    def _transformPass(self, embed_batch, mask=None):
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
            if i == 0:
                fpass_batch = tfmr(fpass_batch, mask=mask)
            # else layerdropping is in play (for performance optimization)
            elif np.random.random() < self.layerdrop:
                fpass_batch = tfmr(fpass_batch)
            # increment
            i+=1
        # return forward pass batch after processed through transformers
        return fpass_batch

    @tf.function(reduce_retracing=True)
    def tag(self, in_batch, mask=None):
        # define sequence
        sequence = in_batch[0]
        # define spans
        spans = in_batch[1]
        # flatten spans (for iteration)
        flat_spans = tf.reshape(spans, [-1])
        span_shape = tf.shape(spans)
        # forward pass on embeddings
        embeddings = self._embedPass(sequence, mask=mask)
        # forward pass through transformers
        transforms = self._transformPass(embeddings, mask=mask)
        # define array to store outputs
        inferred = tf.TensorArray(dtype=tf.int32, size=span_shape[1])
        step = tf.constant(0, dtype = tf.int32)
        # define while loop break condition
        def cond(step, output):
            return step < span_shape[1]
        # define body of loop
        def body(step, output):
            # get next span (on step)
            span = tf.cast(flat_spans[step], tf.int32)
            # slice transformer outputs by span
            slice = transforms[:, step:span+step, :]
            # tag slice
            tags = self.tagger(slice, top_p=self.top_p, num_samples=self.num_samples)
            # write output
            output = output.write(step, tags)
            return step + 1, output
        # Get the initial shape of each loop var
        shape_invariants = [
            tf.TensorShape([]),       # step: scalar
            tf.TensorShape(None)      # output_array: TensorArray is always flexible
        ]
        # fancy loop
        step, output_array = tf.while_loop(
            cond,
            body,
            loop_vars = [step, inferred],
            shape_invariants = shape_invariants
        )
        # return transposed looped output
        return tf.transpose(output_array.stack(), perm=[1, 0])

    @tf.function(reduce_retracing=True)
    def call(self, batch, mask=None):
        inference_batch = self.tag(batch, mask=mask)
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
