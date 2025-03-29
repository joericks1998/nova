import tensorflow as tf
import numpy as np
from .architecture import embedding, transformer, tagging

class Model:
    def __init__(self, d_model=None, num_transformers=None, num_features=None,
                    num_groups=None, vocabulary_size=None, layerdrop=None,
                    num_heads=None, dF=None, dropout_rate=None, temperature=None):
        self.d_model = d_model
        self.num_transformers = num_transformers
        self.num_features = num_features
        self.num_groups = num_groups
        self.vocabulary_size = vocabulary_size
        self.num_heads = num_heads
        self.dff = d_model * dF
        self.layerdrop = layerdrop
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.embedder = embedding.Layer(d_model = self.d_model,
            N = self.vocabulary_size, name = "NERfembed")
        self.transformers = [transformer.Layer(d_model = self.d_model,
            num_heads = self.num_heads, dff = self.dff, dropout_rate=dropout_rate)
            for _ in range(self.num_transformers)]
        self.tagger = tagging.Layer(num_features=self.num_features, num_groups=self.num_groups,
            temperature=self.temperature)
        return

    def _embedPass(self, batch):
        """
        Forward pass for embedding batch...
        """
        # flatten batch
        flat_batch = tf.reshape(batch, [-1])
        # embedding tokenized batch
        embeddings = tf.stack([self.embedder(t) for t in flat_batch.numpy()])
        # return embedding batch in the shape it was recieved in (with added dimension for logits)
        return tf.reshape(embeddings, shape = batch.shape+[embeddings.shape[1]])

    # @tf.function(reduce_retracing=True)
    def _transformPass(self, embed_batch):
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
                fpass_batch = tfmr(fpass_batch, autoregres=False)
            # else layerdropping is in play (for performance optimization)
            elif np.random.random() < self.layerdrop:
                fpass_batch = tfmr(fpass_batch)
            # increment
            i+=1
        # return forward pass batch after processed through transformers
        return fpass_batch

    def tag(self, batch):
        inference_batch = None
        slice_batch = []
        token_batch = batch[0]
        spans_batch = batch[1]
        for sequence, spans in zip(token_batch, spans_batch):
            embeddings = self._embedPass(sequence[tf.newaxis])
            transforms = self._transformPass(embeddings)
            i = 0
            inferred = []
            t_slices = []
            for span in spans:
                slice = transforms[:,i:span+i, :]
                inferred.append(self.tagger(slice).numpy())
                t_slices.append(list(sequence[i:span+i].numpy()))
                i += span.numpy()
            if inference_batch is None:
                inference_batch = tf.constant(inferred)
            else:
                inference_batch = tf.stack([inference_batch, inferred])
            slice_batch.append(t_slices)
        return inference_batch, slice_batch

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
