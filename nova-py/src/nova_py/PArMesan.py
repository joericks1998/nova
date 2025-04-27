import tensorflow as tf
import numpy as np
from .architecture import embedding, transformer, final

class Model(tf.keras.Model):
    # initializer
    def __init__(self, d_model=None, num_transformers=None, input_size = None, output_size=None,
                    layerdrop=None, num_heads=None, dff=None, dropout_rate=None,
                    temperature=None, top_p=None, num_samples=None, name=None):
        super().__init__(name=name)
        '''
        Initializes the model by loading hyperparameters, vocabulary, and the encoder.
        '''
        # inherit keras native methods

        # define inputs as attributes for serialization
        self.d_model = d_model
        self.num_transformers = num_transformers
        self.input_size = input_size
        self.output_size = output_size
        self.layerdrop = layerdrop
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.top_p = top_p
        self.num_samples = num_samples
        self.embedder = embedding.Layer(d_model = self.d_model,
            N = self.input_size, name = "PArMe")
        self.transformers = [transformer.Layer(d_model = self.d_model,
            num_heads = self.num_heads, dff = self.dff, dropout_rate=dropout_rate, autoregressive=True, name =f"PArMf{i}")
            for i in range(self.num_transformers)]
        self.final = final.Layer(self.d_model, self.output_size, self.temperature)

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
                fpass_batch = tfmr(fpass_batch, mask=mask,autoregres=True)
            # else layerdropping is in play (for performance optimization)
            elif np.random.random() < self.layerdrop:
                fpass_batch = tfmr(fpass_batch)
            # increment
            i+=1
        # return forward pass batch after processed through transformers
        return fpass_batch

    @tf.function(reduce_retracing=True)
    def _forwardPass(self, in_batch, mask=None, training = False):
        """
        pass batch through all layers, if training return probabilities for loss
        """
        #embed token batch
        embd_logits = self._embedPass(in_batch, mask=mask)
        # pass through transformer layers
        tfmr_logits = self._transformPass(embd_logits, mask=mask)
        # pass through last layer for probabilities and refiting
        o_tensor = self.final(tfmr_logits, top_p = self.top_p, num_samples=self.num_samples)
        return o_tensor

    #generate model outputs
    # @tf.function(reduce_retracing=True)
    def generate(self, batch, mask=None, token_limit = None):
        """
        main method for output generation
        """
        batch_size = tf.shape(batch)[0]
        seq_len = tf.shape(batch)[1]
        # define output array
        output_array = tf.TensorArray(dtype=tf.int32, size=token_limit)
        step = tf.constant(0)
        # create mask to stop generation
        # define while loop condition
        def cond(step, tokens, output):
            return step < token_limit
        #define body
        def body(step, tokens, output):
            next_token = self._forwardPass(tokens, mask=mask)
            mask = tf.concat([mask, tf.ones((tf.shape(mask)[0], 1), dtype=mask.dtype)], axis=1)
            output = output.write(step, next_token)
            tokens = tf.concat([tokens, tf.expand_dims(next_token, axis=1)], axis=1)
            return step + 1, tokens, output
        # Get the initial shape of each loop var
        shape_invariants = [
            tf.TensorShape([]),                    # step: scalar
            tf.TensorShape([None, None]),          # batch: shape (batch_size, seq_len) grows in axis=1
            tf.TensorShape(None)                   # output_array: TensorArray is always flexible
        ]
        # fancy loop
        step, out_tokens, output_array = tf.while_loop(
            cond,
            body,
            loop_vars = [step, batch, output_array],
            shape_invariants = shape_invariants
        )
        # consume output
        final_output = tf.transpose(output_array.stack(), perm=[1,0])
        return final_output

    def call(self, batch, mask=None, token_limit=250):
        return self.generate(batch, mask=mask, token_limit=token_limit)

    #get config for serialization
    def get_config(self):
        """
        return initializer inputs for serialization
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_transformers": self.num_transformers,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "layerdrop": self.layerdrop,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "temperature": self.temperature
        })
        return

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls, config):
        """
        return initialized class for deserialization
        """
        return cls(**config)

    #parameters getter for model training
    @property
    def Parameters(self):
        """
        get all model parameters for training
        """
        # initially set parameters list to the embedder parameters
        parameters = self.embedder.Parameters
        # add transformer parameters to parameter list
        for tfmr in self.transformers:
            parameters += tfmr.Parameters
        # finally add final layer parameters
        parameters += self.final.Parameters
        # return parameters
        return parameters

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
