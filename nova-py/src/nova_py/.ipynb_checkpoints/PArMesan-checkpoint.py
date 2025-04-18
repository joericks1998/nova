import tensorflow as tf
import numpy as np
from .architecture import embedding, transformer, final

class Model(tf.keras.Model):
    # initializer
    def __init__(self, d_model=None, num_transformers=None, input_size = None, output_size=None,
                    layerdrop=None, num_heads=None, dff=None, dropout_rate=None,
                    temperature=None, name=None):
        '''
        Initializes the model by loading hyperparameters, vocabulary, and the encoder.
        '''
        # inherit keras native methods
        super(self, Model).__init__(name=name)
        # define inputs as attributes for serialization
        self.d_model = d_model
        self.num_transformers = num_transformers
        self.output_size = output_size
        self.layerdrop = layerdrop
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.temperature = None
        self.embedder = embedding.Layer(d_model = self.d_model,
            N = self.input_size, name = "NERfembed")
        self.transformers = [transformer.Layer(d_model = self.d_model,
            num_heads = self.num_heads, dff = self.dff, dropout_rate=dropout_rate, name =f"NERformer {i}")
            for i in range(self.num_transformers)]
        self.final = final.Layer(self.d_model, self.output_size, self.temperature)

    def _embedPass(self, batch):
        """
        Forward pass for embedding batch...
        """
        # flatten batch
        flat_batch = tf.reshape(batch, [-1])
        # embedding tokenized batch
        embeddings = tf.map_fn(self.embedder, tf.cast(flat_batch, dtype=tf.float32))
        # return embedding batch in the shape it was recieved in (with added dimension for logits)
        return tf.reshape(embeddings, shape = batch.shape+[embeddings.shape[1]])

    @tf.function(reduce_retracing=True)
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
                fpass_batch = tfmr(fpass_batch, autoregres=True)
            # else layerdropping is in play (for performance optimization)
            elif np.random.random() < self.layerdrop:
                fpass_batch = tfmr(fpass_batch)
            # increment
            i+=1
    # return forward pass batch after processed through transformers
    return fpass_batch

    def _forwardPass(self, in_batch, training = False):
        """
        pass batch through all layers, if training return probabilities for loss
        """
        #embed token batch
        embd_logits = self._embedPass(in_batch)
        # pass through transformer layers
        tfmr_logits = self._transformPass(embd_logits, training = training)
        # pass through last layer for probabilities and refiting
        o_tensor = self.final(tfmr_logits, top_p = self.top_p, training = training)
        return o_tensor

    #generate model outputs
    def generate(self, batch, token_limit = 250):
        """
        main method for output generation
        """
        batch_size = batch.shape[0]
        seq_len = batch.shape[1]
        # define output array
        output_array = tf.TensorArray(dtype=tf.int32, size=token_limit)
        step = tf.constant(0)
        # define while loop condition
        def cond(step, tokens, output):
            return step < token_limit
        #define body
        def body(step, tokens, output):
            next_token = self._forwardPass(token, training = False)
            output_array = output.write(step, next_token)
            tokens = tf.concat([tokens, next_token[tf.newaxis]], axis = 1)
            return step + 1, tokens, output
        # fancy loop
        step, out_tokens, output_array = tf.while_loop(
            cond, body, loop_vars = [step, batch, output_array]
        )
        return output_array
    # if there is only one input sequence, add an axis
    if len(o_batch.shape) < 3:
        return o_batch[tf.newaxis, :, :]
    # return output batch
    return o_batch

    def __call__(self, batch):
        return self.generate(batch)

    #get config for serialization
    def get_config(self):
        """
        return initializer inputs for serialization
        """
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
        for tfmr in self.tfmrs.values():
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

    # get one hot encoded ground truths
    def getOneHotTruths(self, ground_truths, sequence_length):
        """
        get one hot tensor with ground truths for loss calculation
        """
        # set output list for one hot tensors
        o_list = []
        # loop through each ground truth sequence
        for g_seq in ground_truths:
            # set output tensor
            o_tensor = None
            for i in range(0, sequence_length):
                # hacky approach to setting variable and value one hot tensors after an Id has been set (needs to be corrected)
                try:
                    # if deep enough into the token sequence
                    if i > 2:
                        # if token is "id" set one hot to #VARIABLE index
                        if g_seq[i-2] == "id":
                            one_hot = tf.one_hot(self.vocabulary.index('#VARIABLE'), depth = len(self.vocabulary))
                        # if token is "value" set one hot to #VALUE index
                        elif g_seq[i-2] == "value":
                            one_hot = tf.one_hot(self.vocabulary.index('#VALUE'), depth = len(self.vocabulary))
                    # if none of the above cases are satisfied, search for the appropriate index in the vocabulary and generate one hot tensor
                    else:
                        one_hot = tf.one_hot(self.vocabulary.index(g_seq[i]), depth = len(self.vocabulary))
                # there is no index, set everything in the one hot tensor to zeros
                except tf.errors.InvalidArgumentError:
                    one_hot = tf.zeros(shape = (len(self.vocabulary)))
                # if this is the first one hot tensor, set with one_hot and new dimension
                if o_tensor is None:
                    o_tensor = one_hot[None, :]
                # else concat one hot vector to existing tensor
                else:
                    o_tensor = tf.concat([o_tensor, one_hot[None, :]], axis=0)
            # appended output tensor to list
            o_list.append(o_tensor)
        # return list of one hot tensor based on ground truth batches
        return o_list
