import numpy as np
import tensorflow as tf
from .architecture import embedding, transformer, final
from . import TACO
from .transcribe import MINT
from pathlib import Path
import yaml

def funDecorations(func):
    def wrapper(self, *args, **kwargs):
        return
    return


class Model(tf.keras.Model):
    # initializer
    def __init__(self, _hp_path = 'model/hyperparameters.yaml', _vocab_path = 'model/vocabulary.txt', _encoder_path = 'model/semantics'):
        '''
        Initializes the model by loading hyperparameters, vocabulary, and the encoder.
        '''
        # inherit keras native methods
        super(Model, self).__init__()
        # get parent path for model io
        parent = Path(__file__).resolve().parent
        # define default paths
        self.default_save_path = parent/"nova.keras"
        self._hp_path = parent/_hp_path
        self._vocab_path = parent/_vocab_path
        self._encoder_path = parent/_encoder_path
        # open and read hyperparameters from yaml
        with open(self._hp_path, 'r') as f:
            self.hp = yaml.safe_load(f)
        # open and read vocabulary from text file
        with open(self._vocab_path, 'r') as f:
            self.vocabulary = [line.strip() for line in f]
        # model dimensions
        self.dims = self.hp['nova']['dimensions']
        # run specifications (like dropout)
        self.run_specs = self.hp['nova']['run_specs']
        # training specifications
        self.training_specs = self.hp['nova']['training_specs']
        # create embedder
        self.embedder = embedding.Layer(self.dims['d_model'], name = "nova_embedding_layer")
        # initialize transformers
        self.tfmrs = {i+1: transformer.Layer(self.dims['d_model'], self.dims['num_heads'],
                                            self.dims['dff'], self.run_specs['dropout_rate']) for i in range(0,self.dims['num_transformers'])}
        # initialize final layer
        self.final = final.Layer(self.dims['d_model'], len(self.vocabulary), self.run_specs['temperature'])
        # initialize layer dropout
        self.layerdrop = self.run_specs['layerdrop']
        # set top p sampling parameter
        self.top_p = self.run_specs['top_p']
        # build encoder
        self.MINT = MINT.load()
        # define loss function
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        # define learning_rate
        self.learning_rate = self.training_specs['learning_rate']
        # define number of epochs for training
        self.num_epochs = self.training_specs['num_epochs']

    def _embedPass(self, in_batch):
        """
        Forward pass for embedding batch...
        """
        # flatten batch
        flat_batch = tf.reshape(in_batch, [-1])
        # embedding tokenized batch
        embeddings = tf.stack([self.embedder(t.decode('utf-8')) for t in flat_batch.numpy()])
        # return embedding batch in the shape it was recieved in (with added dimension for logits)
        return tf.reshape(embeddings, shape = in_batch.shape+[embeddings.shape[1]])

    # @tf.function(reduce_retracing=True)
    def _transformPass(self, embed_batch, training=False):
        """
        Forward pass through transformers
        """
        # set forward pass batch
        fpass_batch = embed_batch
        # set increment to zero
        i = 0
        # loop through transformers
        for tfmr in self.tfmrs.values():
            # require at least one forward pass
            if i == 0:
                fpass_batch = tfmr(fpass_batch, training=training)
            # else layerdropping is in play (for performance optimization)
            elif np.random.random() < self.layerdrop:
                fpass_batch = tfmr(fpass_batch, training=training)
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
        # # pass through last layer for probabilities and refiting
        o_tensor = self.final(tfmr_logits, top_p = self.top_p, training = training)
        if training:
            return o_tensor
        return tf.constant([[self.vocabulary[o_tensor.numpy()]]])

    #generate model outputs
    def generate(self, batch, token_limit = 250, clean=True, pretty_print = False):
        """
        main method for output generation
        """
        # tokenize in batch
        token_batch = TACO.inBatch(batch)
        # encode tokens
        encoded_batch = self.MINT(token_batch, translate = True)
        # set generation batch to encodings
        g_batch = encoded_batch
        # set out batch to none
        o_batch = None
        # loop through each sequence in the batch
        for g_seq in g_batch:
            # set stop variable
            stopped = False
            # generate until token limit or stop token is reached
            for i in range(token_limit):
                # generate token if not stopped
                if not stopped:
                    # pass sequence through model
                    g_tensor = self._forwardPass(g_seq)
                    # concat generated token to existing sequence
                    g_seq = tf.concat([g_seq, g_tensor], axis = 1)
                    # if the token is the stop token, set the stop variable to true
                    if g_tensor.numpy()[0,0].decode('utf-8') == '#END':
                        stopped = True
                    # if pretty printing is on, print generated token
                    if pretty_print and not stopped:
                        print(g_tensor.numpy()[0,0].decode('utf-8'), end="")
                # if stopped, generate padding tokens until token limit is reached
                else:
                    # set generated token to the padding token
                    g_tensor = tf.constant([['#PAD']])
                    # concate generated token to existing sequence
                    g_seq = tf.concat([g_seq, g_tensor], axis = 1)
            # if the outbatch hasn't been set, set to sequence
            if o_batch is None:
                o_batch = g_seq
            # else append latest output sequence to the out batch
            else:
                o_batch = tf.stack([o_batch, g_seq])
        # if there is only one input sequence, add an axis
        if len(o_batch.shape) < 3:
            return o_batch[tf.newaxis, :, :]
        # return output batch
        return o_batch

    #get config for serialization
    def get_config(self):
        """
        return initializer inputs for serialization
        """
        return {
            "_hp_path": str(self._hp_path.relative_to(Path(__file__).resolve().parent)),
            "_vocab_path": str(self._vocab_path.relative_to(Path(__file__).resolve().parent)),
            "_encoder_path": str(self._encoder_path.relative_to(Path(__file__).resolve().parent)),
        }

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

    # model training function
    def train(self, batch, ground_truths, teacher_force = True, token_limit = 250):
        """
        train the model!
        """
        # tokenize input batch
        token_batch = TACO.inBatch(batch)
        # tokenize ground truth tokens
        ground_truth_tokens = TACO.inBatch(ground_truths)
        # encoded input tokens
        encoded_batch = self.MINT(token_batch, translate = True)
        # generate one hot encoded vectors for ground truths
        one_hot_truths = self.getOneHotTruths(ground_truth_tokens, ground_truth_tokens.shape[1])
        # define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        # train over number of epochs
        for epoch in range(self.num_epochs):
            # forward pass with gradient tape for gradient calculations
            with tf.GradientTape() as tape:
                # loop through zipped batch with ground truths and one hot truths
                for g_seq, teach_seq, one_hot in zip(encoded_batch, ground_truth_tokens, one_hot_truths):
                    # define loss batch (raw output vectors)
                    loss_batch = None
                    # for token in teacher sequence...
                    for t in teach_seq:
                        # get output tensor from forward pass with training on
                        g_tensor = self._forwardPass(g_seq, training=True)
                        # if the loss batch is empty, define it as the tensor just outputted
                        if loss_batch is None:
                            loss_batch = g_tensor
                        # else concat to existing loss batch
                        else:
                            loss_batch = tf.concat([loss_batch, g_tensor], axis = 0)
                        # if teacher forcing is on, concat the teacher sequence token
                        if teacher_force:
                            g_seq = tf.concat([g_seq, t[tf.newaxis, tf.newaxis]], axis = 1)
                    # calculate loss between the loss batch and the one hot tensors
                    loss = self.loss_fn(y_true=one_hot, y_pred=loss_batch)
                    # generate gradients from gradient tape based on loss
                    gradients = tape.gradient(loss, self.Parameters)
                    # apply gradients with optimizer to model
                    optimizer.apply_gradients(zip(gradients, self.Parameters))
        # return calculated loss
        return loss
    
    # os methods
    def save(self, save_path=None):
        """
        Saves the model to a default directory if no path is provided.
        """
        if save_path is None:
            save_path = self.default_save_path

        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Save model weights using TensorFlow checkpointing
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.write(os.path.join(save_path, "model.ckpt"))
        print(f"Model saved to {save_path}")

    def load(self, load_path=None):
        """
        Loads the model from a checkpoint.
        """
        if load_path is None:
            load_path = self.default_save_path

        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(os.path.join(load_path, "model.ckpt"))
        print(f"Model loaded from {load_path}")
