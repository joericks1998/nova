import numpy as np
import tensorflow as tf
from .architecture import embedding, transformer, final
from . import TACO
from .transcribe import MINT
from pathlib import Path
import yaml

class Model(tf.keras.Model):

    def __init__(self, _hp_path = 'model/hyperparameters.yaml', _vocab_path = 'model/vocabulary.txt', _encoder_path = 'model/semantics'):
        super(Model, self).__init__()
        parent = Path(__file__).resolve().parent
        self.default_save_path = parent/"nova.keras"
        self._hp_path = parent/_hp_path
        self._vocab_path = parent/_vocab_path
        self._encoder_path = parent/_encoder_path
        with open(self._hp_path, 'r') as f:
            self.hp = yaml.safe_load(f)
        with open(self._vocab_path, 'r') as f:
            self.vocabulary = [line.strip() for line in f]
        self.dims = self.hp['nova']['dimensions']
        self.run_specs = self.hp['nova']['run_specs']
        self.training_specs = self.hp['nova']['training_specs']
        self.embedder = embedding.Layer(self.dims['d_model'], name = "nova_embedding_layer")
        #initialize transformers
        self.tfmrs = {i+1: transformer.Layer(self.dims['d_model'], self.dims['num_heads'],
                                            self.dims['dff'], self.run_specs['dropout_rate']) for i in range(0,self.dims['num_transformers'])}
        self.final = final.Layer(self.dims['d_model'], len(self.vocabulary), self.run_specs['temperature'])
        self.layerdrop = self.run_specs['layerdrop']
        self.top_p = self.run_specs['top_p']
        self.MINT = MINT.load()
        self.built = True
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.learning_rate = self.training_specs['learning_rate']
        self.num_epochs = self.training_specs['num_epochs']
        # return
    def _embedPass(self, in_batch):
        flat_batch = tf.reshape(in_batch, [-1])
        # embedding tokenized batch
        embeddings = tf.stack([self.embedder(t.decode('utf-8')) for t in flat_batch.numpy()])
        return tf.reshape(embeddings, shape = in_batch.shape+[embeddings.shape[1]])

    # @tf.function(reduce_retracing=True)
    def _transformPass(self, embed_batch, training=False):
        fpass_batch = embed_batch
        i = 0
        for tfmr in self.tfmrs.values():
            if i == 0:
                fpass_batch = tfmr(fpass_batch, training=training)
            elif np.random.random() < self.layerdrop:
                fpass_batch = tfmr(fpass_batch, training=training)
            i+=1
        return fpass_batch

    def _forwardPass(self, in_batch, training = False):
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
        token_batch = TACO.inBatch(batch)
        encoded_batch = self.MINT(token_batch, translate = True)
        g_batch = encoded_batch
        o_batch = None
        for g_seq in g_batch:
            stopped = False
            for i in range(token_limit):
                if not stopped:
                    g_tensor = self._forwardPass(g_seq)
                    g_seq = tf.concat([g_seq, g_tensor], axis = 1)
                    if g_tensor.numpy()[0,0].decode('utf-8') == '#END':
                        stopped = True
                    if pretty_print and not stopped:
                        print(g_tensor.numpy()[0,0].decode('utf-8'), end="")
                else:
                    g_tensor = tf.constant([['#PAD']])
                    g_seq = tf.concat([g_seq, g_tensor], axis = 1)
            if o_batch is None:
                o_batch = g_seq
            else:
                o_batch = tf.stack([o_batch, g_seq])
        if len(o_batch.shape) < 3:
            return o_batch[tf.newaxis, :, :]
        return o_batch

    #get config for serialization
    def get_config(self):
        return {
            "_hp_path": str(self._hp_path.relative_to(Path(__file__).resolve().parent)),
            "_vocab_path": str(self._vocab_path.relative_to(Path(__file__).resolve().parent)),
            "_encoder_path": str(self._encoder_path.relative_to(Path(__file__).resolve().parent)),
        }

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    #parameters getter for model training
    @property
    def Parameters(self):
        parameters = self.embedder.Parameters
        for tfmr in self.tfmrs.values():
            parameters += tfmr.Parameters
        parameters += self.final.Parameters
        return parameters

    # model size (important for training)
    @property
    def Size(self):
        parameters = self.Parameters
        s = 0
        for p in parameters:
            n_p = 1
            for d in p.shape:
                n_p *= d
            s += n_p
        return s

    # get one hot encoded ground truths
    def getOneHotTruths(self, ground_truths, sequence_length):
        o_list = []
        for g_seq in ground_truths:
            o_tensor = None
            for i in range(0, sequence_length):
                try:
                    if i > 2:
                        if g_seq[i-2] == "id":
                            one_hot = tf.one_hot(self.vocabulary.index('#VARIABLE'), depth = len(self.vocabulary))
                        elif g_seq[i-2] == "value":
                            one_hot = tf.one_hot(self.vocabulary.index('#VALUE'), depth = len(self.vocabulary))
                    else:
                        one_hot = tf.one_hot(self.vocabulary.index(g_seq[i]), depth = len(self.vocabulary))
                except tf.errors.InvalidArgumentError:
                    one_hot = tf.zeros(shape = (len(self.vocabulary)))
                if o_tensor is None:
                    o_tensor = one_hot[None, :]
                else:
                    o_tensor = tf.concat([o_tensor, one_hot[None, :]], axis=0)
            o_list.append(o_tensor)
        return o_list
    # model training function
    def train(self, batch, ground_truths, teacher_force = True, token_limit = 250):
        token_batch = TACO.inBatch(batch)
        ground_truth_tokens = TACO.inBatch(ground_truths)
        encoded_batch = self.MINT(token_batch, translate = True)
        one_hot_truths = self.getOneHotTruths(ground_truth_tokens, ground_truth_tokens.shape[1])
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        for epoch in range(self.num_epochs):
            with tf.GradientTape() as tape:
                for g_seq, teach_seq, one_hot in zip(encoded_batch, ground_truth_tokens, one_hot_truths):
                    loss_batch = None
                    for t in teach_seq:
                        g_tensor = self._forwardPass(g_seq, training=True)
                        if loss_batch is None:
                            loss_batch = g_tensor
                        else:
                            loss_batch = tf.concat([loss_batch, g_tensor], axis = 0)
                        if teacher_force:
                            g_seq = tf.concat([g_seq, t[tf.newaxis, tf.newaxis]], axis = 1)
                    loss = self.loss_fn(y_true=one_hot, y_pred=loss_batch)
                    gradients = tape.gradient(loss, self.Parameters)
                    optimizer.apply_gradients(zip(gradients, self.Parameters))
        return loss
    # os methods
    def save(self, save_path=None):
        """Saves the model to a default directory if no path is provided."""
        if save_path is None:
            save_path = self.default_save_path

        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Save model weights using TensorFlow checkpointing
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.write(os.path.join(save_path, "model.ckpt"))
        print(f"Model saved to {save_path}")

    def load(self, load_path=None):
        """Loads the model from a checkpoint."""
        if load_path is None:
            load_path = self.default_save_path

        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(os.path.join(load_path, "model.ckpt"))
        print(f"Model loaded from {load_path}")
