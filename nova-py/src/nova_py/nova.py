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
        self._hp_path = parent/_hp_path
        self._vocab_path = parent/_vocab_path
        self._encoder_path = parent/_encoder_path
        with open(self._hp_path, 'r') as f:
            self.hp = yaml.safe_load(f)
        with open(self._vocab_path, 'r') as f:
            self.vocabulary = [line.strip() for line in f]
        self.dims = self.hp['nova']['dimensions']
        self.run_specs = self.hp['nova']['run_specs']
        self.embedder = embedding.Layer(self.dims['d_model'], name = "nova_embedding_layer")
        #initialize transformers
        self.tfmrs = {i+1: transformer.Layer(self.dims['d_model'], self.dims['num_heads'],
                                            self.dims['dff'], self.run_specs['dropout_rate']) for i in range(0,self.dims['num_transformers'])}
        self.final = final.Layer(self.vocabulary, self.dims['d_model'], self.run_specs['temperature'])
        self.layerdrop = self.run_specs['layerdrop']
        self.top_p = self.run_specs['top_p']
        self.MINT = MINT.load()
        self.built = True
        # return
    def _embedPass(self, in_batch):
        flat_batch = tf.reshape(in_batch, [-1])
        # embedding tokenized batch
        embeddings = tf.stack([self.embedder(t.decode('utf-8')) for t in flat_batch.numpy()])
        return tf.reshape(embeddings, shape = in_batch.shape+[embeddings.shape[1]])

    # @tf.function(reduce_retracing=True)
    def _transformPass(self, embed_batch):
        fpass_batch = embed_batch
        i = 0
        for tfmr in self.tfmrs.values():
            if i == 0:
                fpass_batch = tfmr(fpass_batch)
            elif np.random.random() < self.layerdrop:
                fpass_batch = tfmr(fpass_batch)
            i+=1
        return fpass_batch

    def _forwardPass(self, in_batch, training = False):
        #embed token batch
        embd_logits = self._embedPass(in_batch)
        # pass through transformer layers
        tfmr_logits = self._transformPass(embd_logits)
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
                    if g_tensor.numpy()[0,0].decode('utf-8') == '<stop>':
                        stopped = True
                else:
                    g_tensor = tf.constant([['<pad>']])
                    g_seq = tf.concat([g_seq, g_tensor], axis = 1)
                if pretty_print:
                    print(g_tensor.numpy()[0,0].decode('utf-8'), end="")
            if o_batch is None:
                o_batch = g_seq
            else:
                print(g_seq.shape)
                o_batch = tf.stack([o_batch, g_seq])
        if len(o_batch.shape) < 3:
            return o_batch[tf.newaxis, :, :]
        return o_batch

    def teacherForce(self, batch, ground_truths):
        token_batch = TACO.inBatch(batch)
        encoded_batch = self.MINT(token_batch, translate = True)
        g_batch = encoded_batch
        o_batch = None
        for g_seq, teach_seq in zip(g_batch, ground_truths):
            for tkn in teach_seq:
                g_tensor = self._forwardPass(g_seq, training = True)
                g_seq = tf.concat([g_seq, tkn], axis = 1)
            if o_batch is None:
                o_batch = g_seq
            else:
                print(g_seq.shape)
                o_batch = tf.stack([o_batch, g_seq])
        if len(o_batch.shape) < 3:
            return o_batch[tf.newaxis, :, :]
        return o_batch
    #get config for serialization
    def get_config(self):
        return model_io.master_config(Model.__init__)


    #custom config method (also for serialization)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    #parameters getter for model training
    @property
    def Parameters(self):
        parameters = self.embed.Parameters
        for tfmr in self.tfmrs.values():
            parameters += tfmr.Parameters
        parameters += self.final.Parameters
        return parameters
    # train model
    def train(self, batch, ground_truths, method = 'teacher_force'):
        if method = 'teacher_force':
