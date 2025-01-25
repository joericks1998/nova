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
            self.vocabulary = f.readlines()
        self.dims = self.hp['nova']['dimensions']
        self.run_specs = self.hp['nova']['run_specs']
        self.embedder = embedding.Layer(self.dims['d_model'], name = "nova_embedding_layer")
        #initialize transformers
        self.tfmrs = {i+1: transformer.Layer(self.dims['d_model'], self.dims['num_heads'],
                                            self.dims['dff'], self.run_specs['dropout_rate']) for i in range(0,self.dims['num_transformers'])}
        self.final = final.Layer(self.vocabulary, self.dims['d_model'], self.run_specs['temperature'])
        self.top_p = self.run_specs['top_p']
        self.MINT = MINT.load()
        # return

    def _embedPass(self, in_batch):
        # embedding tokenized batch
        big_stack = []
        for seq in in_batch:
            small_stack = tf.stack([self.embedder(tkn) for tkn in seq])
            big_stack.append(small_stack)
        # stacking all batches into the embedding batch
        return tf.stack(big_stack)

    def _transformPass(self, embed_batch):
        fpass_batch = embed_batch
        for tfmr in self.tfmrs.values():
            fpass_batch = tfmr(fpass_batch)
        return fpass_batch

    def _forwardPass(self, in_batch):
        #embed token batch
        embd_logits = self._embedPass(in_batch)
        # pass through transformer layers
        tfmr_logits = self._transformPass(embd_logits)
        # pass through last layer for probabilities and refiting
        out = self.final(tfmr_logits, top_p = self.top_p)
        return out
    #generate model outputs
    def generate_regres(self, batch, token_limit = 250):
        token_batch = TACO.inBatch(batch)
        encoded_batch = self.MINT(token_batch, translate = False)
        o_batch = None
        return self._forwardPass(encoded_batch)
        # for sequence in encoded_batch:
        #     o_sequence = []
        #     for i in range(token_limit):
        #         o_token = self._forwardPass(sequence)
        #         o_sequence.append(o_token)
        #     o_tensor = tf.constant(o_sequence)
        #     if o_batch:
        #         o_batch = tf.stack([o_batch, o_tensor])
        #     else:
        #         o_batch = o_tensor
        # return o_batch
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
