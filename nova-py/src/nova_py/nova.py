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
    @tf.function(reduce_retracing=True)
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

    def _forwardPass(self, in_batch):
        #embed token batch
        embd_logits = self._embedPass(in_batch)
        # pass through transformer layers
        tfmr_logits = self._transformPass(embd_logits)
        # # pass through last layer for probabilities and refiting
        idx = self.final(tfmr_logits, top_p = self.top_p)
        return tf.constant([[self.vocabulary[idx.numpy()]]])
    #generate model outputs
    def generate_regres(self, batch, token_limit = 250, clean=True, pretty_print = False):
        token_batch = TACO.inBatch(batch)
        encoded_batch = self.MINT(token_batch, translate = True)
        o_batch = encoded_batch
        for i in range(token_limit):
            o_tensor = self._forwardPass(o_batch)
            o_batch = tf.concat([o_batch, o_tensor], axis = 1)
            if pretty_print:
                print(o_tensor.numpy()[0,0].decode('utf-8'), end="")
        if clean:
            cleaned = list(o_batch.numpy())
            return cleaned
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
