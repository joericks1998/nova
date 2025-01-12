import numpy as np
import tensorflow as tf
from architecture import embedding, transformer, final
# from static import constants, _math
# from utils import model_io

class Nova(tf.keras.Model):
    def __init__(self, d_model = constants.d_model, num_heads=constants.num_heads,
                dff = constants.dff, vocab_len = constants.vocab_len,
                num_tfmrs = constants.nova_tfmr, encoder):
        super(Model, self).__init__()
        self.embed = embedding.Layer(d_model, name = "nova_embedding_layer")
        #initialize padding vector
        tfmrs = {}
        for i in range (1, num_tfmrs + 1):
            tfmrs = {
                **tfmrs,
                **{i: transformer.Layer(d_model, num_heads , dff)}
                }
        self.tfmrs = tfmrs
        self.final = final.Layer(vocab_len, d_model)
        return

    def embedPass(self, in_batch):
        # embedding tokenized batch
        big_stack = []
        for seq in in_batch:
            small_stack = tf.stack([self.embed(tkn) for tkn in seq])
            big_stack.append(small_stack)
        # stacking all batches into the embedding batch
        return tf.stack(big_stack)

    def transformPass(self, embed_batch):
        fpass_batch = embed_batch
        for tfmr in self.tfmrs.values():
            fpass_batch = tfmr(fpass_batch)
        return fpass_batch

    def fPass(self, in_batch, training = False):
        #embed token batch
        embd_logits = self.embedPass(in_batch)

        # pass through transformer layers
        tfmr_logits = self.transformPass(embd_logits)

        # pass through last layer for probabilities and refiting
        probabilities = self.final(tfmr_logits)

        return probabilities

    #generate model outputs
    def generate(self, in_batch, training = False):


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
