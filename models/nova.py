import numpy as np
import tensorflow as tf
from . import embedding, transformer, final
from static import constants, _math

class Model:
    def __init__(self):
        self.embed = embedding.Layer(constants.d_model,
            name = "nova_embedding_layer")
        tfmrs = {}
        for i in range (1, constants.nova_tfmr + 1):
            tfmrs = {
                **tfmrs,
                **{i: transformer.Layer(constants.d_model,
                    constants.num_heads , constants.dff)}
                }
        self.tfmrs = tfmrs
        self.final = final.Layer(len(constants.vocabulary), constants.d_model)
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

    def fPass(self, in_batch):
        #embed token batch
        embed_batch = self.embedPass(in_batch)

        # pass through transformer layers
        tfmr_batch = self.transformPass(embed_batch)

        # pass through last layer for probabilities and refiting
        out_batch = self.final(tfmr_batch)

        # evaluate output batch
        logits = tf.argmax(out_batch[:,-1,:], axis = -1)

        # use logits to obtain next pass, or stop running
        out_batch = [[constants.vocabulary[i]] for i in logits]

        return out_batch

    def genPass(self, in_batch):
        return self.fPass(in_batch)


    def backpropagate(self, in_batch):
        # running forward pass
        fpass_batch = self.fpass(in_batch)

        return fpass_batch
