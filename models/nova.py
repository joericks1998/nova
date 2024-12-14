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
        self.final = final.Layer(len(self.vocabulary), constants.d_model)
        return

    def fpass(self, in_batch):
        # embedding tokenized batch
        big_stack = []
        for seq in in_batch:
            small_stack = tf.stack([self.embed(tkn) for tkn in seq])
            big_stack.append(small_stack)

        # stacking all batches into the embedding batch
        embed_batch = tf.stack(big_stack)

        # pass through transformer layers
        fpass_batch = embed_batch
        for tfmr in self.tfmrs.values():
            fpass_batch = tfmr(fpass_batch)

        # pass through last layer for probabilities and refiting
        out_batch = self.final(fpass_batch)

        # evaluate output batch
        logits = tf.argmax(out_batch[:,-1,:], axis = -1)

        return logits


    def backpropagate(self, in_batch):
        # running forward pass
        fpass_batch = self.fpass(in_batch)

        return fpass_batch
