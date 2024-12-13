import numpy as np
import tensorflow as tf
from neuralnetworks import ffnn, embedding, attention, masking, transformer
from static import constants


class Model:
    def __init__(self, vocabulary):
        self.emd_lyr = embedding.Layer(constants.d_model,
            name = "nova_embedding_layer")
        tfmrs = {}
        for i in range (1, constants.nova_tfmr + 1):
            tfmrs = {
                **tfmrs,
                **{i: transformer.Layer(constants.d_model,
                    constants.num_heads , constants.dff)}
                }
        self.tfmrs = tfmrs
        self.final = FinalLayer(len(vocabulary), constants.d_model)
        return
