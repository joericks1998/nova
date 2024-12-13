import numpy as np
import tensorflow as tf
from neuralnetworks import ffnn, embedding, attention, masking, transformer
from static import constants


class Model:
    def __init__(self, vocabulary):
        self.emd_lyr = embedding.Layer(constants.d_model, name = "nova_embedding_layer")
        self.tfmrs = {}
        self.final = FinalLayer(len(vocabulary), constants.d_model)
        return
