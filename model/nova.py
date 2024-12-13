import numpy as np
import tensorflow as tf
from neuralnetworks import ffnn, embedding, attention, masking, transformer

class FinalLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(TransformerFinalLayer, self).__init__()
        self.projection = tf.keras.layers.Dense(vocab_size)

    def __call__(self, inputs):
        logits = self.projection(inputs)  # Project to vocab_size
        probabilities = tf.nn.softmax(logits, axis=-1)  # Convert to probabilities
        return probabilities

class Nova:
    def __init__(self, vocabulary):
        self.emd_lyr = embedding.EmbeddingLayer(256, name = "nova embedding layer")
        tfmrs = {}
        for i in range (1,33):
            tfmrs = {**tfmrs, **{i: transformer.TransformerLayer(size, batch.shape[1], 4 , 4*size)}}
        self.tfmrs = tfmrs
        self.final = FinalLayer(len(vocabulary), 256)
        return
