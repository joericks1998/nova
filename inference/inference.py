import tensorflow as tf
from training import training
from text import data_io

class Queue:
    def __init__(self):
        self.map = {}
    @property
    def Pair(self):
        return self.map.items()
    @Pair.setter
    def Pair(self, variable, value):
        self.map = {**self.map, **{variable: value}}
        return

def Generator(text, model, tokenizer):
    tokens = tokenizer.word_split(text)
    logits = model.fpass([tokens])
    vocab = data_io.getVocab()
    return logits


def Trainer():
    pass
