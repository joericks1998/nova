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

def vocabMapper(logit, vocab = data_io.getVocab()):
    return vocab[logit]


def Generator(text, model = None, tokenizer = None, max_t = 30):
    tokens = tokenizer.word_split(text)
    batch = [tokens]
    logits = model.fPass(batch)
    output = list(map(vocabMapper, logits))
    i = 0
    while i < max_t:
        print(f"Generated tokens {i} of {max_t}")
        batch[0] += output
        logits = model.fPass(batch)
        output = list(map(vocabMapper, logits))
        if "<stop>" in output:
            return batch
        i+=1
    return batch


def Trainer():
    pass
