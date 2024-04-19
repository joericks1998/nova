from gensim.models import Word2Vec
import numpy as np

class Embed:
    def __init__(self, tokens):
        self.tokens = tokens
        self.model = None
        self.path = '/Users/joericks/Desktop/nova/model/core/nova_embedding.model'

    @property
    def Model(self):
        self.model = Word2Vec.load(self.path)
        return self.model

    def train(self):
        model = self.Model
        ignore_words = []
        model.build_vocab([self.tokens])
        model.train([self.tokens], total_examples=1, epochs = 1)
        model.save(self.path)
        return

    @property
    def Vector(self):
        try:
            result = 0
            arr = [self.Model.wv[t] for t in self.tokens]
            for v in arr:
                result = np.add(result, v)
            return result/len(arr)
        except KeyError:
            print("Training on new words...")
            self.train()
            print("Done")
            return np.sum([self.Model.wv[t] for t in self.tokens])
