import tensorflow as tf
from .architecture import embedding, transformer, tagging

class NERformer:
    def __init__(self, d_model=None, num_transformers=None, num_features=None, num_groups = None, vocabulary=None):
        self.d_model = d_model
        self.num_transformers = num_transformers
        self.num_features = num_features
        self.num_groups = num_groups
        self.vocabulary = vocabulary
        self.embedder = embedding.Layer(d_model = self.d_model, N = len(self.vocabulary), name = "NERformer Embedding Layer")
        self.transformers = [transformer.Layer() for _ in range(self.num_transformers)]
        self.tagger = tagging.Layer(num_features = self.num_features, num_groups = self.num_groups)
        return

    def tag(self, tokens):
        return

    @Property
    def Parameters(self):
        return
