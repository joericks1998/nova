import tensorflow as tf

class EmbeddingLayer(tf.module):
    def __init__(self, embedding_dim, name=None):
        super().__init__(name=name)
        self.initializer = tf.initializer.HeNormal()
        self.embedding = None
        self.hmap = {}

    def __call__(self, word):
        try:
            x = self.hmap[word]
        except:
            if self.embedding is None:
                self.hmap[word] = 0
                tf.Variable(self.initializer([1, embedding_dim]),
                                             name="embeddings",
                                             trainable = True)
        return tf.nn.embedding_lookup(self.embeddings, x)
