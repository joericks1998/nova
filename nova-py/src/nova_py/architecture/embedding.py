import tensorflow as tf

class Layer(tf.Module):
    # Initialize the EmbeddingLayer with the given embedding dimension and optional name
    def __init__(self, d_model=None, N=None, name=None):
        if not d_model or not N:
            msg = '''
                Values for N and d_model must be passed.
            '''
            raise ValueError(msg)
        super(Layer, self).__init__(name=name)
        self.d_model = d_model # Store the dimension of the embeddings for serialization
        self.N = N
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.embeddings = tf.Variable(self.initializer(shape = (N, self.d_model)), trainable=True)  # Initialize embeddings to None
    # Method to retrieve or create the embedding for a given word
    def __call__(self, token):
        if isinstance(token, int) and token <= self.N:
            return tf.nn.embedding_lookup(self.embeddings, token)
        else:
            msg = '''
            Token must be of type "int".
            '''
            raise TypeError(msg)
        # Retrieve the embedding for the given word using its index

    #get config for serialization
    def get_config(self):
        return {
            "d_model": self.d_model,
            "N": self.N,
            "name": self.name
        }

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    #parameters getter for model training
    @property
    def Parameters(self):
        return [self.embeddings]
