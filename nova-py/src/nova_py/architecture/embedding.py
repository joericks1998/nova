import tensorflow as tf

class Layer(tf.Module):
    # Initialize the EmbeddingLayer with the given embedding dimension and optional name
    def __init__(self, d_model, name=None):
        super(Layer, self).__init__(name=name)
        self.d_model = d_model # Store the dimension of the embeddings for serialization
        self.embeddings = tf.zeros(shape = (1, self.d_model))  # Initialize embeddings to None
        self.tokens = {"<pad>": 0}  # Dictionary to map words to their indices
        self.built = True
        #placeholder for tf.Variable
        self.new_embedding = None

    # Method to retrieve or create the embedding for a given word
    @tf.function
    def __call__(self, word):
        # Define initializer
        initializer = tf.keras.initializers.GlorotUniform()
        if word not in self.tokens.keys():
            # If the word is new and not yet in the dictionary
            self.tokens[word] = self.embeddings.shape[0]  # Assign the next index to the new word
            # Create a new embedding and concatenate it to the existing embeddings
            if self.new_embedding is None:
                new_embedding = tf.Variable(initializer(shape = (1, self.d_model)))
            new_embeddings = tf.concat([self.embeddings, new_embedding], axis=0)
            self.embeddings = new_embeddings  # Update embeddings with the new concatenated tensor
        # Retrieve the embedding for the given word using its index
        return tf.nn.embedding_lookup(self.embeddings, self.tokens[word])

    def __add__(self, other_layer):
        # Method to add two EmbeddingLayer instances
        addition_layer = EmbeddingLayer(self.embedding_dim, name=self.name)  # Create a new EmbeddingLayer for the result

        if not isinstance(other_layer, EmbeddingLayer):
            # Ensure that the other_layer is also an EmbeddingLayer
            msg = f"Other embedding must be type: EmbeddingLayer, not {type(other_layer)}."
            raise TypeError(msg)  # Raise an error if the type does not match

        if other_layer.embeddings is None:
            # Ensure that the other_layer has initialized embeddings
            msg = "Embeddings in layer are missing"
            raise ValueError(msg)  # Raise an error if embeddings are missing

        other_h = other_layer.tokens  # Get the word-to-index mapping from the other layer
        # Update the indices in other_layer's dictionary to match the new combined layer
        updated_o_h = {k: v + self.embeddings.shape[0] for k, v in zip(other_h.keys(), other_h.values())}
        # Merge the word-to-index dictionaries from both layers
        addition_layer.tokens = {**self.tokens, **updated_o_h}

        # Concatenate embeddings from both layers along the first axis
        new_layer = tf.concat([self.embeddings, other_layer.embeddings], axis=0)
        addition_layer.embeddings = tf.Variable(new_layer)  # Update the new layer with the combined embeddings

        return addition_layer  # Return the new EmbeddingLayer with combined embeddings

    #get config for serialization
    def get_config(self):
        return master_config(Layer.__init__)

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    #parameters getter for model training
    @property
    def Parameters(self):
        return [self.embeddings]
