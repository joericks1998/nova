import tensorflow as tf

def posEncoding(seq_len, embedding_dim):
    pass

class EmbeddingLayer(tf.Module):
    def __init__(self, embedding_dim, name=None):
        # Initialize the EmbeddingLayer with the given embedding dimension and optional name
        super().__init__(name=name)
        self.embedding_dim = embedding_dim  # Store the dimension of the embeddings
        self.embeddings = None  # Initialize embeddings to None
        self.h = {}  # Dictionary to map words to their indices

    def __call__(self, word):
        # Method to retrieve or create the embedding for a given word
        if not isinstance(word, str):
            return "embedding model is for strings only"  # Ensure the input is a string

        #Test again
        # Define initializer
        initializer = tf.keras.initializers.HeNormal()

        if self.embeddings is None:
            # If embeddings are not initialized, create the first embedding
            self.h[word] = 0  # Assign index 0 to the new word
            self.embeddings = tf.Variable(initializer(shape = (1, self.embedding_dim)))
        elif word not in self.h.keys():
            # If the word is new and not yet in the dictionary
            self.h[word] = self.embeddings.shape[0]  # Assign the next index to the new word
            # Create a new embedding and concatenate it to the existing embeddings
            new_embedding = tf.Variable(initializer(shape = (1, self.embedding_dim)))
            new_embeddings = tf.concat([self.embeddings, new_embedding], axis=0)
            self.embeddings = tf.Variable(new_embeddings)  # Update embeddings with the new concatenated tensor

        # Retrieve the embedding for the given word using its index
        return tf.nn.embedding_lookup(self.embeddings, self.h[word])

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

        other_h = other_layer.h  # Get the word-to-index mapping from the other layer
        # Update the indices in other_layer's dictionary to match the new combined layer
        updated_o_h = {k: v + self.embeddings.shape[0] for k, v in zip(other_h.keys(), other_h.values())}
        # Merge the word-to-index dictionaries from both layers
        addition_layer.h = {**self.h, **updated_o_h}

        # Concatenate embeddings from both layers along the first axis
        new_layer = tf.concat([self.embeddings, other_layer.embeddings], axis=0)
        addition_layer.embeddings = tf.Variable(new_layer)  # Update the new layer with the combined embeddings

        return addition_layer  # Return the new EmbeddingLayer with combined embeddings
