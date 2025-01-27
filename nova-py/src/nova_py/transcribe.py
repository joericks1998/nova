import re
import tensorflow as tf
import numpy as np
import os
import json
import yaml
from pathlib import Path

class MINT(tf.Module):
    """
    A class for encoding sequences based on predefined tagging and transition logic.

    Attributes:
        TransitionMatrix (tf.Variable): Transition matrix used for tagging sequences.
        TransitionStates (dict): Dictionary mapping sequence states to indices.
        tags (dict): Mapping of tags to their indices.
        predefinitions (dict): Predefined tags for specific tokens.
        n_limit (int): Maximum memory limit for sequence length.
        cooling_factor (float): Factor to smooth probabilities during training.
    """

    def __init__(self, _transition_matrix=None, _transition_states=None):
        """
        Initialize the MINT encoder.

        Args:
            _transition_matrix (tf.Variable, optional): A predefined transition matrix. Defaults to None.
            _transition_states (dict, optional): A predefined set of transition states. Defaults to None.
        """
        # Resolve the path of the current file's directory
        self.parent = Path(__file__).resolve().parent

        # Load tag mappings from JSON file
        with open(os.path.join(self.parent, "model/tags.json"), "r") as f:
            self.tags = json.load(f)

        # Load predefined tag mappings from JSON file
        with open(os.path.join(self.parent, "model/predefined_tags.json"), "r") as f:
            self.predefinitions = json.load(f)

        # Load translator from JSON file
        with open(os.path.join(self.parent, "model/translator.json"), "r") as f:
            self.translator = json.load(f)

        # Initialize the transition matrix; if not provided, use a uniform distribution
        if _transition_matrix is not None:
            self.TransitionMatrix = _transition_matrix
        else:
            self.TransitionMatrix = tf.Variable(
                [[1 / len(self.tags) for _ in range(len(self.tags))]], dtype=tf.float64
            )

        # Initialize transition states; defaults to an empty dictionary if not provided
        if _transition_states is not None:
            self.TransitionStates = _transition_states
        else:
            self.TransitionStates = {'': 0}

        # Load hyperparameters from YAML file
        with open(os.path.join(self.parent, "model/hyperparameters.yaml"), "r") as f:
            yams = yaml.safe_load(f)['MINT']
            self.n_limit = yams['n_limit']  # Set the maximum sequence memory limit
            self.cooling_factor = yams['cooling_factor']  # Set the cooling factor for probability smoothing

    def pretag(self, batch, stage = False):
        """
        Assign a predefined tag to a token if it exists in the predefinitions.

        Args:
            token (tf.Tensor): The token to tag.

        Returns:
            str: The corresponding tag or an empty string if no match is found.
        """
        # Iterate over predefined tags and check if the token matches any predefined values
        flat_batch = tf.reshape(batch, [-1])
        tags = []
        for token in flat_batch.numpy():
            tag = ''
            for k, v in self.predefinitions.items():
                if token.decode('utf-8') in v:
                    tag = k
            tags.append(tag)
        out_batch = tf.transpose(tf.stack([flat_batch, tf.constant(tags)]))
        return out_batch

    def tag(self, batch, num_samples):
        """
        Perform sequence tagging and sampling based on transition probabilities.

        Args:
            batch (tf.Tensor): Batch of tokens to be tagged.
            num_samples (int): Number of samples for probabilistic tagging.

        Returns:
            tf.Tensor: Batch with inferred tags.
        """
        # Extract existing tags from the input batch
        existing_tags = batch[:, :, 1]
        # Flatten the tag sequence and decode to strings
        sequence = [j.decode('utf-8') for j in tf.reshape(existing_tags, [-1]).numpy()]
        L = 0  # Initialize the lower bound of the sequence window

        # Iterate through the sequence
        for i in range(len(sequence)):
            # Reset the lower bound at the start of a new sequence
            if i % batch.shape[1] == 0:
                L = i
            # Increment the lower bound if the memory limit is exceeded
            elif i - L > self.n_limit:
                L += 1

            # If the current tag is empty, infer a tag
            if sequence[i] == '':
                # Build a lookup sequence of previous tokens
                lookup = ' -> '.join(sequence[L:i])
                state = self.TransitionStates.get(lookup, 0)  # Get the transition state

                # Fetch transition probabilities for the state
                dist = tf.nn.embedding_lookup(self.TransitionMatrix, state)
                logits = tf.math.log(dist)  # Convert probabilities to logits
                logits_2d = tf.reshape(logits, shape=(1, -1))  # Reshape logits for sampling

                # Sample tags based on probabilities
                samples = tf.random.categorical(logits_2d, num_samples=num_samples)
                samples = tf.squeeze(samples)
                counts = np.bincount(samples.numpy(), minlength=dist.shape[0])
                tag = list(self.tags.keys())[np.argmax(counts)]  # Determine the most frequent sampled tag
                sequence[i] = tag  # Assign the inferred tag

        # Reshape the input tokens and inferred tags into the original batch format
        in_tokens = tf.reshape(batch[:, :, 0], [-1])
        all_tags = tf.constant(sequence)
        return tf.reshape(tf.transpose(tf.stack([in_tokens, all_tags])), shape=batch.shape)

    def translate(self, processed_batch):
        o_batch = None
        for sequence in processed_batch:
            o_sequence = []
            for tup in sequence.numpy():
                if self.translator[tup[1].decode('utf-8')] == 2:
                    o_sequence = [tup[1].decode('utf-8')]
                    break
                elif self.translator[tup[1].decode('utf-8')]:
                    o_sequence.append(tup[1].decode('utf-8'))
                else:
                    o_sequence.append(tup[0].decode('utf-8'))
            o_tensor = tf.constant(o_sequence)
            if o_batch is not None:
                o_batch = tf.stack([o_batch, tf.expand_dims(o_tensor, axis = 0)])
            else:
                o_batch = tf.expand_dims(o_tensor, axis = 0)
        return o_batch

    def __call__(self, batch, num_samples=25, translate = True):
        """
        Encode a sequence and return the tagged batch.

        Args:
            batch (tf.Tensor): Batch of input tokens.
            num_samples (int): Number of samples for probabilistic tagging. Defaults to 25.

        Returns:
            tf.Tensor: Tagged batch.
        """
        # Flatten the batch and pre-tag tokens based on predefined mappings
        pretagged_batch = self.pretag(batch)
        # Reshape the pre-tagged batch for inference
        inference_batch = tf.reshape(pretagged_batch, shape=(batch.shape[0], batch.shape[1], 2))
        # Tag batch
        tagged_batch = self.tag(inference_batch, num_samples)
        # If translate
        if translate:
            return self.translate(tagged_batch) # Perform tagging and return the result
        else:
            return tagged_batch

    def addTransitions(self, tag_sequences):
        """
        Add new transitions to the transition matrix and update states.

        Args:
            tag_sequences (list[str]): List of new tag sequences to add.
        """
        # Generate new transition states based on the provided sequences
        transition_states = {
            tag_sequences[i]: i + len(self.TransitionStates.keys()) for i in range(len(tag_sequences))
        }
        self.TransitionStates = {**self.TransitionStates, **transition_states}  # Merge new states

        # Create new rows for the transition matrix with uniform probabilities
        n = len(tag_sequences)  # Number of new states
        m = self.TransitionMatrix.shape[1]  # Number of tags
        new_transitions = tf.cast(tf.fill([n, m], 1 / m), tf.float64)
        self.TransitionMatrix = tf.concat([self.TransitionMatrix, new_transitions], axis=0)  # Append new rows

    def train(self, ground_truths, num_epochs=None):
        """
        Train the transition matrix based on ground truth sequences.

        Args:
            ground_truths (tf.Tensor): Ground truth sequences with tokens and tags.
            num_epochs (int, optional): Number of epochs for training. Defaults to None.
        """
        # Flatten the ground truth tags into a sequence
        token_sequence = [t.decode('utf-8') for t in tf.reshape(ground_truths[:, :, 1], [-1]).numpy()]
        L = 0  # Initialize the lower bound of the sequence window

        # Iterate through the sequence
        for i in range(len(token_sequence) - 1):
            # Reset the lower bound at the start of a new sequence
            if i % ground_truths.shape[1] == 0:
                L = i
            # Increment the lower bound if the memory limit is exceeded
            elif i - L > self.n_limit:
                L += 1

            # Build a lookup sequence of previous tokens
            lookup = ' -> '.join(token_sequence[L:i])
            state = self.TransitionStates.get(lookup, 0)  # Get the transition state

            # Fetch transition probabilities for the state
            dist = tf.nn.embedding_lookup(self.TransitionMatrix, state)
            target = token_sequence[i]  # Get the target tag
            one_hot = tf.cast(tf.one_hot(self.tags[target], len(self.tags.keys())), tf.float64)  # One-hot encode the target

            # Update the transition probabilities over epochs
            for _ in range(num_epochs or 1):
                dist = tf.reduce_mean([dist, one_hot / self.cooling_factor], axis=0)  # Blend probabilities
                dist = dist / tf.reduce_sum(dist)  # Normalize the probabilities

            # Update the transition matrix with the new distribution
            self.TransitionMatrix = tf.tensor_scatter_nd_update(
                self.TransitionMatrix, tf.constant([[state]]), [dist]
            )
    def save(self):
        """
        Save the encoder's state and transition matrix to disk.
        """
        path = os.path.join(self.parent, "model")  # Define the save path
        os.makedirs(path, exist_ok=True)  # Ensure the directory exists

        # Save the transition states as a JSON file
        with open(os.path.join(path, "transition_states.json"), "w") as f:
            json.dump(self.TransitionStates, f, indent=4)

        # Save the transition matrix as a NumPy file
        np.save(os.path.join(path, "transition_matrix.npy"), self.TransitionMatrix.numpy())

    @classmethod
    def load(cls):
        """
        Load an encoder instance from saved files.

        Returns:
            MINT: An instance of the MINT encoder.
        """
        path = os.path.join(Path(__file__).resolve().parent, "model")  # Define the load path

        # Load transition states from JSON file
        with open(os.path.join(path, "transition_states.json"), "r") as f:
            transition_states = json.load(f)

        # Load transition matrix from NumPy file
        transition_matrix = tf.Variable(
            np.load(os.path.join(path, "transition_matrix.npy"))
        )

        # Return a new instance of the class with loaded parameters
        return cls(_transition_states=transition_states, _transition_matrix=transition_matrix)
