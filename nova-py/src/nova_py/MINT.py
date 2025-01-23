import re
import tensorflow as tf
import numpy as np
import os
import json
import yaml
from pathlib import Path

# Memory class manages a dictionary-like structure for storing and retrieving data sequences.
class Memory:
    def __init__(self):
        self.Q = {}  # Dictionary to store sequences, keyed by optional identifiers.

    # Pop the last element of a sequence associated with key `k`.
    def pop(self, k=None):
        out = self.Q[k][len(self.Q[k]) - 1]  # Retrieve the last element.
        if len(self.Q[k]) > 2:  # If there are more than two elements, trim the sequence.
            self.Q[k] = self.Q[k][:len(self.Q[k]) - 2]
        return out

    # Push a new token to the sequence associated with key `k`.
    def push(self, token, k=None):
        self.Q[k].append(token)  # Append the token to the appropriate sequence.
        return

# Encoder class for encoding sequences based on predefined tagging and transition logic.
class Encoder(tf.Module):
    def __init__(self, _transition_matrix = None, _transition_states = None):
        # Load in tags
        self.parent = Path(__file__).resolve().parent
        with open(os.path.join(self.parent, "model/tags.json"), "r") as f:
            self.tags = json.load(f) # Dictionary of tag mappings.
        # Load predefinitions
        with open(os.path.join(self.parent, "model/predefined_tags.json"), "r") as f:
            self.predefinitions = json.load(f) # Predefined tag mappings for tokens.
        # Initialize transition matrix; defaults to a zero matrix if not provided.
        if _transition_matrix:
            self.TransitionMatrix = _transition_matrix
        else:
            self.TransitionMatrix = tf.Variable([[1/len(self.tags) for i in range(len(self.tags))]], dtype = tf.float64)
        # Initialize transition states; defaults to an empty dictionary if not provided.
        if _transition_states:
            self.TransitionStates = _transition_states
        else:
            self.TransitionStates = {'':0}
        # Store constants for encoder pattern limits
        with open(os.path.join(self.parent, "model/hyperparameters.yaml"), "r") as f:
            self.n_limit = yaml.safe_load(f)['encoder']['n_limit']

    def pretag(self, token):
        for tag, v in self.predefinitions.items():
            if token.numpy().decode('utf-8') in v:
                return tag
        return ''

    def tag(self, batch, num_samples):
        existing_tags = batch[:,:,1]
        sequence = [j.decode('utf-8') for j in tf.reshape(existing_tags, [-1]).numpy()]
        L = 0
        for i in range(0, len(sequence)):
            # reset lower bound when max sequence length is hit
            if i%batch.shape[1] == 0:
                L = i
            # increment lower bound limit when max memory limit is reached
            elif i - L > self.n_limit:
                L += 1
            # if tag is empty...
            if sequence[i] == '':
                # lookup sequence of previous tokens
                lookup = ' -> '.join(sequence[L:i])
                state = self.TransitionStates[lookup]
                # lookup probabilities using transition state
                dist = tf.nn.embedding_lookup(self.TransitionMatrix, state)
                # Convert probabilities to logits if needed (for stability)
                logits = tf.math.log(dist)
                # Reshape logits to 2D as `tf.random.categorical` expects 2D input
                logits_2d = tf.reshape(logits, shape=(1, -1))
                # Sampling
                samples = tf.random.categorical(logits_2d, num_samples=num_samples)
                samples = tf.squeeze(samples)
                counts = np.bincount(samples.numpy(), minlength = dist.shape[0])
                tag = list(self.tags.keys())[np.argmax(counts)]
                # assign sampled tag to sequence
                sequence[i] = tag
        # flatten tokens from in batch
        in_tokens = tf.reshape(batch[:,:,0], [-1])
        # make inference tags a tensor
        all_tags = tf.constant(sequence)
        # return inference batch in its original form with tags filled
        return tf.reshape(tf.transpose(tf.stack([in_tokens, all_tags])), shape = batch.shape)

    # Encoding function that processes a sequence and optionally uses a Memory instance.
    def __call__(self, batch, memory=None, num_samples=25):
        # Pre-tag tokens in the sequence based on predefinitions.
        flat_batch = tf.reshape(batch, [-1])
        tag_batch = tf.map_fn(self.pretag, flat_batch)
        pretagged_batch = tf.transpose(tf.stack([flat_batch, tag_batch]))
        inference_batch = tf.reshape(pretagged_batch, shape = (batch.shape[0], batch.shape[1], 2))
        # Begin inference
        out_batch = self.tag(inference_batch, num_samples)
        return out_batch # Return the encoded sequence as a string.
    # Add a new transition to the transition matrix.
    def addTransitions(self, tag_sequences):
        # add transition states
        transition_states = {tag_sequences[i]: i+len(self.TransitionStates.keys()) for i in range(0,len(tag_sequences))}
        self.TransitionStates = {**self.TransitionStates, **transition_states}
        # add rows to the matrix
        n = len(tag_sequences)
        m = self.TransitionMatrix.shape[1]
        new_transitions = tf.cast(tf.fill([n,m], 1/m), tf.float64)
        self.TransitionMatrix = tf.concat([self.TransitionMatrix, new_transitions], axis = 0)
        return


    def train(self, sequence_batch, ground_truths, num_epochs = None):
        self.addTransitions(sequence_batch)
        for sequence, ground_truth in zip(sequence_batch, ground_truths):
            gt_idx = self.tags[ground_truth]
            seq_idx = tf.constant([[self.TransitionStates[sequence]]])
            for i in range(num_epochs):
                probabilities = tf.nn.embedding_lookup(self.TransitionMatrix, seq_idx)
                one_hot_tensor = tf.one_hot(seq_idx, self.TransitionMatrix.shape[0])
                probabilities = tf.reduce_mean([probabilities, one_hot_tensor], axis=0)
                self.TransitionMatrix = tf.tensor_scatter_nd_update(self.TransitionMatrix, seq_idx, [probabilities])
                # however I assign this back to the transition matrix

    # Save the encoder state and transition matrix.
    def save(self):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(self.parent, "model")
        with open(os.path.join(path, "transition_states.json"), "w") as f:
            json.dump(self.TransitionStates, f)
        np.save(os.path.join(path, "transition_matrix.npy"), self.TransitionMatrix.numpy())
        return

    # Load an encoder instance from saved files.
    @classmethod
    def load(cls):
        path = os.path.join(self.parent, "model")
        with open(os.path.join(path, "transition_states.json"), "r") as f:
            transition_states = json.load(f)
        transition_matrix = tf.Variable(np.load(os.path.join(path, "transition_matrix.npy")))
        return cls(_transition_states=transition_states, _transition_matrix=transition_matrix)
