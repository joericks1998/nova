import re
import tensorflow as tf
import numpy as np
import os
import json
import Levenshtein

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
    def __init__(self, tags=None, n_limit=None, predefinitions=None,
                 transition_matrix=None, transition_states=None):
        # Store constants for encoder pattern limits
        self.constants = {
            "n_limit": n_limit,
            "max_combos": (len(tags) - 1) ** n_limit
        }

        # Initialize transition matrix; defaults to a zero matrix if not provided.
        if transition_matrix is None:
            self.TransitionMatrix = None
        else:
            self.TransitionMatrix = transition_matrix

        # Initialize transition states; defaults to an empty dictionary if not provided.
        if not transition_states:
            self.TransitionStates = {}
        else:
            self.TransitionStates = transition_states

        self.tags = tags  # Dictionary of tag mappings.
        self.predefinitions = predefinitions  # Predefined tag mappings for tokens.
    @staticmethod
    def reenforce(probabilities, mode, num_epochs = 1, is_bad = False, num_bad = 0):
        for epoch in range(num_epochs):
            if is_bad:
                # num_bad.assign_add(1)  # Increment the number of bad updates
                # Update probabilities by averaging the wrong answer's probability with 0
                probabilities = tf.tensor_scatter_nd_update(probabilities, [[mode]], [tf.reduce_mean([probabilities[mode], 0])])
                probabilities /= tf.reduce_sum(probabilities)  # Normalize after update
            else:
                # Reduce the wrong answers' probabilities to 0 (like penalizing)
                one_hot_tensor = tf.one_hot(mode, probabilities.shape[0])  # Create a tensor of zeros with the same shape as wrong_probs
                probabilities = tf.reduce_mean([probabilities, one_hot_tensor], axis=0)
       # Normalize after update
        return probabilities
    # Encoding function that processes a sequence and optionally uses a Memory instance.
    def __call__(self, batch, memory=None, training=(False, False)):
        # Pre-tag tokens in the sequence based on predefinitions.
        encoded_batch = None
        for sequence in batch:
            tagged_data = []
            for tkn in sequence.numpy():
                token = tkn.decode('utf-8')
                tagged = False
                for k in self.predefinitions.keys():
                    if token in self.predefinitions[k]:
                        tagged_data.append((token, k))  # Assign the predefined tag.
                        tagged = True
                if not tagged:
                    tagged_data.append((token, None))  # Assign no tag if no predefinition matches.
            tag_dict = dict(tagged_data)  # Create a dictionary of tokens and their tags.
            i = 0
            for k in tag_dict.keys():
                tag_seq = list(tag_dict.values())
                if not tag_dict[k]:  # If the token has no tag.
                    if i == 0:
                        idx = self.TransitionStates[""]  # Default state if no prior tokens.
                    else:
                        # Lookup the current state based on prior sequence.
                        idx = self.TransitionStates[" -> ".join([tkn for tkn in tag_seq[:i]])]
                    # Lookup the embedding vector from the transition matrix.
                    vec = tf.nn.embedding_lookup(self.TransitionMatrix, idx)
                    print(vec)
                    tag_tensor = tf.Variable([list(map(float, self.tags.values()))])
                    print(tag_tensor)
                    dot = tf.tensordot(tag_tensor, vec, axes=1)[0]  # Compute dot product.
                    tag_dict[k] = [k for k in self.tags.keys() if self.tags[k] == dot][0]  # Assign tag.
                i += 1
            encoded_arr = []
            for k, v in tag_dict.items():
                if v == '~pad~':  # Add padding tokens
                    encoded_arr.append(v)
                    # continue
                elif v == '~relation~':  # Handle relation tokens.
                    encoded_arr.append(k)
                else:
                    if memory:
                        memory.push(v, k=k)  # Push variable to memory if applicable.
                    # Further classify value tokens.
                    #
                    if v == '~value~':
                        if k.isdigit():
                            v = '~value.int~'
                        elif k.isdecimal():
                            v = '~value.float~'
                        elif k.lower() in ("null", "none"):
                            v = '~value.null~'
                        else:
                            v = '~value.string~'
                    encoded_arr.append(v)  # Add token to encoded array.
            encoded_vec = tf.Variable([encoded_arr])
            if encoded_batch is None:
                encoded_batch = encoded_vec
            else:
                encoded_batch = tf.concat([encoded_batch, encoded_vec], axis = 0)
        return encoded_batch  # Return the encoded sequence as a string.
    # Add a new transition to the transition matrix.
    def addTransition(self, tag_seq):
        if tag_seq in self.TransitionStates.keys():
            return
        if not self.TransitionStates:
            self.TransitionStates = {tag_seq: 0}
            self.TransitionMatrix = tf.Variable([[1/len(self.tags) for i in self.tags]],
                        dtype = tf.float64)
        else:
            depth = len(self.tags.keys())
            scores = tf.constant([1-Levenshtein.distance(tag_seq.replace(' -> ', ''),
                    k.replace(' -> ', ''))/max(len(k), len(tag_seq)) for k, v in self.TransitionStates.items()],
                    dtype = tf.float64)
            p_matrix = self.TransitionMatrix * scores[:, tf.newaxis]
            # print(scores)
            if p_matrix.shape[0]>1:
                logits = tf.constant([[sum(p_matrix[:,i].numpy()) for i in range(0, p_matrix.shape[1])]],
                        dtype = tf.float64)
            else:
                logits = p_matrix
            transition = tf.nn.softmax(logits, axis=-1)
            print(logits)
            self.TransitionMatrix = tf.concat([self.TransitionMatrix, transition], axis=0)  # Update matrix.
            self.TransitionStates[tag_seq] = self.TransitionMatrix.shape[0]-1  # Record the transition state.
        return

    # Delete a row from the transition matrix.
    def deleteTransition(self, tag_seq, target):
        if isinstance(target, str):
            target = [target]
        for tkn in target:
            if not tkn in self.tags.keys():
                raise ValueError("Invalid Target Tag")

        row_to_delete = self.TransitionStates[tag_seq]  # Get row index to delete.
        new_transition_matrix = tf.concat(
            [self.TransitionMatrix[:row_to_delete],
             self.TransitionMatrix[row_to_delete + 1:]],
            axis=0
        )  # Remove the row.
        self.TransitionMatrix.assign(new_transition_matrix)  # Update matrix.
        return

    # Update an existing transition.
    def updateTransition(self, tag_seq, target):
        self.deleteTransition(tag_seq, target)  # Delete the current transition.
        self.addTransition(tag_seq, target)  # Add the new transition.
        return

    # Save the encoder state and transition matrix.
    def save(self, path=None):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "predefined_tags.json"), "w") as f:
            json.dump(self.predefinitions, f)
        with open(os.path.join(path, "tags.json"), "w") as f:
            json.dump(self.tags, f)
        with open(os.path.join(path, "transition_states.json"), "w") as f:
            json.dump(self.TransitionStates, f)
        with open(os.path.join(path, "constants.json"), "w") as f:
            json.dump(self.constants, f)
        np.save(os.path.join(path, "transition_matrix.npy"), self.TransitionMatrix.numpy())
        return

    # Load an encoder instance from saved files.
    @classmethod
    def load(cls, path=None):
        with open(os.path.join(path, "predefined_tags.json"), "r") as f:
            predef_tags = json.load(f)
        with open(os.path.join(path, "tags.json"), "r") as f:
            tags = json.load(f)
        with open(os.path.join(path, "transition_states.json"), "r") as f:
            transition_states = json.load(f)
        with open(os.path.join(path, "constants.json"), "r") as f:
            n_limit = json.load(f)["n_limit"]
        transition_matrix = tf.Variable(np.load(os.path.join(path, "transition_matrix.npy")))
        return cls(tags=tags, n_limit=n_limit, predefinitions=predef_tags,
                   transition_states=transition_states, transition_matrix=transition_matrix)
