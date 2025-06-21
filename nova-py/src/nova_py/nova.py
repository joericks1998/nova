import tensorflow as tf
from . import TACO, NERf, PArMesan
import pathlib
import pickle
import json
import functools
import ast

tf.keras.mixed_precision.set_global_policy('mixed_float16')

MODEL_PATH = pathlib.Path(__file__).parent/"model"

class Model(tf.keras.Model):
    def __init__(self, feature_struct = None, name="nova", **kwargs):
        super().__init__(**kwargs)
        assert feature_struct is not None
        self.feature_struct = feature_struct
        with open(MODEL_PATH/"vocab.pkl", "rb") as p:
            self.vocabulary = pickle.load(p)
        with open(MODEL_PATH/"hyperparameters.json", "r") as f:
            self.hp = json.load(f)
            pass
        nerf_hp = {**self.hp["NERf"], **{"vocabulary_size": len(self.vocabulary.taco["tokens"].values()),
                                            "num_features": sum(self.feature_struct)}}
        parm_hp = {**self.hp["PArM"], **{"input_size": sum(self.feature_struct)+len(self.vocabulary.performer["out_tokens"].values()),
                                        "output_size": len(self.vocabulary.performer["out_tokens"].values())}}
        self.__pad = self.vocabulary.taco["tokens"]["<pad>"]
        self.taco = functools.partial(TACO.batch, Vocab=self.vocabulary.taco, pad_token=self.__pad)
        self.nerf = NERf.Model(**nerf_hp)
        self.parm = PArMesan.Model(**parm_hp)
        return
    # default save function
    def Save(self):
        self.save(filepath=MODEL_PATH/"nova.keras")
        return
    # call model
    @tf.function(reduce_retracing=True)
    def call(self, tokens, spans, training=False, token_limit=None):
        nerf_pad_mask = tf.cast(tokens != self.__pad, self.dtype)
        parm_pad_mask = tf.cast(spans != 0, self.dtype)
        # print("tagging...")
        nerf_pass = self.nerf(tokens, spans, training=training, mask=nerf_pad_mask)
        nerf_pass = tf.cast(nerf_pass, tf.int32) * tf.cast(parm_pad_mask, tf.int32)
        # print("generating...")
        parm_pass = self.parm(nerf_pass, training=training, token_limit=token_limit, mask=parm_pad_mask)
        return parm_pass

    def handle(self, in_batch, training=False, token_limit=250):
        if isinstance(in_batch, str):
            in_batch = [in_batch]
        elif not isinstance(in_batch, list):
            msg = '''
                inputs must be a list
            '''
            raise TypeError(msg)
        print("tokenizing...")
        tokens, spans = self.taco(in_batch)
        print("passing forward...")
        token_limit = tf.constant(token_limit, dtype=tf.int32)
        # forward pass
        forward_pass = self(tokens, spans, training=training, token_limit=token_limit)
        # Set up your vocabulary mapping
        # forward_pass: tf.Tensor of shape (batch_size, sequence_length), dtype=tf.int32
        output_ids = forward_pass.numpy()  # Convert to NumPy array
        token_map = self.vocabulary.performer["out_tokens"]  # Dict[int] -> str
        # Apply mapping with list comprehension
        decoded = [[token_map.get(token_id, "[UNK]") for token_id in sequence if token_id not in [0,1]] for sequence in output_ids]
        output = ["".join(arr) for arr in decoded]
        return output

def load_model(path=MODEL_PATH / "nova.keras"):
    return tf.keras.models.load_model(path, custom_objects={"Model": Model, "NERf.Model": NERf.Model, "PArMesan.Model": PArMesan.Model})

# def prepPythonDataset(python_strings, model=None):
#     if not isinstance(python_strings, list):
#         msg = "input must be of type \'list\'"
#         raise TypeError(msg)
#     trees = [ast.parse(ps) for ps in python_strings]
#     dumps = [ast.dump(tree) for tree in trees]
#     tokens, spans = model.taco(python_strings)
#     groundtruth_tokens =
#     return tokens, spans

# def train(model, dataset, epochs=1, learning_rate=1e-3):
#     """
#     Simple training loop for your model.
#
#     Args:
#         model: your tf.keras.Model (e.g., Nova)
#         dataset: tf.data.Dataset yielding (tokens, spans) and labels
#         epochs: number of times to loop through the dataset
#         learning_rate: optimizer learning rate
#     """
#     # Define optimizer
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     # Define loss function (assuming classification for now)
#     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     # Metric to track loss
#     train_loss = tf.keras.metrics.Mean(name='train_loss')
#     for epoch in range(epochs):
#         print(f"\nEpoch {epoch+1}/{epochs}")
#         # Reset the metric at the start of each epoch
#         train_loss.reset_states()
#         for step, (inputs, labels) in enumerate(dataset):
#             tokens, spans = inputs  # unpack input tuple (tokens, spans)
#             with tf.GradientTape() as tape:
#                 # Forward pass
#                 predictions = model(tokens, spans, training=True)
#                 # Compute loss
#                 loss = loss_fn(labels, predictions)
#             # Compute gradients
#             gradients = tape.gradient(loss, model.Parameters)
#             # Apply gradients
#             optimizer.apply_gradients(zip(gradients, model.Parameters))
#             # Update loss metric
#             train_loss.update_state(loss)
#             if step % 100 == 0:
#                 print(f"Step {step}: Loss = {train_loss.result().numpy():.4f}")
#         print(f"Epoch {epoch+1} Loss: {train_loss.result().numpy():.4f}")
