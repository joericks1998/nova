from . import PArMesan
import tensorflow as tf
import pathlib
import json

#Path constants
MODEL_PATH = pathlib.Path(__file__).parent.resolve()/"model"

# custom handler (may need to change based on tokenizer)
def hpHandler(vocab_size):
    with open(MODEL_PATH/'hyperparameters.json', 'r') as f:
        params = json.load(f)
    nerf_params = {**params['NERf'], **{'vocab_size': vocab_size, 'num_features': params['concepts']}}
    parm_params = {**params['PArM'], **{'vocab_size': vocab_size, 'num_concepts': params['concepts'], 'nerf_params': nerf_params}}
    return parm_params

class Model(tf.keras.Model):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(name="MILAn", **kwargs)
        self.params = hpHandler(tokenizer.vocab_size)
        self.tokenizer = tokenizer
        self.learning_rate = 1e-4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def build(self, input_shape):
        self.PArM = PArMesan.Model(**self.params)
        self.ckpt = tf.train.Checkpoint(model=self.PArM, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=MODEL_PATH, max_to_keep=5)
    # model call from text prompt
    # @tf.function(reduce_retracing = True)
    def call(self, prompts=None, token_limit=100):
        in_batch = self.tokenizer(prompts, padding = True, return_tensors="tf")['input_ids']
        pad_mask = tf.cast(in_batch != 0, self.compute_dtype)
        model_response = self.PArM(in_batch, pad_mask, token_limit=token_limit)
        decoded_response = self.tokenizer.batch_decode(model_response.numpy(), skip_special_tokens = True)
        return decoded_response
    # basic training function
    # @tf.function(reduce_retracing = True)
    def train(self, inputs, mask, targets):
        with tf.GradientTape() as tape:
            logits = self.PArM(inputs, targets=targets, mask=mask, training=True)
            loss = self.loss_fn(targets, logits)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.PArM.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.PArM.trainable_variables))
        return loss

    @property
    def Size(self):
        return self.PArM.Size
