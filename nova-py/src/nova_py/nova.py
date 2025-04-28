import tensorflow as tf
from . import TACO, NERf, PArMesan
import pathlib
import pickle
import json
import functools

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
    def call(self, in_batch, token_limit=250):
        if isinstance(in_batch, str):
            in_batch = [in_batch]
        elif not isinstance(in_batch, list):
            msg = '''
                inputs must be a list
            '''
            raise TypeError(msg)
        print("tokenizing...")
        tokens = self.taco(in_batch)
        nerf_pad_mask = tf.cast(tokens[0] != self.__pad, self.dtype)
        parm_pad_mask = tf.cast(tokens[1] != 0, self.dtype)
        print("tagging...")
        nerf_pass = self.nerf(tokens, mask=nerf_pad_mask)
        nerf_pass = tf.cast(nerf_pass, tf.int32) * tf.cast(parm_pad_mask, tf.int32)
        print("generating...")
        parm_pass = self.parm(nerf_pass, token_limit=token_limit, mask=parm_pad_mask)
        return parm_pass

def load_model(path=MODEL_PATH / "nova.keras"):
    return tf.keras.models.load_model(path, custom_objects={"Model": Model, "NERf.Model": NERf.Model, "PArMesan.Model": PArMesan.Model})
