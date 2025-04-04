import tensorflow as tf
from . import attention, ffnn
from static import constants
from utils.model_io import master_config

class Layer(tf.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate = constants.dropout_rate, name = None):
        super(Layer, self).__init__(name = name)
        self.attention = attention.Layer(d_model, num_heads)
        self.ffnn = ffnn.Layer(d_model, dff)
        self.config = {
            "d_model": d_model,
            "num_heads": num_heads,
            "dff": dff,
            "dropout_rate": dropout_rate
        }
        # layer normalization
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon = 1e6)
        # dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def __call__(self, batch, training=False):
        # feed through attention mechanism
        attentionized = self.attention(batch, batch, batch)
        attentionized = self.dropout(attentionized, training=training)
        #residual connection
        attention_o = self.layernorm(batch + attentionized)

        # forward pass
        outputs = self.ffnn(attentionized)
        outputs = self.dropout(outputs, training=training)
        # residual connection
        o_actual = self.layernorm(attention_o + outputs)

        return outputs

    #parameters getter for model training
    def get_config(self):
        return master_config(Layer.__init__)

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls,config):
        return cls(**config)

    #parameters getter for model training
    @property
    def Parameters(self):
        tfmr_trainables = [self.layernorm.gamma,
                            self.layernorm.beta]
        return self.attention_mech.Parameters + self.ffnn.Parameters + tfmr_trainables
