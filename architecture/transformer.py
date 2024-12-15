import tensorflow as tf
from . import attention, ffnn
from static import constants

class Layer(tf.Module):
    def __init__(self, embed_dim, num_heads, dff, dropout_rate = constants.dropout_rate, name = None):
        super(Layer, self).__init__(name = name)
        self.attention_mech = attention.Layer(embed_dim, num_heads)
        self.ffnn = ffnn.Layer(embed_dim, dff)
        # layer normalization
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon = 1e6)
        # dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def __call__(self, batch, training=False):
        # feed through attention mechanism
        attentionized = self.attention_mech(batch, batch, batch)
        attentionized = self.dropout(attentionized, training=training)
        #residual connection
        attention_o = self.layernorm(batch + attentionized)

        # forward pass
        outputs = self.ffnn(attentionized)
        outputs = self.dropout(outputs, training=training)
        # residual connection
        o_actual = self.layernorm(attention_o + outputs)

        return outputs
    def get_config(self):
        return

    @property
    def Parameters(self):
        tfmr_trainables = [self.layernorm.gamma,
                            self.layernorm.beta]
        return self.attention_mech.Parameters + self.ffnn.Parameters + tfmr_trainables
