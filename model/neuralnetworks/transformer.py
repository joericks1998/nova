import tensorflow as tf
from . import attention, ffnn
from .constants import transformer_constants

class TransformerLayer(tf.Module):
    def __init__(self, embed_dim, batch_size, dropout_rate = 0.1, name = None):
        super().__init__(name = name)
        self.attention_mech = attention.PerformerAttention(embed_dim, transformer_constants.num_attention_heads)
        self.ffnn = ffnn.FFNetwork(embed_dim, transformer_constants.dff)
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
