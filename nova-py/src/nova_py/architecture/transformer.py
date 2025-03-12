import tensorflow as tf
from . import attention, ffnn

class Layer(tf.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate, name = None):
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
        self.built = True
    # @tf.function(reduce_retracing=True)
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
        return

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls,config):
        return cls(**config)

    #parameters getter for model training
    @property
    def Parameters(self):
        tfmr_trainables = self.layernorm.trainable_variables
        return self.attention.Parameters + self.ffnn.Parameters + tfmr_trainables
