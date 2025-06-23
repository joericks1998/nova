import tensorflow as tf
from . import attention, ffnn

class Layer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate, autoregressive=True, name = None, **kwargs):
        super(Layer, self).__init__(name = name, **kwargs)
        # store inputs as attributes (for serialization)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.autoregressive = autoregressive
        # create dropout
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        # initialize attention layer
        self.attention = attention.PerformerLayer(d_model=self.d_model, num_heads=self.num_heads, autoregressive=self.autoregressive)
        # initialize deep layer
        self.ffnn = ffnn.Layer(self.d_model, self.dff)
        # create layer normalization
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon = 1e6)
        self.layernorm.trainable = True
        return
        
    # main transformer call
    @tf.function(reduce_retracing=True)
    def call(self, batch, autoregres=True, training=False, mask=None):
        # feed through attention mechanism (using self attention)
        attentionized = self.attention(batch, batch, batch, mask=mask)
        attentionized = self.dropout(attentionized, training=training)
        #residual connection
        attention_o = self.layernorm(batch + attentionized)
        # forward pass
        outputs = self.ffnn(attention_o)
        outputs = self.dropout(outputs, training=training)
        # residual connection
        o_actual = self.layernorm(attention_o + outputs)
        return outputs

    #parameters getter for model training
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config

    #custom config method (also for serialization)
    @classmethod
    def from_config(cls,config):
        return cls(**config)

    #parameters getter for model training
    @property
    def Parameters(self):
        tfmr_trainables = self.layernorm.trainable_variables
        return self.attention.Parameters + self.ffnn.Parameters + tfmr_trainables
