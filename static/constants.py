from text import data_io

d_model = 2**8
nova_tfmr = 32
dropout_rate = 0.1
dff = 4*d_model
num_heads = 4
# vocab specs
vocab_len = len(data_io.getVocab())
