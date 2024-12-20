from text import data_io

d_model = 2**7
nova_tfmr = 8

dff = 4*d_model
num_heads = 32
# vocab specs
vocab_len = len(data_io.getVocab())

# temperature and dropout (will autoupdate model)
dropout_rate = 0.5
temperature = 0.02
