import os

model_path = "model"

with open(os.path.join(model_path, 'vocabulary.txt'), 'r') as f:
    vocab = f.read().split('\n')

d_model = 2**7
nova_tfmr = 8

dff = 4*d_model
num_heads = 32
# vocab specs
vocab_len = len(vocab)

# temperature and dropout (will autoupdate model)
dropout_rate = 0.5
temperature = 0.02
