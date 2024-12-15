d_model = 2**8
nova_tfmr = 32
dropout_rate = 0.1
dff = 4*d_model
num_heads = 4
# vocab specs

vocab_path = "model/vocabulary.txt"
vocabulary = []
with open(vocab_path, 'r') as file:
    vocabulary = file.read().split('\n')
