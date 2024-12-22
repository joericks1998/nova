import tensorflow as tf
from training import training
from text import data_io
from static import _math
import numpy as np

model_path = "/Users/joericks/Desktop/nova/model"

def vocabMapper(logit, vocab = data_io.getVocab(path = model_path)):
    return vocab[logit]

def inBatch(text_batch, tokenizer):
    tokenBatch = list(map(tokenizer.word_split, text_batch))
    max_seq_len = max(list(map(len, tokenBatch)))
    padBatch = []
    for seq in tokenBatch:
        if len(seq) < max_seq_len:
            pads = ["<pad>" for i in range(0, max_seq_len - len(seq))]
            padBatch.append(seq+pads)
        else:
            padBatch.append(seq)
    return padBatch

def InferAll(ps):
    idx = ps.shape[1]-1
    return {
        "means": [[_math.mean(i) for i in seq][idx] for seq in ps],
        "shannon_entropies": [[-sum(i*np.log(i)) for i in seq][idx] for seq in ps],
        "argmax": [[tf.argmax(i) for i in seq][idx] for seq in ps]
    }

def InferEfficient(ps):
    return tf.argmax(ps, axis=2)[:,-2]

def Generator(text_batch, model = None, tokenizer = None, max_t = 25):
    print(f"Performing first pass..")
    in_batch = tf.Variable(inBatch(text_batch, tokenizer))
    in_len = in_batch.shape[1]
    j = 0
    out_batch = None
    print(f"Generating...")
    while j < max_t:
        if out_batch is not None:
            in_batch = tf.concat([in_batch, out_batch], axis = 1)
        probabilities = model.fPass(in_batch)
        inference = InferEfficient(probabilities)
        out_list = list(map(vocabMapper, inference))
        out_batch = tf.reshape(tf.Variable([out_list]), shape = (len(out_list), 1))
        j+=1
        if "<stop>" in out_batch.numpy():
            in_batch = tf.concat([in_batch, out_batch], axis = 1)
            return in_batch
    in_batch = tf.concat([in_batch, out_batch], axis = 1)
    responses = ["".join(list(map(str, s.numpy()))) for s in in_batch[:,in_len:]]
    return responses


def Trainer(data, model = None, tokenizer = None):
    #first build the batch from the training data
    #then forward pass and train
    pass
