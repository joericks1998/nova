import tensorflow as tf
from training import training
from text import data_io
from static import _math
from semantics import parser
import numpy as np

model_path = "/Users/joericks/Desktop/nova/model"

def vocabMapper(logit, vocab = data_io.getVocab(path = model_path)):
    return vocab[logit]

def bytify(text_batch):
    for i in range(0, len(text_batch)):
        if isinstance(text_batch[i], str):
            text_batch[i] = text_batch[i].encode("utf-8")
        elif isinstance(text_batch[i], list):
            bytify(text_batch[i])
    return text_batch

def debytify(byte_batch):
    for i in range(0, len(byte_batch)):
        if isinstance(byte_batch[i], bytes):
            byte_batch[i] = byte_batch[i].decode("utf-8")
        elif isinstance(byte_batch[i], (list, np.ndarray)):
            debytify(byte_batch[i])
    return byte_batch

def inBatch(text_batch, tokenizer):
    token_batch = list(map(tokenizer.word_split, text_batch))
    max_seq_len = max(list(map(len, token_batch)))
    byte_batch = bytify(token_batch)
    pad_batch = []
    for seq in byte_batch:
        if len(seq) < max_seq_len:
            pads = [b"<pad>" for i in range(0, max_seq_len - len(seq))]
            pad_batch.append(seq+pads)
        else:
            pad_batch.append(seq)
    return tf.Variable(pad_batch)

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
    encoder = parser.Encoder.load("model/semantics")
    in_batch = inBatch(text_batch, tokenizer)
    in_len = in_batch.shape[1]
    in_batch = encoder(in_batch)
    print(in_batch)
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
    byted = in_batch[:,in_len:].numpy()
    debyted = debytify(byted)
    responses = []
    for seq in debyted.tolist():
        responses.append(" ".join(seq))
    return responses


def Trainer(data, model = None, tokenizer = None):
    #first build the batch from the training data
    #then forward pass and train
    pass
