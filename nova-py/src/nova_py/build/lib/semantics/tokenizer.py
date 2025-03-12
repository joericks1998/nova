import re
import json
import tensorflow as tf

def word_split(string):
    if string == "":
        msg = "Input string must not be of length 0"
        raise ValueError(msg)
    t = []
    t += string.readlines()
    return t

def inBatch(text_batch):
    token_batch = list(map(word_split, text_batch))
    max_seq_len = max(list(map(len, token_batch)))
    byte_batch = tf.strings.unicode_encode(token_batch, 'UTF-8')
    pad_batch = []
    for seq in byte_batch:
        if len(seq) < max_seq_len:
            pads = [b"<pad>" for i in range(0, max_seq_len - len(seq))]
            pad_batch.append(seq+pads)
        else:
            pad_batch.append(seq)
    return tf.constant(pad_batch)
