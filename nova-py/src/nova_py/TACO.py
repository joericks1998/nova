import re
import json
import tensorflow as tf

def word_split(string):
    if string == "":
        return
    space_arr = re.split(r'\s+', string)
    response = []
    for tkn in space_arr:
        response += [t.lower() for t in re.split(r'([,;"\'\n])', tkn) if t != '']
    return response

def inBatch(text_batch):
    token_batch = [t for t in map(word_split, text_batch) if t]
    max_seq_len = max(list(map(len, token_batch)))
    pad_batch = []
    for seq in token_batch:
        if len(seq) < max_seq_len:
            pads = [b"<pad>" for i in range(0, max_seq_len - len(seq))]
            pad_batch.append(seq+pads)
        else:
            pad_batch.append(seq)
    return tf.constant(pad_batch)
