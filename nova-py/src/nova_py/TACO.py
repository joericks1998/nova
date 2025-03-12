import re
import json
import tensorflow as tf

def inTokens(string):
    if string == "":
        return
    quote_match = r'"[^"]*"|[^"\s]+'
    splits = re.findall(quote_match, string)
    custom_match = r'\w+|[^\w\s]'
    quote_split = r'\"+|.+(?<!\")'
    tokens = []
    for i in splits:
        if "\"" in i:
            arr = re.findall(quote_split, i)
            tokens += arr
        else:
            arr = re.findall(custom_match, i)
            tokens += arr
    return tokens

def inBatch(text_batch):
    token_batch = [t for t in map(inTokens, text_batch) if t]
    max_seq_len = max(list(map(len, token_batch)))
    pad_batch = []
    for seq in token_batch:
        if len(seq) < max_seq_len:
            pads = [b"<pad>" for i in range(0, max_seq_len - len(seq))]
            pad_batch.append(seq+pads)
        else:
            pad_batch.append(seq)
    return tf.constant(pad_batch)
