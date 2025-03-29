import re
import json
import tensorflow as tf
import pickle
from pathlib import Path
from .architecture.vocabulary import Vocabulary

Base_path = Path(__file__).parent

with open(Base_path/"model/vocab.pkl", "rb") as f:
    Vocab = pickle.load(f).taco

def partitions(n, m, min_val=1):
    if m == 1:
        if n >= min_val:
            return 1
        else:
            return 0
    count = 0
    for i in range(min_val, n // m + 1 + 1):
        count += partitions(n - i, m - 1, i)
    if n % 2 == 0:
        return count
    else:
        return count + 1

def tokenize(string):
    if string == "":
        return
    quote_match = r'"[^"]*"|[^"\s]+'
    splits = re.findall(quote_match, string)
    custom_match = r'\w+|[^\w\s]'
    quote_split = r'\"+|.+(?<!\")'
    words = []
    for i in splits:
        if "\"" in i:
            arr = re.findall(quote_split, i)
            words += arr
        else:
            arr = re.findall(custom_match, i)
            words += arr
    subwords = [word[i:i+2] for word in words for i in range(0, len(word), 2)]
    tokens = [Vocab["tokens"][sub] for sub in subwords]
    spans = [partitions(len(word), 2) for word in words]
    return tokens, spans

def batch(text_batch):
    pad_token = Vocab["tokens"]["<pad>"]
    token_batch = [t[0] for t in map(tokenize, text_batch) if t]
    t_max_seq_len = max(list(map(len, token_batch)))
    pad_batch = []
    for seq in token_batch:
        if len(seq) < t_max_seq_len:
            pads = [pad_token for i in range(0, t_max_seq_len - len(seq))]
            pad_batch.append(seq+pads)
        else:
            pad_batch.append(seq)
    span_batch = [t[1] for t in map(tokenize, text_batch) if t]
    s_max_seq_len = max(list(map(len, span_batch)))
    pad_spans = []
    for seq in span_batch:
        if len(seq) < s_max_seq_len:
            pads = [0 for i in range(0, s_max_seq_len - len(seq))]
            pad_spans.append(seq+pads)
        else:
            pad_spans.append(seq)
    return tf.constant(pad_batch), tf.constant(pad_spans)
