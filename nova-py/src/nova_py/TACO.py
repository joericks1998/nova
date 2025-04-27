import re
import json
import tensorflow as tf
import pickle
from pathlib import Path
from .architecture.vocabulary import Vocabulary
import functools

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

def tokenize(string, Vocab=None):
    if string == "":
        return
    if Vocab is None:
        msg = """
            Rules must be passed!
        """
        raise ValueError(msg)
    patterns = Vocab["patterns"]
    splits = re.findall(patterns['quote_search'], string)
    words = []
    for i in splits:
        if "\"" in i:
            arr = re.findall(patterns["quote_split"], i)
            words += arr
        else:
            arr = re.findall(patterns["word_match"], i)
            words += arr
    subwords = [word[i:i+2] for word in words for i in range(0, len(word), 2)]
    tokens = [Vocab["tokens"][sub] for sub in subwords]
    spans = [partitions(len(word), 2) for word in words]
    return tokens, spans

def batch(text_batch, pad_token=None, Vocab=None):
    tokenizer = functools.partial(tokenize, Vocab=Vocab)
    token_batch = [t[0] for t in map(tokenizer, text_batch) if t]
    t_max_seq_len = max(list(map(len, token_batch)))
    pad_batch = []
    for seq in token_batch:
        if len(seq) < t_max_seq_len:
            pads = [pad_token for i in range(0, t_max_seq_len - len(seq))]
            pad_batch.append(seq+pads)
        else:
            pad_batch.append(seq)
    span_batch = [t[1] for t in map(tokenizer, text_batch) if t]
    s_max_seq_len = max(list(map(len, span_batch)))
    pad_spans = []
    for seq in span_batch:
        if len(seq) < s_max_seq_len:
            pads = [0 for i in range(0, s_max_seq_len - len(seq))]
            pad_spans.append(seq+pads)
        else:
            pad_spans.append(seq)
    return tf.constant(pad_batch), tf.constant(pad_spans)
