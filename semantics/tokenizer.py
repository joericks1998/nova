import re
import json
## need to update this tokenizer file with my current tokenizer made in different directory
def word_split(string):
    if string == "":
        msg = "Input string must not be of length 0"
        raise ValueError(msg)
    t = []
    t += string.lower().split(" ")
    return t
# this will be our input vocabulary
def create_vocab(texts): ## get the words from the text
    """generate vocab of individual words from input
    """
    texts_lower = texts.lower()
    words = texts_lower.split(" ")## split words
    vocab = set()
    for word in words:
        vocab.add(word)
    alph = "abcdefghijklmnopqrstuvwxyz"
    for char in texts:
        if char not in alph:
            vocab.add(char) 
    return vocab


def tokenize(text):
    vocab = create_vocab(text)
    tokens = []
    buffer = ""
    for char in text:
        buffer += char
        if buffer in vocab:
            tokens.append(buffer)
            buffer = "" ## reset buffer
    with open("input_vocab.txt", "w") as input:
        for item in vocab:
            input.write(item + "\n")
    return tokens