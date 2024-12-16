import os
import json

def getVocab(path = "vocabulary.txt"):
    with open(path, "r") as f:
        vocab = file.read().split('\n')
    return vocab
