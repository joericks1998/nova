import os
import json

def getVocab(path = None):
    with open(os.path.join(path, "vocabulary.txt"), "r") as file:
        vocab = file.read().split('\n')
    return vocab
