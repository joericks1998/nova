import os
import json

def getVocab(path = os.path.join(os.path.dirname(__file__), "vocabulary.txt")):
    with open(path, "r") as file:
        vocab = file.read().split('\n')
    return vocab
