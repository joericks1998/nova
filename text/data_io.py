import os
import json
## output vocab
def getVocab(path = None):
    with open(os.path.join(path, "vocabulary.txt"), "r") as file:
        vocab = file.readlines()
    return vocab
