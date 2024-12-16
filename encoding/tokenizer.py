import re
import json

class SpecialTokens:
    def __init__(self):
        with open('tokenization.json', 'r') as f:
            self.token_data = json.load(f)
    @property
    def Data(self):
        return self.token_data
    @Data.setter
    def Data(self):
        return

def word_split(string):
    t = []
    t += string.split(" ")
    return t

def build_go(tokens, ground_truth = True):
    if ground_truth = False:
        return
    return
