import re

class Tokenizer:
    def __init__(self):
        self.token_q = []
        self.var_q = []

    def __call__(self, string):
        self.token_q += string.split(" ")
        return
