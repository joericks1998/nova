import re
import json

def word_split(string):
    if string == "":
        msg = "Input string must not be of length 0"
        raise ValueError(msg)
    t = []
    t += string.lower().split(" ")
    return t
