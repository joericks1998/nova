class Vocabulary:
    def __init__(self):
        self.taco = {
            "patterns": {
                "quote_search": r'"[^"]*"|[^"\s]+',
                "word_match": r'\w+|[^\w\s]',
                "quote_split": r'\"+|.+(?<!\")'
            },
            "splits": ["\"", "'"],
            "tokens": {}
        }
        self.performer = {
            "in_tokens": {},
            "out_tokens": {}
        }
