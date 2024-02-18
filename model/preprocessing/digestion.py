import nltk
# Download the 'punkt' tokenizer if not already downloaded
nltk.download('punkt')

# Open the file and read its content
with open('/Users/joericks/Desktop/nova/training/first_q.txt', "r") as file:
    q = file.read()

# Define delimiters for special characters
delims = ['``',"''"]

# Function to tokenize the input text
def tokenize(q):
    # Tokenize the input text using nltk.word_tokenize
    raw_tokenized = nltk.word_tokenize(q)
    tokenized = []  # List to store the tokenized text
    passed = False  # Flag to track if we are inside a pair of delimiters
    temp_str = []   # List to store tokens between delimiters
    for t in raw_tokenized:
        # Check if the token is a delimiter
        if t in delims:
            # If already inside delimiters, append the closing double quote
            if passed:
                temp_str.append('\"')
                # Add the tokenized string to the list and reset temp_str
                tokenized.append(" ".join(temp_str))
                temp_str = []
                passed = False  # Reset the flag
            else:
                # If not inside delimiters, append the opening double quote
                temp_str.append('\"')
                passed = True  # Set the flag
        # If inside delimiters, append the token to temp_str
        elif passed:
            temp_str.append(t)
        # If not inside delimiters, append the token to the tokenized list
        else:
            tokenized.append(t)
    return tokenized

# Call the tokenize function and print the result
print(tokenize(q))
