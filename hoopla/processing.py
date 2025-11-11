import string

from nltk.stem import PorterStemmer

PUNCTUATION_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))

with open("data/stopwords.txt") as f:
    STOPWORDS = f.read().splitlines()

STEMMER = PorterStemmer()


def remove_punctuation(input_text: str):
    return input_text.translate(PUNCTUATION_TABLE)


def generate_tokens(input_text: str) -> list[str]:
    tokens = remove_punctuation(input_text.lower()).split()
    return [STEMMER.stem(token) for token in tokens if token not in STOPWORDS]
