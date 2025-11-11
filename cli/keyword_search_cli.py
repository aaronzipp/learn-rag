import argparse
import itertools
import json
import re
import string

from nltk.stem import PorterStemmer

PUNCTUATION_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def remove_punctuation(input_text: str):
    return input_text.translate(PUNCTUATION_TABLE)


def remove_multiple_whitespace(input_text: str):
    return re.sub(r"\s{2,}", " ", input_text.strip())


def generate_tokens(
    input_text: str, stopwords: list[str], stemmer: PorterStemmer
) -> list[str]:
    tokens = remove_punctuation(input_text.lower()).split()
    return [stemmer.stem(token) for token in set(tokens) - set(stopwords)]


def search(search_term: str) -> None:
    with open("movies.json", "r") as f:
        movies = json.load(f)

    with open("data/stopwords.txt") as f:
        stopwords = f.read().splitlines()

    stemmer = PorterStemmer()

    search_tokens = generate_tokens(search_term, stopwords, stemmer)
    result_label = 1
    for movie in movies["movies"]:
        title_tokens = generate_tokens(movie["title"], stopwords, stemmer)
        is_matching = any(
            search_token in title_token
            for search_token, title_token in itertools.product(
                search_tokens, title_tokens
            )
        )
        if is_matching:
            print(f"{result_label}. {movie['title']}")
            result_label += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
