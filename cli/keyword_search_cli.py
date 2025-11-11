import argparse
import itertools
import json
import re
import string

PUNCTUATION_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def remove_punctuation(input_text: str):
    return input_text.translate(PUNCTUATION_TABLE)


def remove_multiple_whitespace(input_text: str):
    return re.sub(r"\s{2,}", " ", input_text.strip())


def generate_tokens(input_text: str) -> list[str]:
    return remove_punctuation(input_text.lower()).split()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open("movies.json", "r") as f:
        movies = json.load(f)

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_tokens = generate_tokens(args.query)
            result_label = 1
            for movie in movies["movies"]:
                title_tokens = generate_tokens(movie["title"])
                is_matching = any(
                    search_token in title_token
                    for search_token, title_token in itertools.product(
                        search_tokens, title_tokens
                    )
                )
                if is_matching:
                    print(f"{result_label}. {movie['title']}")
                    result_label += 1
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
