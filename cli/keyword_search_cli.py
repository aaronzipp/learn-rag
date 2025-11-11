import argparse
import json
import re
import string

PUNCTUATION_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def remove_punctuation(input_text: str):
    return input_text.translate(PUNCTUATION_TABLE)


def remove_multiple_whitespace(input_text: str):
    return re.sub(r"\s{2,}", " ", input_text.strip())


def process_text(input_text: str):
    return remove_multiple_whitespace(remove_punctuation(input_text.lower()))


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
            search_term = process_text(args.query)
            n_results = 1
            for movie in movies["movies"]:
                if search_term in process_text(movie["title"]):
                    print(f"{n_results}. {movie['title']}")
                    n_results += 1
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
