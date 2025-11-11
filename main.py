import argparse

from hoopla.index import InvertedIndex
from hoopla.processing import generate_tokens


class CacheNotFoundError(Exception):
    """Raised when the required cache is missing."""

    pass


def search(search_term: str) -> None:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        msg = f"Cache is missing: {e}"
        raise CacheNotFoundError(msg) from e

    search_tokens = generate_tokens(search_term)
    results = []
    for token in search_tokens:
        results.extend(index.get_documents(token))
        if len(results) >= 5:
            results = results[:5]
            break
    for i, result in enumerate(results):
        print(f"{i + 1}. {index.docmap[result]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search(args.query)
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
