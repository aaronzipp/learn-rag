import argparse
import math

from hoopla.index import InvertedIndex, load_index
from hoopla.processing import generate_tokens, get_single_token


def search(search_term: str) -> None:
    index = load_index()

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

    tf_parser = subparsers.add_parser(
        "tf", help="Get the term frequency of a term for a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="A term")

    idf_parser = subparsers.add_parser(
        "idf", help="Get the inverse document frequency of a term"
    )
    idf_parser.add_argument("term", type=str, help="A term")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search(args.query)
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case "tf":
            index = load_index()
            print(index.get_tf(args.doc_id, args.term))
        case "idf":
            index = load_index()
            token = get_single_token(args.term)
            term_doc_count = len(index.index[token])
            idf = math.log((len(index.docmap) + 1) / (term_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
