import math
from typing import Annotated

import typer

from hoopla.index import BM25_K1, InvertedIndex, load_index
from hoopla.processing import generate_tokens, get_single_token

app = typer.Typer()


@app.command()
def search(search_term: str) -> None:
    print(f"Searching for: {search_term}")
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


@app.command()
def build() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


@app.command()
def tf(doc_id: int, term: str) -> None:
    index = load_index()
    print(index.get_tf(doc_id, term))


@app.command()
def idf(term: str):
    index = load_index()
    token = get_single_token(term)
    term_doc_count = len(index.index[token])
    idf = math.log((len(index.docmap) + 1) / (term_doc_count + 1))
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


@app.command()
def bm25tf(doc_id: int, term: str, k1: Annotated[float, typer.Argument()] = BM25_K1):
    index = load_index()
    tf = index.get_bm25_tf(doc_id, term, k1)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {tf:.2f}")


@app.command()
def bm25idf(term: str):
    index = load_index()
    bm25 = index.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25:.2f}")


@app.command()
def tfidf(doc_id: int, term: str) -> None:
    index = load_index()
    token = get_single_token(term)
    tf = index.get_tf(doc_id, term)
    term_doc_count = len(index.index[token])
    idf = math.log((len(index.docmap) + 1) / (term_doc_count + 1))
    tf_idf = tf * idf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


if __name__ == "__main__":
    app()
