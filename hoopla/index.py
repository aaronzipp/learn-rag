import itertools
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from hoopla.processing import generate_tokens, get_single_token

CACHE_DIR = Path("cache/")
INDEX_CACHE = CACHE_DIR / "index.pkl"
DOCMAP_CACHE = CACHE_DIR / "docmap.pkl"
TERM_FREQUENCIES_CACHE = CACHE_DIR / "term_frequencies.pkl"
DOC_LENGTHS_CACHE = CACHE_DIR / "doc_lengths.pkl"
CACHES = {
    "index": INDEX_CACHE,
    "docmap": DOCMAP_CACHE,
    "term_frequencies": TERM_FREQUENCIES_CACHE,
    "doc_lengths": DOC_LENGTHS_CACHE,
}

BM25_K1 = 1.5
BM25_B = 0.75


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, list[int]] = defaultdict(lambda: [])
        self.docmap: dict[int, str] = dict()
        self.term_frequencies: dict[int, Counter] = dict()
        self.doc_lengths: dict[int, int] = dict()

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = generate_tokens(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].append(doc_id)

    def _get_average_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths.values())

    def get_documents(self, term: str) -> list[int]:
        return self.index[term.lower()]

    def get_tf(self, doc_id: int, term: str) -> float:
        token = get_single_token(term)
        return self.term_frequencies[doc_id][token]

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        length_normalization = (
            1 - b + b * (self.doc_lengths[doc_id] / self._get_average_doc_length())
        )
        return (tf * (1 + k1)) / (tf + k1 * length_normalization)

    def get_bm25_idf(self, term: str) -> float:
        token = get_single_token(term)
        document_frequency = len(self.index[token])
        return math.log(
            (len(self.docmap) - document_frequency + 0.5) / (document_frequency + 0.5)
            + 1
        )

    def get_bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, term: str, limit: int = 5) -> list[tuple[int, float]]:
        tokens = set(generate_tokens(term))
        scores = dict()
        for doc_id in self.doc_lengths.keys():
            scores[doc_id] = sum(self.get_bm25(doc_id, token) for token in tokens)

        return itertools.islice(
            dict(
                sorted(scores.items(), key=lambda item: item[1], reverse=True)
            ).items(),
            limit,
        )

    def build(self):
        with open("movies.json", "r") as f:
            movies = json.load(f)
        for movie in tqdm(movies["movies"], desc="Building index"):
            self._add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie["title"]

    def save(self):
        CACHE_DIR.mkdir(exist_ok=True)
        for attr, cache in CACHES.items():
            with open(cache, "wb") as f:
                pickle.dump(dict(getattr(self, attr)), f)

    def load(self):
        check_cache_existence(CACHES.values())

        for attr, cache in CACHES.items():
            with open(cache, "rb") as f:
                setattr(self, attr, pickle.load(f))


def check_cache_existence(files: list[Path]) -> None:
    for file in files:
        if not file.exists():
            msg = f"The following cache does not exist: {file}."
            raise FileNotFoundError(msg)


def load_index() -> InvertedIndex:
    index = InvertedIndex()
    index.load()
    return index
