import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from hoopla.processing import generate_tokens

CACHE_DIR = Path("cache/")
INDEX_CACHE = CACHE_DIR / "index.pkl"
DOCMAP_CACHE = CACHE_DIR / "docmap.pkl"
TERM_FREQUENCIES_CACHE = CACHE_DIR / "term_frequencies.pkl"


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, list[int]] = defaultdict(lambda: [])
        self.docmap: dict[int, str] = dict()
        self.term_frequencies: dict[int, Counter] = dict()

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = generate_tokens(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in tokens:
            self.index[token].append(doc_id)

    def get_documents(self, term: str) -> list[int]:
        return self.index[term.lower()]

    def get_tf(self, doc_id: int, term: str):
        token = generate_tokens(term)
        if len(token) > 1:
            msg = f"Passed multiple tokens: {term}. Only a single token is allowed!"
            raise ValueError(msg)
        token = token[0]
        return self.term_frequencies[doc_id][token]

    def build(self):
        with open("movies.json", "r") as f:
            movies = json.load(f)
        for movie in tqdm(movies["movies"], desc="Building index"):
            self._add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie["title"]

    def save(self):
        CACHE_DIR.mkdir(exist_ok=True)
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump(dict(self.index), f)
        with open(DOCMAP_CACHE, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_FREQUENCIES_CACHE, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if not INDEX_CACHE.exists():
            msg = f"The index cache does not exist at {INDEX_CACHE}."
            raise FileNotFoundError(msg)
        if not DOCMAP_CACHE.exists():
            msg = f"The docmap cache does not exist at {DOCMAP_CACHE}."
            raise FileNotFoundError(msg)
        if not TERM_FREQUENCIES_CACHE.exists():
            msg = f"The docmap cache does not exist at {TERM_FREQUENCIES_CACHE}."
            raise FileNotFoundError(msg)

        with open(INDEX_CACHE, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_CACHE, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TERM_FREQUENCIES_CACHE, "rb") as f:
            self.term_frequencies = pickle.load(f)
