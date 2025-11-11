import json
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from hoopla.processing import generate_tokens

CACHE_DIR = Path("cache/")
INDEX_CACHE = CACHE_DIR / "index.pkl"
DOCMAP_CACHE = CACHE_DIR / "docmap.pkl"


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, list[int]] = defaultdict(lambda: [])
        self.docmap = dict()

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = generate_tokens(text)
        for token in tokens:
            self.index[token].append(doc_id)

    def get_documents(self, term: str) -> list[int]:
        return self.index[term.lower()]

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

    def load(self):
        if not INDEX_CACHE.exists():
            msg = f"The index cache does not exist at {INDEX_CACHE}."
            raise FileNotFoundError(msg)
        if not DOCMAP_CACHE.exists():
            msg = f"The docmap cache does not exist at {DOCMAP_CACHE}."
            raise FileNotFoundError(msg)
        with open(INDEX_CACHE, "rb") as f:
            self.index = pickle.load(f)

        with open(DOCMAP_CACHE, "rb") as f:
            self.docmap = pickle.load(f)
