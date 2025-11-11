import json
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from hoopla.processing import generate_tokens


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
        Path("cache/").mkdir(exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(dict(self.index), f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
