from __future__ import annotations

from dataclasses import dataclass
import pickle
import re

from ingestion.chunker import Chunk
from retrieval.hybrid_search import ScoredChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


@dataclass
class BM25Store:
    chunks: list[Chunk]

    def __post_init__(self) -> None:
        tokenized = [tokenize(chunk.text) for chunk in self.chunks]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            self._bm25 = None
        self._tokenized = tokenized

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        query_tokens = tokenize(query)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
        else:
            query_set = set(query_tokens)
            scores = [sum(1 for token in tokens if token in query_set) for tokens in self._tokenized]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [ScoredChunk(chunk=self.chunks[index], score=float(score)) for index, score in ranked if score > 0]

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            pickle.dump(self.chunks, file)

    @classmethod
    def load(cls, path: str) -> "BM25Store":
        with open(path, "rb") as file:
            chunks = pickle.load(file)
        return cls(chunks)
