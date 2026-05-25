from __future__ import annotations

from retrieval.hybrid_search import ScoredChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[ScoredChunk], top_k: int = 3) -> list[ScoredChunk]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self.model.predict(pairs)
        rescored = [ScoredChunk(chunk=hit.chunk, score=float(score)) for hit, score in zip(hits, scores)]
        return sorted(rescored, key=lambda hit: hit.score, reverse=True)[:top_k]


def rerank(query: str, hits: list[ScoredChunk], top_k: int = 3) -> list[ScoredChunk]:
    try:
        return CrossEncoderReranker().rerank(query, hits, top_k=top_k)
    except ImportError:
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
