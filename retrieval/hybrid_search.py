from __future__ import annotations

from dataclasses import dataclass

from ingestion.chunker import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def fuse(vector_hits: list[ScoredChunk], bm25_hits: list[ScoredChunk], k: int = 60, top_k: int = 10) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}

    for rank, hit in enumerate(vector_hits):
        chunks.setdefault(hit.chunk.id, hit.chunk)
        scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + rrf_score(rank, k)
        scores[hit.chunk.id] += 1e-9 / (rank + 1)

    for rank, hit in enumerate(bm25_hits):
        chunks.setdefault(hit.chunk.id, hit.chunk)
        scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + rrf_score(rank, k)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [ScoredChunk(chunk=chunks[chunk_id], score=score) for chunk_id, score in ranked]


def hybrid_search(query: str, vector_store, bm25_store, top_k: int = 10) -> list[ScoredChunk]:
    vector_hits = vector_store.search(query, top_k=top_k)
    bm25_hits = bm25_store.search(query, top_k=top_k)
    return fuse(vector_hits, bm25_hits, top_k=top_k)
