from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3

from indexing.bm25_store import BM25Store, tokenize
from ingestion.chunker import Chunk
from retrieval.hybrid_search import ScoredChunk
from retrieval.router import Intent, classify_intent
from retrieval.text_to_sql import UnsupportedStructuredQuery, build_structured_query, execute_structured_query


@dataclass
class ContractQueryResult:
    query: str
    intent: Intent
    hits: list[ScoredChunk] = field(default_factory=list)
    structured_rows: list[dict] = field(default_factory=list)
    structured_kind: str | None = None
    structured_error: str | None = None


def _search_chunks(query: str, chunks: list[Chunk], top_k: int) -> list[ScoredChunk]:
    if not chunks:
        return []
    hits = BM25Store(chunks).search(query, top_k=top_k)
    if hits:
        return hits

    query_tokens = set(tokenize(query))
    scored = [
        ScoredChunk(chunk=chunk, score=float(sum(1 for token in tokenize(chunk.text) if token in query_tokens)))
        for chunk in chunks
    ]
    return [hit for hit in sorted(scored, key=lambda hit: hit.score, reverse=True)[:top_k] if hit.score > 0]


def _load_chunks(chunks: list[Chunk], bm25_path: str | None) -> list[Chunk]:
    if chunks or not bm25_path:
        return chunks
    try:
        return BM25Store.load(bm25_path).chunks
    except (OSError, EOFError, ValueError):
        return chunks


def run_contract_query(
    query: str,
    chunks: list[Chunk],
    db_path: str | None = None,
    bm25_path: str | None = None,
    top_k: int = 3,
) -> ContractQueryResult:
    intent = classify_intent(query)
    result = ContractQueryResult(query=query, intent=intent)
    available_chunks = _load_chunks(chunks, bm25_path)

    if intent == "structured" and db_path:
        try:
            structured = build_structured_query(query)
            result.structured_kind = structured.kind
            result.structured_rows = execute_structured_query(db_path, query)
        except (UnsupportedStructuredQuery, OSError, ValueError, sqlite3.Error) as exc:
            result.structured_kind = "unsupported"
            result.structured_error = str(exc)

    if not result.structured_rows:
        result.hits = _search_chunks(query, available_chunks, top_k=top_k)

    return result
