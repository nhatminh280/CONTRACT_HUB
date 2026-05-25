from __future__ import annotations

from ingestion.chunker import Chunk
from retrieval.hybrid_search import ScoredChunk


class ChromaVectorStore:
    def __init__(self, persist_directory: str = "data/index/chroma", collection_name: str = "contract_chunks") -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.collection.add(
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "contract_id": chunk.contract_id,
                    "clause_number": chunk.clause_number,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "clause_type": chunk.clause_type,
                }
                for chunk in chunks
            ],
            ids=[chunk.id for chunk in chunks],
        )

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        result = self.collection.query(query_texts=[query], n_results=top_k)
        hits: list[ScoredChunk] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else [0.0] * len(ids)
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            chunk = Chunk(
                id=chunk_id,
                text=text,
                contract_id=metadata["contract_id"],
                clause_number=metadata["clause_number"],
                page_start=int(metadata["page_start"]),
                page_end=int(metadata["page_end"]),
                clause_type=metadata.get("clause_type", "general"),
            )
            hits.append(ScoredChunk(chunk=chunk, score=1.0 / (1.0 + float(distance))))
        return hits
