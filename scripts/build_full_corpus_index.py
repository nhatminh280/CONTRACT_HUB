from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indexing.bm25_store import BM25Store
from indexing.sql_store import ContractRecord, SQLStore
from ingestion.chunker import Chunk, chunk_blocks
from ingestion.extractor import extract_deterministic_record


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "full_corpus_index"
MANIFEST_PATH = ROOT / "data" / "ground_truth" / "contract_manifest.json"


@dataclass(frozen=True)
class FullCorpusBuildResult:
    output_dir: Path
    sqlite_path: Path
    bm25_path: Path
    chunks_path: Path
    structured_path: Path
    contract_count: int
    chunk_count: int


BlockLoader = Callable[[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["contract_id"]: row for row in rows}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_reference_pages(contract_id: str) -> list[dict[str, Any]]:
    rows = load_jsonl(ROOT / "data" / "processed" / "pdf_page_text_reference.jsonl")
    pages = [row for row in rows if row["contract_id"] == contract_id]
    return sorted(pages, key=lambda row: row["page_number"])


def load_blocks(contract_id: str, manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pdf_path = ROOT / manifest[contract_id]["pdf_path"]
    if has_module("fitz"):
        from ingestion.parser import parse_pdf

        return parse_pdf(str(pdf_path))

    return [
        {"text": row["text"], "page": row["page_number"], "type": "text_reference"}
        for row in load_reference_pages(contract_id)
    ]


def _chunk_payload(chunks: list[Chunk]) -> list[dict[str, Any]]:
    return [
        {
            "id": chunk.id,
            "text": chunk.text,
            "contract_id": chunk.contract_id,
            "clause_number": chunk.clause_number,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "clause_type": chunk.clause_type,
            "citation": chunk.citation,
        }
        for chunk in chunks
    ]


def _structured_payload(records: list[ContractRecord]) -> list[dict[str, Any]]:
    return [
        {
            "contract_id": record.contract_id,
            "title": record.title,
            "parties": record.parties,
            "value": record.value,
            "currency": record.currency,
            "effective_date": record.effective_date,
            "expiry_date": record.expiry_date,
            "governing_law": record.governing_law,
            "clauses": record.clauses,
        }
        for record in records
    ]


def build_full_corpus_index(
    manifest: dict[str, dict[str, Any]] | None = None,
    contract_ids: list[str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    block_loader: BlockLoader = load_blocks,
) -> FullCorpusBuildResult:
    manifest = manifest or load_manifest()
    selected_ids = contract_ids or list(manifest)
    output_dir = Path(output_dir)
    sqlite_path = output_dir / "contracts.sqlite"
    bm25_path = output_dir / "bm25_chunks.pkl"
    chunks_path = output_dir / "chunks.json"
    structured_path = output_dir / "structured_fields.json"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Chunk] = []
    records: list[ContractRecord] = []
    for contract_id in selected_ids:
        blocks = block_loader(contract_id, manifest)
        contract_chunks = chunk_blocks(blocks, contract_id=contract_id)
        chunks.extend(contract_chunks)
        records.append(extract_deterministic_record(contract_id, contract_chunks))

    store = SQLStore(str(sqlite_path))
    store.initialize()
    for record in records:
        store.upsert_contract(record)

    BM25Store(chunks).save(str(bm25_path))
    chunks_path.write_text(json.dumps(_chunk_payload(chunks), ensure_ascii=False, indent=2), encoding="utf-8")
    structured_path.write_text(json.dumps(_structured_payload(records), ensure_ascii=False, indent=2), encoding="utf-8")

    return FullCorpusBuildResult(
        output_dir=output_dir,
        sqlite_path=sqlite_path,
        bm25_path=bm25_path,
        chunks_path=chunks_path,
        structured_path=structured_path,
        contract_count=len(records),
        chunk_count=len(chunks),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic full-corpus demo index.")
    parser.add_argument("--contracts", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_full_corpus_index(contract_ids=args.contracts, output_dir=args.output_dir)
    print(f"Wrote {result.contract_count} contracts and {result.chunk_count} chunks to {result.output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
