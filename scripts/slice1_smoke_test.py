from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_DIR = ROOT / "outputs"
SMOKE_INDEX_DIR = OUTPUT_DIR / "slice1_smoke_index"
RESULTS_PATH = OUTPUT_DIR / "slice1_smoke_test_results.md"
PAGE_TEXT_PATH = OUTPUT_DIR / "slice1_processed_page_text.jsonl"
CHUNKS_PATH = OUTPUT_DIR / "slice1_chunks.json"
STRUCTURED_PATH = OUTPUT_DIR / "slice1_structured_fields.json"
SQLITE_PATH = SMOKE_INDEX_DIR / "contracts.sqlite"
BM25_PATH = SMOKE_INDEX_DIR / "bm25_chunks.pkl"
CHROMA_DIR = SMOKE_INDEX_DIR / "chroma"

PDF_CONTRACT_ID = "contract_005"
SCAN_CONTRACT_ID = "contract_005"
QUERY_IDS = ["q003", "q004", "q007"]

from generation.prompts import format_context
from indexing.bm25_store import BM25Store
from indexing.sql_store import ContractRecord, SQLStore
from ingestion.chunker import Chunk, chunk_blocks
from retrieval.hybrid_search import ScoredChunk, fuse


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_reference_pages(contract_id: str) -> list[dict[str, Any]]:
    rows = load_jsonl(ROOT / "data" / "processed" / "pdf_page_text_reference.jsonl")
    pages = [row for row in rows if row["contract_id"] == contract_id]
    return sorted(pages, key=lambda row: row["page_number"])


def load_manifest() -> dict[str, dict[str, Any]]:
    with (ROOT / "data" / "ground_truth" / "contract_manifest.json").open(encoding="utf-8") as file:
        rows = json.load(file)
    return {row["contract_id"]: row for row in rows}


def pdf_blocks(contract_id: str, manifest: dict[str, dict[str, Any]], statuses: list[str]) -> list[dict[str, Any]]:
    pdf_path = ROOT / manifest[contract_id]["pdf_path"]
    if has_module("fitz"):
        from ingestion.parser import parse_pdf
        from ingestion.router import route

        kind = route(str(pdf_path))
        blocks = parse_pdf(str(pdf_path))
        statuses.append(f"PDF ingest: {pdf_path.relative_to(ROOT)} routed as `{kind}` with {len(blocks)} parsed blocks.")
        return blocks

    pages = load_reference_pages(contract_id)
    statuses.append(
        "PDF ingest: PyMuPDF is not installed, so smoke test used "
        "`data/processed/pdf_page_text_reference.jsonl` as prepared page-text fallback."
    )
    return [{"text": row["text"], "page": row["page_number"], "type": "text_reference"} for row in pages]


def scanned_blocks(contract_id: str, manifest: dict[str, dict[str, Any]], statuses: list[str]) -> list[dict[str, Any]]:
    scan_dir = ROOT / manifest[contract_id]["scan_dir"]
    with (ROOT / "data" / "raw" / "scanned_simulated" / "scan_manifest.csv").open(encoding="utf-8") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["contract_id"] == contract_id and row["variant"] == "clean_render"
        ]
    missing_images = [row["image_path"] for row in rows if not (ROOT / row["image_path"]).exists()]
    if missing_images:
        raise FileNotFoundError(f"Missing scanned images: {missing_images}")

    if has_module("paddleocr") and os.getenv("SLICE1_REAL_OCR", "1") != "0":
        try:
            from ingestion.ocr import PaddleOCRVLRunner

            runner = PaddleOCRVLRunner()
            blocks = runner.parse_image_folder(str(scan_dir))
            runner.unload()
            statuses.append(
                f"Scanned ingest: ran PaddleOCR-VL on {len(rows)} clean rendered images "
                f"in `{scan_dir.relative_to(ROOT)}` and produced {len(blocks)} OCR blocks."
            )
            return blocks
        except Exception as exc:
            statuses.append(
                "Scanned ingest: PaddleOCR-VL was available but failed at runtime; "
                f"falling back to prepared reference text. Error: `{type(exc).__name__}: {exc}`"
            )

    pages_by_number = {row["page_number"]: row for row in load_reference_pages(contract_id)}
    blocks = []
    for row in sorted(rows, key=lambda item: int(item["page_number"])):
        page_number = int(row["page_number"])
        reference = pages_by_number.get(page_number)
        if reference:
            blocks.append({"text": reference["text"], "page": page_number, "type": "simulated_ocr_reference"})

    statuses.append(
        f"Scanned ingest: found {len(rows)} clean rendered images in `{scan_dir.relative_to(ROOT)}`; "
        "used prepared reference text to simulate OCR output for this smoke test."
    )
    return blocks


def write_page_text(pdf: list[dict[str, Any]], scanned: list[dict[str, Any]]) -> None:
    with PAGE_TEXT_PATH.open("w", encoding="utf-8") as file:
        for source, blocks in [("pdf", pdf), ("scanned_simulated", scanned)]:
            for block in blocks:
                file.write(
                    json.dumps(
                        {
                            "source": source,
                            "page_number": block["page"],
                            "type": block["type"],
                            "char_count": len(block["text"]),
                            "text": block["text"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def extract_structured_fields(contract_id: str, manifest: dict[str, dict[str, Any]], chunks: list[Chunk]) -> ContractRecord:
    first_line = next((line.strip() for line in chunks[0].text.splitlines() if line.strip()), contract_id)
    clauses = [
        {
            "number": chunk.clause_number,
            "type": chunk.clause_type,
            "page": chunk.page_start,
            "summary": chunk.text[:240].replace("\n", " "),
        }
        for chunk in chunks
    ]
    return ContractRecord(
        contract_id=contract_id,
        title=first_line,
        parties=[],
        effective_date=None,
        expiry_date=None,
        governing_law=None,
        clauses=clauses,
    )


def persist_structured(record: ContractRecord) -> None:
    SQLStore(str(SQLITE_PATH)).initialize()
    SQLStore(str(SQLITE_PATH)).upsert_contract(record)
    STRUCTURED_PATH.write_text(
        json.dumps(
            {
                "contract_id": record.contract_id,
                "title": record.title,
                "parties": record.parties,
                "effective_date": record.effective_date,
                "expiry_date": record.expiry_date,
                "governing_law": record.governing_law,
                "clauses": record.clauses,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def sqlite_counts() -> dict[str, int]:
    with sqlite3.connect(SQLITE_PATH) as conn:
        return {
            "contracts": conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0],
            "parties": conn.execute("SELECT COUNT(*) FROM parties").fetchone()[0],
            "clauses": conn.execute("SELECT COUNT(*) FROM clauses").fetchone()[0],
        }


def vectorize(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        vector[index] += 1.0
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def build_chroma(chunks: list[Chunk], statuses: list[str]) -> Any | None:
    if not has_module("chromadb"):
        statuses.append("Chroma index: skipped because `chromadb` is not installed.")
        return None

    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("slice1_smoke_chunks")
    existing = collection.get(limit=1)
    if existing.get("ids"):
        collection.delete(ids=collection.get()["ids"])
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=[vectorize(chunk.text) for chunk in chunks],
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
    )
    statuses.append(f"Chroma index: built collection `slice1_smoke_chunks` with {len(chunks)} chunks.")
    return collection


def search_chroma(collection: Any | None, query: str, chunks_by_id: dict[str, Chunk], top_k: int = 10) -> list[ScoredChunk]:
    if collection is None:
        return []
    result = collection.query(query_embeddings=[vectorize(query)], n_results=top_k)
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0] if result.get("distances") else [0.0] * len(ids)
    return [ScoredChunk(chunks_by_id[chunk_id], 1.0 / (1.0 + float(distance))) for chunk_id, distance in zip(ids, distances)]


def snippet_for_query(hit: ScoredChunk, expected_terms: list[str]) -> str:
    text = " ".join(hit.chunk.text.split())
    lower = text.lower()
    positions = [lower.find(term.lower()) for term in expected_terms if lower.find(term.lower()) >= 0]
    if positions:
        start = max(min(positions) - 120, 0)
        end = min(max(positions) + 320, len(text))
        return text[start:end]
    return text[:420]


def retrieval_query(test_case: dict[str, Any]) -> str:
    match = re.search(r'"([^"]+)"', test_case["query"])
    return match.group(1) if match else test_case["query"]


def run_queries(chunks: list[Chunk], collection: Any | None) -> list[dict[str, Any]]:
    with (ROOT / "data" / "ground_truth" / "test_cases.json").open(encoding="utf-8") as file:
        cases_by_id = {case["query_id"]: case for case in json.load(file)}
    test_cases = [cases_by_id[query_id] for query_id in QUERY_IDS]

    bm25 = BM25Store(chunks)
    bm25.save(str(BM25_PATH))
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    results = []
    for case in test_cases:
        focused_query = retrieval_query(case)
        vector_hits = search_chroma(collection, focused_query, chunks_by_id)
        bm25_hits = bm25.search(focused_query, top_k=10)
        fused = fuse(vector_hits, bm25_hits, top_k=3)
        top_hit = fused[0] if fused else None
        expected_page = int(case["expected_page"])
        page_matches = bool(
            top_hit and top_hit.chunk.page_start <= expected_page <= top_hit.chunk.page_end
        )
        contract_matches = bool(top_hit and top_hit.chunk.contract_id == case["expected_contract_id"])
        contains_expected_term = bool(
            top_hit
            and any(term.lower() in top_hit.chunk.text.lower() for term in case["expected_contains"])
        )
        passed = bool(contract_matches and page_matches and contains_expected_term)
        answer = "Không có trong tài liệu."
        if top_hit is not None:
            answer = f"{snippet_for_query(top_hit, case['expected_contains'])} {top_hit.chunk.citation}"
        results.append(
            {
                "query_id": case["query_id"],
                "query": case["query"],
                "retrieval_query": focused_query,
                "expected_contract_id": case["expected_contract_id"],
                "expected_page": case["expected_page"],
                "expected_contains": case["expected_contains"],
                "top_citation": top_hit.chunk.citation if top_hit else None,
                "top_score": top_hit.score if top_hit else 0.0,
                "contract_matches": contract_matches,
                "page_matches": page_matches,
                "contains_expected_term": contains_expected_term,
                "passed": passed,
                "answer": answer,
                "context": format_context(fused),
            }
        )
    return results


def write_chunks(chunks: list[Chunk]) -> None:
    CHUNKS_PATH.write_text(
        json.dumps(
            [
                {
                    "id": chunk.id,
                    "contract_id": chunk.contract_id,
                    "clause_number": chunk.clause_number,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "clause_type": chunk.clause_type,
                    "citation": chunk.citation,
                    "char_count": len(chunk.text),
                }
                for chunk in chunks
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_markdown(statuses: list[str], chunks: list[Chunk], query_results: list[dict[str, Any]]) -> None:
    passed = sum(1 for result in query_results if result["passed"])
    failed = len(query_results) - passed
    real_ocr = any("ran PaddleOCR-VL" in status for status in statuses)
    remaining_gaps = [
        "- Structured extraction is deterministic smoke extraction, not Claude Haiku JSON extraction.",
        "- Citations are clause/page citations from deterministic chunks; Claude answer synthesis is still pending.",
    ]
    if not real_ocr:
        remaining_gaps.insert(
            0,
            "- Scanned image OCR is simulated with prepared reference text because PaddleOCR-VL did not complete successfully.",
        )
    lines = [
        "# Slice 1 Smoke Test Results",
        "",
        "## Commands",
        "- `.venv/bin/python scripts/slice1_smoke_test.py`",
        "- `python3 -m unittest discover -s tests`",
        "- `python3 -m compileall ingestion indexing retrieval generation ui scripts tests`",
        "",
        "## Inputs",
        f"- PDF contract: `{PDF_CONTRACT_ID}`",
        f"- Scanned simulated contract: `{SCAN_CONTRACT_ID}`",
        f"- Queries: `{', '.join(QUERY_IDS)}` from `data/ground_truth/test_cases.json`",
        "",
        "## Dependency Mode",
        f"- PyMuPDF available: `{has_module('fitz')}`",
        f"- ChromaDB available: `{has_module('chromadb')}`",
        f"- rank_bm25 available: `{has_module('rank_bm25')}`",
        f"- PaddleOCR available: `{has_module('paddleocr')}`",
        "- Claude API call: `not used`; answers are extractive snippets from retrieved chunks for smoke-test determinism.",
        "",
        "## Summary",
        f"- Query checks passed: `{passed}/{len(query_results)}`",
        f"- Query checks failed: `{failed}`",
        f"- Overall status: `{'PASS' if failed == 0 else 'FAIL'}`",
        "",
        "## Pipeline Status",
    ]
    lines.extend(f"- {status}" for status in statuses)
    lines.extend(
        [
            f"- Processed page text: `{PAGE_TEXT_PATH.relative_to(ROOT)}`",
            f"- Chunks JSON: `{CHUNKS_PATH.relative_to(ROOT)}` ({len(chunks)} chunks)",
            f"- Structured JSON: `{STRUCTURED_PATH.relative_to(ROOT)}`",
            f"- SQLite DB: `{SQLITE_PATH.relative_to(ROOT)}` with counts {sqlite_counts()}",
            f"- BM25 index pickle: `{BM25_PATH.relative_to(ROOT)}`",
            "",
            "## Query Results",
        ]
    )
    for result in query_results:
        lines.extend(
            [
                f"### {result['query_id']}",
                "",
                f"- Expected contract: `{result['expected_contract_id']}`",
                f"- Expected page: `{result['expected_page']}`",
                f"- Retrieval query: `{result['retrieval_query']}`",
                f"- Top citation: `{result['top_citation']}`",
                f"- Contract matches: `{result['contract_matches']}`",
                f"- Page matches: `{result['page_matches']}`",
                f"- Contains expected term: `{result['contains_expected_term']}`",
                f"- Status: `{'PASS' if result['passed'] else 'FAIL'}`",
                "",
                "**Answer**",
                "",
                result["answer"],
                "",
            ]
        )
    lines.extend(
        [
            "## Remaining Gaps",
            *remaining_gaps,
            "",
        ]
    )
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    SMOKE_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    statuses: list[str] = []
    manifest = load_manifest()

    pdf = pdf_blocks(PDF_CONTRACT_ID, manifest, statuses)
    scanned = scanned_blocks(SCAN_CONTRACT_ID, manifest, statuses)
    write_page_text(pdf, scanned)

    chunks = chunk_blocks(pdf, contract_id=PDF_CONTRACT_ID)
    write_chunks(chunks)

    structured = extract_structured_fields(PDF_CONTRACT_ID, manifest, chunks)
    persist_structured(structured)

    collection = build_chroma(chunks, statuses)
    query_results = run_queries(chunks, collection)
    write_markdown(statuses, chunks, query_results)
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")
    if not all(result["passed"] for result in query_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
