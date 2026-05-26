from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs"
SMOKE_INDEX_DIR = OUTPUT_DIR / "slice2_multi_contract_index"
RESULTS_PATH = OUTPUT_DIR / "slice2_multi_contract_smoke_test_results.md"
CHUNKS_PATH = OUTPUT_DIR / "slice2_multi_contract_chunks.json"
STRUCTURED_PATH = OUTPUT_DIR / "slice2_multi_contract_structured_fields.json"
SQLITE_PATH = SMOKE_INDEX_DIR / "contracts.sqlite"
BM25_PATH = SMOKE_INDEX_DIR / "bm25_chunks.pkl"
CHROMA_DIR = SMOKE_INDEX_DIR / "chroma"

CONTRACT_IDS = ["contract_004", "contract_005"]
QUERY_IDS = ["q004", "q007", "q013", "q015", "q016"]

from generation.prompts import format_context
from generation.answer import answer_with_citations
from indexing.bm25_store import BM25Store
from indexing.sql_store import ContractRecord, SQLStore
from ingestion.chunker import Chunk
from ingestion.chunker import chunk_blocks
from ingestion.extractor import contract_record_from_llm_json, extract_structured_json, format_chunks_for_llm_extraction
from ingestion.extractor import extract_deterministic_record
from retrieval.hybrid_search import ScoredChunk, fuse
from retrieval.router import classify_intent
from retrieval.text_to_sql import UnsupportedStructuredQuery, build_structured_query, execute_structured_query


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Slice 2 multi-contract smoke test.")
    parser.add_argument("--use-llm-extractor", action="store_true")
    parser.add_argument("--use-llm-answer", action="store_true")
    return parser.parse_args(argv)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def validate_llm_mode(use_llm_extractor: bool, use_llm_answer: bool) -> None:
    if not (use_llm_extractor or use_llm_answer):
        return
    from config.llm import llm_api_key, llm_provider

    provider = llm_provider()
    if not llm_api_key(provider=provider):
        key_name = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(provider, "GEMINI_API_KEY")
        raise SystemExit(f"{key_name} is required when LLM flags are enabled.")
    required_module = "anthropic" if provider == "anthropic" else "openai"
    if not has_module(required_module):
        raise SystemExit(f"The {required_module} package is required when LLM flags are enabled.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_manifest() -> dict[str, dict[str, Any]]:
    with (ROOT / "data" / "ground_truth" / "contract_manifest.json").open(encoding="utf-8") as file:
        rows = json.load(file)
    return {row["contract_id"]: row for row in rows}


def load_reference_pages(contract_id: str) -> list[dict[str, Any]]:
    rows = load_jsonl(ROOT / "data" / "processed" / "pdf_page_text_reference.jsonl")
    pages = [row for row in rows if row["contract_id"] == contract_id]
    return sorted(pages, key=lambda row: row["page_number"])


def load_blocks(contract_id: str, manifest: dict[str, dict[str, Any]], statuses: list[str]) -> list[dict[str, Any]]:
    pdf_path = ROOT / manifest[contract_id]["pdf_path"]
    if has_module("fitz"):
        from ingestion.parser import parse_pdf
        from ingestion.router import route

        kind = route(str(pdf_path))
        blocks = parse_pdf(str(pdf_path))
        statuses.append(
            f"{contract_id}: PDF ingest routed `{pdf_path.relative_to(ROOT)}` as `{kind}` "
            f"with {len(blocks)} parsed blocks."
        )
        return blocks

    pages = load_reference_pages(contract_id)
    statuses.append(
        f"{contract_id}: PyMuPDF unavailable; used prepared page-text reference fallback "
        f"with {len(pages)} pages."
    )
    return [{"text": row["text"], "page": row["page_number"], "type": "text_reference"} for row in pages]


def extract_structured_fields(contract_id: str, chunks: list[Chunk], use_llm: bool = False) -> ContractRecord:
    if use_llm:
        text = format_chunks_for_llm_extraction(chunks)
        payload = extract_structured_json(text)
        return contract_record_from_llm_json(contract_id, payload, chunks)
    return extract_deterministic_record(contract_id, chunks)


def persist_structured(records: list[ContractRecord]) -> None:
    store = SQLStore(str(SQLITE_PATH))
    store.initialize()
    for record in records:
        store.upsert_contract(record)
    STRUCTURED_PATH.write_text(
        json.dumps(
            [
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
            ],
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
    collection = client.get_or_create_collection("slice2_multi_contract_chunks")
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
    statuses.append(f"Chroma index: built collection `slice2_multi_contract_chunks` with {len(chunks)} chunks.")
    return collection


def search_chroma(collection: Any | None, query: str, chunks_by_id: dict[str, Chunk], top_k: int = 10) -> list[ScoredChunk]:
    if collection is None:
        return []
    result = collection.query(query_embeddings=[vectorize(query)], n_results=top_k)
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0] if result.get("distances") else [0.0] * len(ids)
    return [ScoredChunk(chunks_by_id[chunk_id], 1.0 / (1.0 + float(distance))) for chunk_id, distance in zip(ids, distances)]


def load_test_cases() -> list[dict[str, Any]]:
    with (ROOT / "data" / "ground_truth" / "test_cases.json").open(encoding="utf-8") as file:
        cases_by_id = {case["query_id"]: case for case in json.load(file)}
    return [cases_by_id[query_id] for query_id in QUERY_IDS]


def retrieval_query(test_case: dict[str, Any]) -> str:
    match = re.search(r'"([^"]+)"', test_case["query"])
    category = match.group(1) if match else test_case["query"]
    details = test_case["query"].split("Details:", 1)[-1].strip() if "Details:" in test_case["query"] else ""
    return f"{category}. {details}" if details else category


def _ranked_hits(value: Chunk | list[Chunk] | None) -> list[Chunk]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _matches_case(chunk: Chunk, case: dict[str, Any]) -> bool:
    expected_page = int(case["expected_page"])
    return bool(
        chunk.contract_id == case["expected_contract_id"]
        and chunk.page_start <= expected_page <= chunk.page_end
        and any(term.lower() in chunk.text.lower() for term in case["expected_contains"])
    )


def evaluate_query_results(
    cases: list[dict[str, Any]],
    top_hits: dict[str, Chunk | list[Chunk] | None],
) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        ranked_hits = _ranked_hits(top_hits.get(case["query_id"]))
        top_hit = ranked_hits[0] if ranked_hits else None
        matched_hit = next((hit for hit in ranked_hits if _matches_case(hit, case)), None)
        focused_query = retrieval_query(case)
        intent = classify_intent(focused_query)
        expected_page = int(case["expected_page"])
        contract_matches = bool(matched_hit)
        page_matches = bool(matched_hit)
        contains_expected_term = bool(
            matched_hit
            and any(term.lower() in matched_hit.text.lower() for term in case["expected_contains"])
        )
        passed = bool(matched_hit)
        results.append(
            {
                "query_id": case["query_id"],
                "query": case["query"],
                "retrieval_query": focused_query,
                "intent": intent,
                "expected_contract_id": case["expected_contract_id"],
                "expected_page": expected_page,
                "expected_contains": case["expected_contains"],
                "top_citation": top_hit.citation if top_hit else None,
                "matched_citation": matched_hit.citation if matched_hit else None,
                "contract_matches": contract_matches,
                "page_matches": page_matches,
                "contains_expected_term": contains_expected_term,
                "passed": passed,
                "answer": f"{matched_hit.text[:420]} {matched_hit.citation}" if matched_hit else "Không có trong tài liệu.",
            }
        )
    return results


def answer_for_query(query: str, hits: list[ScoredChunk], use_llm_answer: bool = False) -> str:
    if use_llm_answer:
        return answer_with_citations(query, hits)
    if not hits:
        return "Không có trong tài liệu."
    return f"{hits[0].chunk.text[:420]} {hits[0].chunk.citation}"


def run_queries(chunks: list[Chunk], collection: Any | None, use_llm_answer: bool = False) -> list[dict[str, Any]]:
    test_cases = load_test_cases()
    bm25 = BM25Store(chunks)
    bm25.save(str(BM25_PATH))
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    top_hits: dict[str, list[Chunk]] = {}
    fused_hits: dict[str, list[ScoredChunk]] = {}
    contexts: dict[str, str] = {}

    for case in test_cases:
        focused_query = retrieval_query(case)
        vector_hits = search_chroma(collection, focused_query, chunks_by_id)
        bm25_hits = bm25.search(focused_query, top_k=10)
        fused = fuse(vector_hits, bm25_hits, top_k=3)
        fused_hits[case["query_id"]] = fused
        top_hits[case["query_id"]] = [hit.chunk for hit in fused]
        contexts[case["query_id"]] = format_context(fused)

    results = evaluate_query_results(test_cases, top_hits)
    for result in results:
        result["context"] = contexts[result["query_id"]]
        if use_llm_answer:
            result["answer"] = answer_for_query(
                result["query"],
                fused_hits[result["query_id"]],
                use_llm_answer=True,
            )
        if result["intent"] == "structured":
            try:
                structured = build_structured_query(result["retrieval_query"])
                sql_rows = execute_structured_query(str(SQLITE_PATH), result["retrieval_query"])
                result["structured_sql_kind"] = structured.kind
                result["structured_sql"] = structured.sql
                result["structured_sql_row_count"] = len(sql_rows)
            except UnsupportedStructuredQuery as exc:
                result["structured_sql_kind"] = "unsupported"
                result["structured_sql"] = str(exc)
                result["structured_sql_row_count"] = None
        else:
            result["structured_sql_kind"] = None
            result["structured_sql"] = None
            result["structured_sql_row_count"] = None
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


def write_markdown(
    statuses: list[str],
    chunks: list[Chunk],
    query_results: list[dict[str, Any]],
    use_llm_extractor: bool = False,
    use_llm_answer: bool = False,
) -> None:
    passed = sum(1 for result in query_results if result["passed"])
    failed = len(query_results) - passed
    by_contract = {contract_id: sum(1 for chunk in chunks if chunk.contract_id == contract_id) for contract_id in CONTRACT_IDS}
    lines = [
        "# Slice 2 Multi-Contract Smoke Test Results",
        "",
        "## Commands",
        "- `.venv/bin/python scripts/slice2_multi_contract_smoke_test.py`",
        "- `.venv/bin/python -m unittest tests.test_slice2_multi_contract`",
        "- `.venv/bin/python -m unittest discover -s tests`",
        "- `.venv/bin/python -m compileall ingestion indexing retrieval generation ui scripts tests`",
        "",
        "## Inputs",
        f"- Contracts: `{', '.join(CONTRACT_IDS)}`",
        f"- Queries: `{', '.join(QUERY_IDS)}` from `data/ground_truth/test_cases.json`",
        "",
        "## Dependency Mode",
        f"- PyMuPDF available: `{has_module('fitz')}`",
        f"- ChromaDB available: `{has_module('chromadb')}`",
        f"- rank_bm25 available: `{has_module('rank_bm25')}`",
        "- OCR: `not used`; Slice 2 foundation uses text PDFs/reference text for faster multi-contract indexing.",
        f"- LLM extractor: `{'used' if use_llm_extractor else 'not used'}`.",
        f"- LLM answer synthesis: `{'used' if use_llm_answer else 'not used'}`.",
        "- Intent router: deterministic local classifier from `retrieval/router.py`; retrieval still uses hybrid search while SQL is shown as diagnostics.",
        "- Text-to-SQL: deterministic local translator from `retrieval/text_to_sql.py`; report-only until metadata extraction is richer.",
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
            f"- Chunks JSON: `{CHUNKS_PATH.relative_to(ROOT)}` ({len(chunks)} chunks, {by_contract})",
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
                f"- Intent: `{result['intent']}`",
                f"- Structured SQL kind: `{result['structured_sql_kind']}`",
                f"- Structured SQL row count: `{result['structured_sql_row_count']}`",
                f"- Top citation: `{result['top_citation']}`",
                f"- Matched citation: `{result['matched_citation']}`",
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
            "- Query router and text-to-SQL are deterministic and report-only; routing behavior can now be wired into UI/search commands.",
            "- Structured extraction remains deterministic unless `--use-llm-extractor` is enabled.",
            "",
        ]
    )
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def reset_outputs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if SMOKE_INDEX_DIR.exists():
        shutil.rmtree(SMOKE_INDEX_DIR)
    SMOKE_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_llm_mode(args.use_llm_extractor, args.use_llm_answer)
    reset_outputs()
    statuses: list[str] = []
    manifest = load_manifest()
    all_chunks: list[Chunk] = []
    records: list[ContractRecord] = []

    for contract_id in CONTRACT_IDS:
        blocks = load_blocks(contract_id, manifest, statuses)
        chunks = chunk_blocks(blocks, contract_id=contract_id)
        statuses.append(f"{contract_id}: chunked into {len(chunks)} clause-aware chunks.")
        all_chunks.extend(chunks)
        records.append(extract_structured_fields(contract_id, chunks, use_llm=args.use_llm_extractor))

    write_chunks(all_chunks)
    persist_structured(records)
    collection = build_chroma(all_chunks, statuses)
    query_results = run_queries(all_chunks, collection, use_llm_answer=args.use_llm_answer)
    write_markdown(
        statuses,
        all_chunks,
        query_results,
        use_llm_extractor=args.use_llm_extractor,
        use_llm_answer=args.use_llm_answer,
    )
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")
    if not all(result["passed"] for result in query_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
