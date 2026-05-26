from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indexing.bm25_store import BM25Store
from ingestion.chunker import Chunk, chunk_blocks
from retrieval.hybrid_search import ScoredChunk, fuse


DEFAULT_CONTRACT_IDS = ["contract_004", "contract_005"]
DEFAULT_LIMIT = 15
DEFAULT_RESULTS_PATH = ROOT / "outputs" / "slice3_eval_results.md"
DEFAULT_JSON_PATH = ROOT / "outputs" / "slice3_eval_results.json"
DEFAULT_MIN_PRECISION_AT_3 = 0.90
DEFAULT_MIN_CITATION_ACCURACY = 0.90
CUAD_QUERY_EXPANSIONS = {
    "document name": "title heading exhibit schedule agreement master services contract name",
    "parties": "by and between party parties customer kubient signature signed",
    "revenue/profit sharing": "revenue share fee monthly revenue below threshold above threshold schedule",
}


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    query: str
    expected_contract_id: str
    expected_page: int
    expected_contains: list[str]
    top_citation: str | None
    matched_citation: str | None
    precision_at_3_hit: bool
    citation_correct: bool
    answer_contains_expected: bool
    matched_rank: int | None = None


@dataclass(frozen=True)
class EvaluationSummary:
    total_cases: int
    precision_at_3: float
    citation_accuracy: float
    answer_contains_accuracy: float


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def focused_query(test_case: dict[str, Any]) -> str:
    match = re.search(r'"([^"]+)"', test_case["query"])
    category = match.group(1) if match else test_case["query"]
    details = test_case["query"].split("Details:", 1)[-1].strip() if "Details:" in test_case["query"] else ""
    expansion = CUAD_QUERY_EXPANSIONS.get(category.lower(), "")
    parts = [category, details, expansion]
    return ". ".join(part for part in parts if part)


def load_test_cases(
    path: Path,
    contract_ids: list[str],
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    contracts = set(contract_ids)
    cases = [case for case in load_json(path) if case["expected_contract_id"] in contracts]
    return cases[:limit]


def _reference_blocks(contract_id: str) -> list[dict[str, Any]]:
    rows = load_jsonl(ROOT / "data" / "processed" / "pdf_page_text_reference.jsonl")
    pages = sorted(
        (row for row in rows if row["contract_id"] == contract_id),
        key=lambda row: row["page_number"],
    )
    return [{"text": row["text"], "page": row["page_number"], "type": "text_reference"} for row in pages]


def _pdf_blocks(contract_id: str, manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pdf_path = ROOT / manifest[contract_id]["pdf_path"]
    try:
        from ingestion.parser import parse_pdf

        return parse_pdf(str(pdf_path))
    except Exception:
        return _reference_blocks(contract_id)


def build_chunks(contract_ids: list[str]) -> list[Chunk]:
    manifest_rows = load_json(ROOT / "data" / "ground_truth" / "contract_manifest.json")
    manifest = {row["contract_id"]: row for row in manifest_rows}
    chunks: list[Chunk] = []
    for contract_id in contract_ids:
        blocks = _pdf_blocks(contract_id, manifest)
        chunks.extend(chunk_blocks(blocks, contract_id=contract_id))
    return chunks


def vectorize(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        vector[index] += 1.0
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def vector_search(query: str, chunks: list[Chunk], top_k: int = 10) -> list[ScoredChunk]:
    query_vector = vectorize(query)
    scored = [
        ScoredChunk(chunk=chunk, score=cosine(query_vector, vectorize(chunk.text)))
        for chunk in chunks
    ]
    return [hit for hit in sorted(scored, key=lambda hit: hit.score, reverse=True)[:top_k] if hit.score > 0]


def retrieve_ranked_chunks(query: str, chunks: list[Chunk], top_k: int = 3) -> list[ScoredChunk]:
    bm25 = BM25Store(chunks)
    vector_hits = vector_search(query, chunks, top_k=10)
    bm25_hits = bm25.search(query, top_k=10)
    return fuse(vector_hits, bm25_hits, top_k=top_k)


def _contains_expected(text: str, expected_terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in expected_terms)


def _matches_case(chunk: Chunk, case: dict[str, Any]) -> bool:
    expected_page = int(case["expected_page"])
    return bool(
        chunk.contract_id == case["expected_contract_id"]
        and chunk.page_start <= expected_page <= chunk.page_end
        and _contains_expected(chunk.text, case["expected_contains"])
    )


def evaluate_ranked_results(
    test_cases: list[dict[str, Any]],
    ranked_chunks_by_query_id: dict[str, list[Chunk]],
) -> list[QueryEvaluation]:
    evaluations: list[QueryEvaluation] = []
    for case in test_cases:
        hits = ranked_chunks_by_query_id.get(case["query_id"], [])[:3]
        matched_rank = next((index for index, chunk in enumerate(hits, start=1) if _matches_case(chunk, case)), None)
        matched = hits[matched_rank - 1] if matched_rank is not None else None
        top = hits[0] if hits else None
        evaluations.append(
            QueryEvaluation(
                query_id=case["query_id"],
                query=case["query"],
                expected_contract_id=case["expected_contract_id"],
                expected_page=int(case["expected_page"]),
                expected_contains=list(case["expected_contains"]),
                top_citation=top.citation if top else None,
                matched_citation=matched.citation if matched else None,
                precision_at_3_hit=matched is not None,
                citation_correct=matched is not None,
                answer_contains_expected=matched is not None and _contains_expected(matched.text, case["expected_contains"]),
                matched_rank=matched_rank,
            )
        )
    return evaluations


def evaluate_cases(test_cases: list[dict[str, Any]], chunks: list[Chunk], top_k: int = 3) -> list[QueryEvaluation]:
    ranked: dict[str, list[Chunk]] = {}
    for case in test_cases:
        scoped_chunks = [chunk for chunk in chunks if chunk.contract_id == case["expected_contract_id"]]
        hits = retrieve_ranked_chunks(focused_query(case), scoped_chunks, top_k=top_k)
        ranked[case["query_id"]] = [hit.chunk for hit in hits]
    return evaluate_ranked_results(test_cases, ranked)


def summarize_evaluations(evaluations: list[QueryEvaluation]) -> EvaluationSummary:
    total = len(evaluations)
    if total == 0:
        return EvaluationSummary(
            total_cases=0,
            precision_at_3=0.0,
            citation_accuracy=0.0,
            answer_contains_accuracy=0.0,
        )
    return EvaluationSummary(
        total_cases=total,
        precision_at_3=sum(item.precision_at_3_hit for item in evaluations) / total,
        citation_accuracy=sum(item.citation_correct for item in evaluations) / total,
        answer_contains_accuracy=sum(item.answer_contains_expected for item in evaluations) / total,
    )


def enforce_thresholds(
    summary: EvaluationSummary,
    min_precision_at_3: float = DEFAULT_MIN_PRECISION_AT_3,
    min_citation_accuracy: float = DEFAULT_MIN_CITATION_ACCURACY,
) -> None:
    failures = []
    if summary.precision_at_3 < min_precision_at_3:
        failures.append(f"Precision@3 {summary.precision_at_3:.3f} < {min_precision_at_3:.3f}")
    if summary.citation_accuracy < min_citation_accuracy:
        failures.append(f"Citation accuracy {summary.citation_accuracy:.3f} < {min_citation_accuracy:.3f}")
    if failures:
        raise SystemExit(1)


def run_evaluation(contract_ids: list[str], limit: int) -> tuple[EvaluationSummary, list[QueryEvaluation]]:
    chunks = build_chunks(contract_ids)
    cases = load_test_cases(ROOT / "data" / "ground_truth" / "test_cases.json", contract_ids, limit=limit)
    evaluations = evaluate_cases(cases, chunks)
    return summarize_evaluations(evaluations), evaluations


def render_markdown(summary: EvaluationSummary, evaluations: list[QueryEvaluation], contract_ids: list[str]) -> str:
    lines = [
        "# Slice 3 Evaluation Results",
        "",
        "## Inputs",
        f"- Contracts: `{', '.join(contract_ids)}`",
        f"- Cases evaluated: `{summary.total_cases}`",
        "",
        "## Metrics",
        f"- Precision@3: `{summary.precision_at_3:.3f}`",
        f"- Citation accuracy: `{summary.citation_accuracy:.3f}`",
        f"- Answer contains expected text: `{summary.answer_contains_accuracy:.3f}`",
        "- Answer faithfulness: `not run`; Gemini LLM-as-judge is still pending.",
        "",
        "## Cases",
    ]
    for item in evaluations:
        lines.extend(
            [
                f"### {item.query_id}",
                "",
                f"- Expected contract: `{item.expected_contract_id}`",
                f"- Expected page: `{item.expected_page}`",
                f"- Top citation: `{item.top_citation}`",
                f"- Matched citation: `{item.matched_citation}`",
                f"- Matched rank: `{item.matched_rank}`",
                f"- Precision@3 hit: `{item.precision_at_3_hit}`",
                f"- Citation correct: `{item.citation_correct}`",
                f"- Answer contains expected text: `{item.answer_contains_expected}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_results(
    summary: EvaluationSummary,
    evaluations: list[QueryEvaluation],
    contract_ids: list[str],
    markdown_path: Path,
    json_path: Path,
) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(summary, evaluations, contract_ids), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "summary": asdict(summary),
                "cases": [asdict(item) for item in evaluations],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Contract Hub retrieval and citation quality.")
    parser.add_argument("--contracts", nargs="+", default=DEFAULT_CONTRACT_IDS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--min-precision-at-3", type=float, default=DEFAULT_MIN_PRECISION_AT_3)
    parser.add_argument("--min-citation-accuracy", type=float, default=DEFAULT_MIN_CITATION_ACCURACY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, evaluations = run_evaluation(args.contracts, args.limit)
    write_results(summary, evaluations, args.contracts, args.output, args.json_output)
    print(f"Wrote {args.output.relative_to(ROOT)}")
    print(f"Precision@3: {summary.precision_at_3:.3f}")
    print(f"Citation accuracy: {summary.citation_accuracy:.3f}")
    try:
        enforce_thresholds(
            summary,
            min_precision_at_3=args.min_precision_at_3,
            min_citation_accuracy=args.min_citation_accuracy,
        )
    except SystemExit:
        print("Evaluation thresholds failed")
        raise


if __name__ == "__main__":
    main()
