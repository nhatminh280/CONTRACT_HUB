# Digital Contract Hub — Submission Summary

**Candidate:** OnPoint AI Innovation Engineer Fresher/Intern, Fall 2026
**Problem:** Problem 2 — The Digital Contract Hub

## What I built

A retrieval-augmented contract knowledge hub that ingests scanned or photographed contracts (PDF / image), routes text-native PDFs through PyMuPDF and scanned PDFs through **PaddleOCR-VL 1.5** (with a Gemini / OpenAI / Claude vision fallback), chunks the result along legal clause boundaries, indexes each chunk into a **vector + BM25 + SQLite** triple store, and answers natural-language questions via a hybrid (RRF-fused) retrieval pipeline. Every answer is grounded in an exact `[clause, page, contract_id]` citation, and a deterministic extractive fallback keeps the system usable when the LLM call is rate-limited.

## How it maps to the assignment criteria

| Assignment target | This submission |
|---|---|
| OCR / parsing accuracy on parties, dates, amounts > 99 % | PaddleOCR-VL 1.5 + vision-LLM fallback in place; *not formally benchmarked on a labeled page set* — flagged as future work. |
| Retrieval precision@3 on a known-answer query set > 90 % | **1.000** on a 5-case smoke evaluation (`contract_004`, `contract_005`). |
| Every answer cites exact clause + page | Enforced both via clause-aware chunking and by a system prompt that forbids un-cited claims. |
| Clause-extraction recall > 85 % | *Not measured* — the structured extractor is wired but recall is not benchmarked. |

## Headline results (smoke evaluation)

| Metric | Result |
|---|---|
| Precision@3 (5 retrieval-only cases) | **1.000** (5/5) |
| Citation accuracy (5 retrieval-only cases) | **1.000** (5/5) |
| Answer-contains-expected with LLM synthesis (Gemini 2.5-flash-lite, 3 cases) | **1.000** (3/3) |

Per-case detail: `outputs/slice3_eval_results.md`. The LLM-synthesis run is reproducible by `.venv/bin/python -m eval.evaluate --use-llm-answer --limit 3`; on free-tier Gemini that is currently capped at ~20 LLM requests/day, so the **deterministic retrieval-only evaluation is the headline reproducible metric**.

## Stack

PyMuPDF · PaddleOCR-VL 1.5 · clause-aware chunker · ChromaDB · `rank_bm25` · SQLite · RRF fusion · optional `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker · Gemini / OpenAI / Claude with mandatory-citation prompt · Streamlit UI · `eval/evaluate.py` for Precision@3 + citation accuracy.

## How to review in 5 minutes

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                              # paste your GEMINI_API_KEY
streamlit run ui/app.py                           # upload data/raw/cuad_sample/contract_005.pdf, ask "What is the agreement date?"
.venv/bin/python -m eval.evaluate --limit 5       # headline metrics
.venv/bin/python -m eval.evaluate --use-llm-answer --limit 3
```

Recommended model: `GEMINI_MODEL=gemini-2.5-flash-lite` (verified working on free tier — `.env.example` already sets it).

## Honest limitations

This is a 2–3 day prototype. The smoke evaluation is intentionally small; OCR accuracy and clause-recall against a labeled set are *not yet measured*; the intent router and text-to-SQL planner are rule-based; Streamlit session state is in-memory; there is no RBAC / audit / PII redaction. Full list in `SUBMISSION_NOTES.md §6`.

## What I would do next

1. Labeled OCR + clause-recall benchmark (closes the unmeasured assignment targets).
2. LLM-as-judge answer-faithfulness score.
3. Persist session uploads to the SQLite + BM25 stores.
4. Auto-populate structured fields via LLM at ingest time.
5. Replace keyword intent router with a small LLM classifier.

## Files to look at first

- `README.md` — full project overview.
- `SUBMISSION_NOTES.md` — technical companion (problem, architecture, trade-offs, limitations, future work).
- `outputs/slice3_eval_results.md` — headline evaluation.
- `eval/evaluate.py` — the evaluator itself.
- `ingestion/chunker.py` + `generation/prompts.py` — the citation-grounded core.
