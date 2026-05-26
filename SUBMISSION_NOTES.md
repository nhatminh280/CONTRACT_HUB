# Submission Notes — Digital Contract Hub

> OnPoint AI Innovation Engineer Fresher / Intern Assignment, Fall 2026.
> This document is the **technical companion to `README.md`** — it summarises the selected problem, what was built, how it was evaluated, what trade-offs were made, and what was knowingly left out.

---

## 1. Selected problem

**Problem 2 — The Digital Contract Hub.**

OnPoint manages a growing volume of hard-copy contracts across Legal, Finance, and operational teams. Finding a specific clause, tracking renewal dates, or comparing terms across vendors is currently manual and unscalable. The system must digitise scanned/photographed contracts and produce structured records that can be searched by party, date, clause type, or free text — with exact citations.

Assignment evaluation criteria (verbatim):
- OCR / parsing accuracy on parties / dates / amounts: **> 99 %**.
- Retrieval precision@3 on a known-answer set: **> 90 %**.
- Source citation: every answer cites exact clause + page.
- Clause-extraction recall for clause type / renewal dates / governing law: **> 85 %**.

## 2. Implemented solution

A clause-aware, citation-grounded RAG system over heterogeneous contract inputs:

- **Ingestion** routes text-native PDFs through PyMuPDF and scanned / photographed PDFs through **PaddleOCR-VL 1.5**, with a vision-LLM API fallback (Gemini / OpenAI / Claude) for any page where local OCR fails or returns empty text.
- **Clause-aware chunking** splits documents along legal boundaries — bilingual (`Điều X`, `Article X`, `Section X`, CUAD-style uppercase headings like `NON-SOLICITATION:`). Each chunk preserves `(contract_id, clause_number, page_start, page_end, clause_type)` so the resulting citation is page- and clause-precise.
- **Three-store index**:
  - **ChromaDB / vector** for semantic search.
  - **BM25** (`rank_bm25`) for keyword and exact-token matches.
  - **SQLite** for structured fields (`contracts`, `parties`, `clauses`).
- **Hybrid retrieval** = intent router → semantic + BM25 → **RRF fusion (k=60)** → optional `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker → top-k.
- **Answer generation** via Gemini / OpenAI / Claude with a system prompt that forbids answers without citations and falls back to a deterministic extractive answer if the LLM call fails (rate limit, network).
- **Streamlit UI** with upload, chat, citation badges, click-to-expand source clauses, and CSV export.
- **Evaluation script** (`eval/evaluate.py`) computes Precision@3, citation accuracy, and answer-contains-expected on a known-answer set; exits non-zero on threshold breach.

## 3. Architecture

```
PDF / image
   ↓
[router]  text → PyMuPDF parser
   └──   scanned → PaddleOCR-VL 1.5 ──(on failure)──→ Gemini / OpenAI / Claude vision
                       ↓
                clause-aware chunker
                       ↓
          ┌──────────┬───────────────┬──────────────┐
          ↓          ↓               ↓              ↓
       Chroma      BM25         SQLite         (extractor.py)
      (semantic) (keyword)    (structured)
                       ↓
              intent router → hybrid_search → RRF fusion → (rerank)
                       ↓
               LLM answer (mandatory citation)
                       ↓
            Streamlit chat + citation badges + CSV
```

Full directory map is in `README.md` §5.

## 4. Evaluation results

> **Smoke evaluation on 5 hand-curated cases against `contract_004` + `contract_005`. Not a production benchmark.**

| Metric | Result | Target |
|---|---|---|
| Precision@3 (retrieval-only, 5 cases) | **1.000** (5/5) | > 0.90 |
| Citation accuracy (retrieval-only, 5 cases) | **1.000** (5/5) | every answer cites clause + page |
| Answer-contains-expected with LLM synthesis (Gemini `2.5-flash-lite`, 3 cases) | **1.000** (3/3) | — |
| OCR accuracy on parties / dates / amounts | *not measured on a labeled set* | > 0.99 |
| Clause-extraction recall | *not measured* | > 0.85 |
| Answer faithfulness (LLM-as-judge) | *not run* | proposed |

Detailed per-case output: `outputs/slice3_eval_results.md` and `outputs/slice3_eval_results.json`.

Sanity-check artifacts from earlier slices: `outputs/slice1_smoke_test_results.md`, `outputs/slice2_multi_contract_smoke_test_results.md`.

## 5. Trade-offs

| Decision | Why we chose it | What we gave up |
|---|---|---|
| PaddleOCR-VL 1.5 + vision-LLM fallback | SOTA quality, runs locally on a 6 GB GPU, free-tier vision fallback covers edge cases. | ~20–30 s/page on a single GPU; not real-time. |
| Clause-aware chunking via regex | Deterministic, debuggable, language-aware (VN + EN + CUAD uppercase). | Misses unusual heading styles; needs a learned splitter for production. |
| Three-store index (Chroma + BM25 + SQLite) | Each store solves a query class it is best at; RRF fuses cheaply. | Three things to maintain; small consistency risk if ingestion is interrupted. |
| RRF fusion, no learned weights | Hyperparameter-free; works out of the box. | Cannot tune for a specific corpus distribution without going to learned-to-rank. |
| Mandatory-citation prompt | Eliminates the most common hallucination class. | Slight over-conservatism: the model occasionally says *"Not found"* when partial evidence exists. |
| Streamlit | Fastest path from "works in a script" to a clickable demo. | Single-user, no auth, ephemeral session state. |
| Rule-based intent router & text-to-SQL | Zero-cost, deterministic, easy to read. | Brittle paraphrase handling — *"when does it expire"* won't match `expir` keyword token search. |
| Deterministic extractive fallback | Demo stays usable when the LLM call fails. | Fallback answers are quotes, not synthesised — readable but less polished. |

## 6. Known limitations

These are documented in `README.md §14` and re-stated here so a reviewer reading only this file does not miss them:

1. **OCR accuracy is not formally measured.** No labeled-page benchmark exists in this repo; the *>99 %* assignment target is aspirational.
2. **Clause-extraction recall is not measured.** Only retrieval-side metrics are evaluated.
3. **The smoke eval is 5 cases.** P@3 = 1.0 is directional, not statistically significant.
4. **Vietnamese is implemented but lightly tested** — chunker has VN regex, but the bundled eval is English (CUAD).
5. **Intent router is keyword-based**, not learned. Robust to exact terms, brittle to paraphrase.
6. **Text-to-SQL handles two query shapes** (`expiring_contracts`, `party_contract_value`) and runs against the pre-built demo SQLite, not your session uploads.
7. **Reranker is optional**; demo numbers are produced without it.
8. **No PII redaction / RBAC / audit log** — would be table-stakes for a real Legal deployment.
9. **Streamlit holds session uploads in memory**; restart clears them. Persisting session ingestions is future work.
10. **Free-tier Gemini caveat.** During submission prep, `gemini-2.0-flash-lite` was returning `429 quota=0`. We pinned `.env.example` to `gemini-2.5-flash-lite`, which works. The extractive fallback keeps the UI usable if any LLM fails.

## 7. Future work

In rough priority order:

1. Labeled OCR + clause-extraction benchmark (the missing assignment-target measurements).
2. LLM-as-judge answer-faithfulness metric (the dataclass slot is already there).
3. Persist session-time UI ingestions to the SQLite + BM25 stores so refresh ≠ data loss.
4. LLM-driven structured extraction (`ingestion/extractor.py`) at ingest time so the SQLite `contracts` row is auto-populated.
5. Replace the keyword intent router with a small LLM classifier.
6. Implement the `GEMINI_FALLBACK_MODELS` chain end-to-end (today it's documented but not auto-applied).
7. NetworkX / Neo4j knowledge graph layer for multi-hop vendor questions.
8. Per-tenant access controls, audit logging, and PII redaction.

## 8. How a reviewer should evaluate this submission

```bash
# 5-minute path (full instructions in README §16):
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                              # paste GEMINI_API_KEY
streamlit run ui/app.py                           # try the UI flow
.venv/bin/python -m eval.evaluate --limit 5       # headline metrics
.venv/bin/python -m eval.evaluate --use-llm-answer --limit 3
```

The final pasteable summary lives at `outputs/submission_summary.md`.
