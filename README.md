# Digital Contract Hub

> OnPoint AI Innovation Engineer — Fresher/Intern Assignment, Problem 2.
> A retrieval-augmented contract search prototype that turns scanned and digital contracts into a searchable, citation-grounded knowledge hub for Legal, Finance, and Operations teams.

---

## 1. The problem this solves

OnPoint has a growing pile of hard-copy contracts spread across departments. Today, finding a specific clause, checking a renewal date, or comparing terms across vendors is a manual process: someone walks to a cabinet, flips through PDFs, ctrl-Fs through OCR'd text, and copies the answer into an email. It is slow, error-prone, and does not scale as the contract base grows.

The **Digital Contract Hub** turns every contract — whether a clean PDF or a photo of a signed paper copy — into a structured, searchable record. Users ask questions in natural language and get answers backed by **exact clause + page citations** from the original document.

## 2. Who uses it

| Team | Typical question |
|---|---|
| **Legal** | "What is the governing law and termination notice on the AP Master Services Agreement?" |
| **Finance** | "Which contracts with vendor X expire in the next 60 days, and what is their total value?" |
| **Operations** | "Show me every contract that has a non-solicitation clause." |

All three personas hit the same Streamlit UI; the answers always carry a citation pointing back to a specific page and clause so legal review is trivially auditable.

## 3. What the tool does

1. **Ingests** a contract — PDF (text-native or scanned) or photographed images.
2. **Routes** text PDFs to PyMuPDF and scanned/photo PDFs to PaddleOCR-VL 1.5, with a vision-LLM fallback if the local OCR fails.
3. **Chunks** the document along clause boundaries (`Article 5`, `Điều 8.2`, `NON-SOLICITATION:`, …) so each chunk maps to a clean clause + page citation.
4. **Indexes** chunks into three stores: a vector index (semantic), a BM25 index (keyword), and a SQLite table of structured fields (parties, dates, value, governing law).
5. **Retrieves** with a hybrid pipeline — intent router → semantic + BM25 → RRF fusion → optional cross-encoder rerank.
6. **Answers** the user with an LLM (Gemini / OpenAI / Claude) that is **strictly forbidden from answering without a citation**. If no relevant clause is found, the system says so honestly instead of hallucinating.

## 4. Why this is better than manual search

| Manual today | Contract Hub |
|---|---|
| Walk to cabinet, find paper file. | Upload once, search forever. |
| Open scanned PDFs in a viewer, eyeball OCR. | Clause-aware OCR + chunking, automatic citations. |
| Re-read 20 pages to find a renewal date. | Type "expiry date" → top-3 clauses + page numbers. |
| Side-by-side comparison done by hand. | Multi-contract retrieval over a shared index. |
| Easy to misquote a clause in an email. | Every answer carries `[clause, page, contract_id]`. |

## 5. System architecture

```
        ┌──────────────────────── INGESTION ────────────────────────┐
        │                                                           │
        │   PDF / image                                             │
        │       ↓                                                   │
        │   [Router]  ── text PDF ── → PyMuPDF parser               │
        │       └──── scanned PDF ─→ PaddleOCR-VL 1.5               │
        │                                  │                        │
        │                                  ├─ fallback → Gemini/    │
        │                                  │             OpenAI/    │
        │                                  │             Claude     │
        │                                  ↓                        │
        │                       Clause-aware chunker                │
        │                          /        |         \             │
        │                  Vector idx   BM25 idx    SQLite          │
        │                  (semantic)  (keyword)  (structured)      │
        └───────────────────────────────────────────────────────────┘
                                       ↓
        ┌──────────────────────── RETRIEVAL ────────────────────────┐
        │                                                           │
        │   User query → [Intent router]                            │
        │                   ↓        ↓         ↓                    │
        │              semantic  structured  keyword                │
        │                   └────────┴───────────┘                  │
        │                            ↓                              │
        │                   RRF fusion (k=60)                       │
        │                            ↓                              │
        │              (optional) cross-encoder rerank              │
        │                            ↓                              │
        │              LLM answer with mandatory citation           │
        │                            ↓                              │
        │           Streamlit chat ── citation badges ── CSV export │
        └───────────────────────────────────────────────────────────┘
```

Repository layout:

```
.
├── ingestion/     router, PyMuPDF parser, PaddleOCR-VL wrapper, chunker, extractor
├── indexing/      ChromaDB / BM25 / SQLite stores
├── retrieval/     hybrid search, RRF fusion, reranker, intent router, text-to-SQL
├── generation/    prompts + Gemini/OpenAI/Anthropic answer call with citation enforcement
├── ui/            Streamlit app (upload, chat, browse, export)
├── eval/          precision@3 + citation evaluator
├── scripts/       smoke tests, full-corpus indexer
└── data/          CUAD sample contracts + ground-truth test cases
```

## 6. Demo flow

A reviewer should be able to do this in under 5 minutes:

1. Open the Streamlit UI.
2. Drop a contract (PDF or image) into the sidebar uploader and click **Ingest**.
3. Watch the status panel report `Ingested N chunks`.
4. Type a question in the chat box — e.g. *"What is the agreement date?"*
5. Read the cited answer; click the citation badge to see the underlying clause text.
6. Export search hits to CSV from the same row.

See [§ 16 "How to review this submission in 5 minutes"](#16-how-to-review-this-submission-in-5-minutes) for the exact commands.

## 7. How to run locally

```bash
# 1. Clone & enter the repo
git clone <repo-url> contract-hub && cd contract-hub

# 2. Create a virtualenv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# → open .env and fill in GEMINI_API_KEY (or OPENAI_/ANTHROPIC_API_KEY)

# 4. Launch the UI
streamlit run ui/app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

## 8. Required environment variables

See [`.env.example`](.env.example) for the canonical list. The minimum needed to run the demo:

| Variable | Purpose | Demo default |
|---|---|---|
| `LLM_PROVIDER` | Which LLM answers queries and OCR fallback. | `gemini` |
| `GEMINI_API_KEY` | API key for Gemini. | *(set yours)* |
| `GEMINI_MODEL` | Chat model. **Verified-working free-tier model is** `gemini-2.5-flash-lite`. | `gemini-2.5-flash-lite` |
| `GEMINI_OCR_MODEL` | Vision model used as OCR fallback. | `gemini-2.5-flash-lite` |
| `OCR_DEVICE` | `gpu:0` if CUDA is present, else `cpu`. | `gpu:0` |
| `OCR_FALLBACK_ENABLED` | If `true`, retry failed pages via the vision LLM. | `true` |
| `OCR_API_FALLBACK_PROVIDER` | Which provider serves the OCR fallback. | `gemini` |

Optional alternatives — set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`, or `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`, to swap in a different model.

> **Note on Gemini models & quota.** During submission prep we observed that `gemini-2.0-flash-lite` was returning `429 quota=0` on the free tier and that legacy IDs such as `gemini-1.5-flash` / `gemini-2.0-flash` now respond with *"API key expired"*. Using `gemini-2.5-flash-lite` (or `gemini-2.5-flash`) avoided both. The `.env.example` already points at the working model.
>
> The Gemini free tier on `gemini-2.5-flash-lite` is also rate-limited to **20 requests/day**, which is enough for the 3-case LLM eval but will exhaust quickly during interactive demos. If a reviewer sees a `429 RESOURCE_EXHAUSTED`, the deterministic extractive fallback in `ui/app.py` and `eval/evaluate.py` continues to function and the headline retrieval-only eval (`.venv/bin/python -m eval.evaluate --limit 5`) does not require any LLM call at all.

## 9. How to ingest a contract

In the UI:

1. Open **Upload Contract** in the left sidebar.
2. Drop a PDF or one-or-more page images (`png`, `jpg`, `jpeg`, `tif`, `bmp`, `webp`).
3. Set a **Contract ID** (e.g. `HD-2024-001`). Defaults to `HD-2024-001` and auto-increments on collisions.
4. Click **Ingest**. The status panel logs each step:
   - `<id>: detected text PDF` or `<id>: detected scanned PDF` (router decision)
   - `<id>: <N> contract image(s)` (image uploads)
   - `Ingested <N> chunks` on success.
5. The contract appears in the **Ingested contracts** list. Each row has a remove button.

From the command line (bulk re-index the bundled CUAD sample contracts into the same SQLite + BM25 stores the UI uses):

```bash
.venv/bin/python scripts/build_full_corpus_index.py
```

## 10. How to ask questions

Use the chat input at the bottom of the main panel. Three pre-baked example chips on the empty-state screen show the supported question styles:

- *"What is the agreement date?"* — semantic
- *"Which contracts expire soon?"* — structured (routes through SQLite text-to-SQL)
- *"What are the payment terms?"* — semantic

The intent router (`retrieval/router.py`) routes by keyword:
- mentions of `date`, `value`, `sum`, `expiry`, `expiration`, `ngày`, `giá trị`, `tổng`, `hết hạn` → **structured** (SQL on `contracts` / `parties` / `clauses` tables).
- mentions of `Điều X` / `Article X` / `Section X` or a contract ID like `HD-…` → **keyword**.
- everything else → **semantic** (BM25 + vector + RRF).

## 11. How citations work

Every chunk created by `ingestion/chunker.py` carries:

```python
Chunk(
    id=..., text=..., contract_id="HD-2024-001",
    clause_number="Điều 5",  # or "NON-SOLICITATION", "Article 8.2", "Document"
    page_start=3, page_end=3, clause_type="payment_terms",
)
```

…and exposes a `citation` property:

```
[Điều 5, trang 3, HD-2024-001]
```

The system prompt (`generation/prompts.py`) **forbids the LLM from answering without quoting one of those citations**. If retrieval returns zero hits, the answer pipeline short-circuits to a deterministic `"Không có trong tài liệu."` rather than letting the model hallucinate. If the LLM call itself fails (rate limit, network), the UI falls back to an extractive answer that quotes the top BM25 hit verbatim, with the same citation format.

In the UI, each answer renders:
- a chip-style citation badge per supporting chunk;
- a click-to-expand block showing the raw clause text;
- a CSV export of the top-k retrieved chunks.

## 12. How evaluation works

`eval/evaluate.py` runs a known-answer query set against `contract_004` + `contract_005` from the CUAD sample. For each case it computes:

- **Precision@3** — does any of the top-3 retrieved chunks match the expected `contract_id` + `page` + an expected keyword?
- **Citation accuracy** — is the matched chunk's `[clause, page, contract_id]` citation correct?
- **Answer-contains-expected** — does the rendered answer (extractive or LLM) include one of the expected substrings?
- **Answer faithfulness** — *planned, not implemented*; would use an LLM-as-judge to check that every claim is grounded in the cited chunk.

Run it:

```bash
# Deterministic retrieval-only eval (no LLM calls)
.venv/bin/python -m eval.evaluate --limit 5

# Adds Gemini answer synthesis on top
.venv/bin/python -m eval.evaluate --use-llm-answer --limit 3
```

Outputs are written to `outputs/slice3_eval_results.{md,json}` and the script exits non-zero if `Precision@3 < 0.90` or `Citation accuracy < 0.90`.

## 13. Current evaluation result

> ⚠️ **This is a smoke evaluation on 5 hand-curated cases, not a production-grade benchmark.** The numbers below say the *retrieval pipeline is wired correctly end-to-end on the demo corpus*; they should **not** be read as "this system gets 100% on all contracts everywhere".

Run on commit `955e78f` against `contract_004` + `contract_005`:

| Metric | Result | Assignment target |
|---|---|---|
| Precision@3 (retrieval-only, 5 cases) | **1.000** (5/5) | > 0.90 |
| Citation accuracy (retrieval-only, 5 cases) | **1.000** (5/5) | implicit — every answer must cite |
| Answer-contains-expected with LLM synthesis (Gemini `2.5-flash-lite`, 3 cases) | **1.000** (3/3) | — |
| Answer faithfulness (LLM-as-judge) | *not run* | proposed metric |

This prototype validates the retrieval and citation path end-to-end. OCR field accuracy and clause-level recall need a labeled extraction benchmark before they should be reported as headline metrics — see [§ 14 Limitations](#14-limitations).

Full per-case breakdown: [`outputs/slice3_eval_results.md`](outputs/slice3_eval_results.md).

## 14. Limitations

These are the validation boundaries for this prototype:

- **OCR field accuracy needs a labeled extraction benchmark.** PaddleOCR-VL 1.5 is used for scanned inputs and the demo corpus processes cleanly, but the current evaluation focuses on retrieval/citation correctness rather than reporting a page-level *>99% parties/dates/amounts* score.
- **Clause-extraction recall needs clause-level labeling.** The chunker preserves clause/page citations and supports English + Vietnamese heading patterns; a clause-by-clause recall benchmark is the next validation step before claiming the *>85%* target as a measured result.
- **The smoke eval is 5 cases.** Precision@3 = 1.00 is informative, not statistically meaningful.
- **No statistical significance testing**, no held-out test set, no train/val split — this is a 2–3 day prototype.
- **Vietnamese support is implemented but lightly tested.** The chunker has Vietnamese regex (`Điều`) and the citation format renders `trang N` in Vietnamese, but the bundled eval is English (CUAD).
- **The intent router is a keyword classifier, not learned.** "Which contracts expire soon?" is matched only when the query literally contains `expir`-prefix tokens listed in `retrieval/router.py`.
- **The text-to-SQL planner is rule-based**, handles two query shapes (`expiring_contracts`, `party_contract_value`), and runs against a SQLite snapshot of the demo CUAD corpus — not the contract you just uploaded in this session.
- **The reranker is optional and skipped** if `sentence-transformers` is not installed, so the demo's headline numbers are produced without it.
- **No PII redaction, no role-based access control, no audit log.** Real deployments would need all three.
- **Session state.** Streamlit holds uploaded chunks in memory — restart = lose your session-only ingestions. Persisting them to the SQLite / BM25 stores is a one-line UI addition (future work).
- **The `gemini-2.0-flash-lite` model that earlier docs referenced no longer has free-tier quota on test accounts.** We switched the recommendation to `gemini-2.5-flash-lite`. The deterministic extractive fallback keeps the system usable even if LLM quota fails.

## 15. Cost & scalability considerations

| Layer | Today | At 10× scale | At production |
|---|---|---|---|
| OCR | PaddleOCR-VL on local 6 GB GPU; LLM vision fallback. | Same; batch-ingest off-hours. | Queue + autoscaled GPU workers; pre-process orientation/skew. |
| Vector store | Chroma (file-backed) / BM25 pickle. | Same on a single machine. | Migrate to pgvector or Qdrant + Postgres for the structured side. |
| Structured store | SQLite. | Still SQLite (~10k contracts). | Postgres with proper indexes on `parties.name`, `contracts.expiry_date`. |
| LLM calls | Gemini `2.5-flash-lite` free-tier. | Paid tier (~$0.07 / 1M input tokens at time of writing). | Cache popular Q&A; cheap model for retrieval re-write, premium for synthesis. |
| UI | Streamlit single-user. | Streamlit Cloud / docker container. | Replace with a real auth'd frontend; Streamlit was chosen for POC velocity. |
| Footprint | One process, ~6 GB VRAM during OCR. | Containerise; idle VRAM=0. | Decouple OCR from search service. |

The hot cost path is OCR + LLM tokens. Routing text-native PDFs around OCR (already implemented in `ingestion/router.py`) eliminates the dominant cost for clean digital contracts. The answer prompt is ~1–2k input tokens per query, which is cheap on Gemini Flash but worth caching for repeat questions.

## 16. How to review this submission in 5 minutes

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# → open .env and paste your GEMINI_API_KEY (the recommended model
#   is gemini-2.5-flash-lite; .env.example already sets it).

# 3. Run the UI
streamlit run ui/app.py
# In the sidebar: upload data/raw/cuad_sample/contract_005.pdf
# (Contract ID: HD-2024-001), click Ingest, then ask:
#   "What is the agreement date?"
# Expected answer cites [Document, trang 1, HD-2024-001] and quotes
# "Effective Date: February 5, 2020".

# 4. Run the headline evaluation
.venv/bin/python -m eval.evaluate --limit 5
# Should print Precision@3 ~ 1.000 and Citation accuracy ~ 1.000,
# writing outputs/slice3_eval_results.md.

# 5. Run with LLM answer synthesis (uses your Gemini key)
.venv/bin/python -m eval.evaluate --use-llm-answer --limit 3

# 6. Where to find evidence
ls outputs/
#   slice3_eval_results.md   ← headline retrieval eval
#   slice1_smoke_test_results.md
#   slice2_multi_contract_smoke_test_results.md
#   submission_summary.md    ← email-pasteable summary
```

The full submission walkthrough — selected problem, architecture, trade-offs — lives in [`SUBMISSION_NOTES.md`](SUBMISSION_NOTES.md).

## 17. What I would improve with more time

Ranked by impact:

1. **Labeled OCR + clause-extraction benchmark.** Add page-level labels for parties/dates/amounts and clause-level labels for recall, then wire those metrics into `eval/evaluate.py`. ~½ day to label 5 contracts page-by-page, ~½ day to implement the metrics.
2. **LLM-as-judge answer faithfulness.** The `summary.answer_faithfulness` slot is already in the metrics dataclass; just needs a judge prompt + a small rubric. ~2 hours.
3. **Persist UI ingestions to the SQLite + BM25 stores** so a Streamlit restart does not wipe state. ~2 hours.
4. **Auto-extract structured fields via LLM** (parties, value, currency, effective_date, expiry_date) at ingest time. The schema and `ingestion/extractor.py` are already there; just needs the LLM call wired in and validated.
5. **Smarter intent router** — replace the keyword classifier with a small LLM call (already supported by the provider abstraction). Better handles paraphrases ("when does it expire?" should match the `expiring_contracts` SQL path).
6. **Gemini fallback chain** — `GEMINI_FALLBACK_MODELS` is documented in `.env.example` but not yet auto-applied; on a 429, the answer step should walk the chain before returning the extractive fallback.
7. **Neo4j / NetworkX graph layer** for multi-hop queries ("which vendors appear in more than one master agreement?") — sketched in `docs/CONTRACT_HUB_PLAN.md`.
8. **Per-contract upload directory cleanup** + audit log; today the demo just overwrites by `contract_id`.

## 18. Agentic coding journey

This submission was built in a tight loop with **Claude Code as a paired coding agent**. A few notes that may be useful to a reviewer who cares about the process, not just the artifact:

- **Vertical-slice plan first.** Before writing code, I had Claude draft `docs/CONTRACT_HUB_PLAN.md` — a vibe-coding plan that broke the assignment into Slice 1 (single contract, citations) → Slice 2 (multi-contract, intent router) → Slice 3 (eval + polish). Each slice was demoable on its own, which kept the project from collapsing under its own scope.
- **Tests before refactor.** Whenever the assistant touched ingestion or chunking logic, it wrote/updated unit tests in `tests/test_slice*.py` *first*, then made the change. Today there are 13 test files covering router, chunker, BM25, SQL store, OCR runner kwargs, query engine, text-to-SQL, UI helpers, env loading, and the evaluation script.
- **Systematic debugging during submission prep.** While writing this README, the UI was returning *"Không có trong tài liệu."* after every upload. Claude Code, using its `systematic-debugging` skill, refused to guess a fix until the data flow had been traced end-to-end — and pinned the root cause to a Gemini free-tier quota = 0 on `gemini-2.0-flash-lite`. The fix was a single `.env` line (`GEMINI_MODEL=gemini-2.5-flash-lite`), not a code change. That investigation is worth more than the fix.
- **Honest README pass.** The final README was written under the explicit constraint *"do not overclaim — separate assignment targets from current POC results"*. Every metric in [§ 13](#13-current-evaluation-result) has a sample size next to it, and validation boundaries are called out in [§ 14 Limitations](#14-limitations).
- **What the AI did not do.** Architectural choices (clause-aware chunking, hybrid RRF, mandatory citation prompt, three-store index) came from a reading of the assignment, not the model. The model was excellent at the *plumbing* — wiring stores together, writing tests, normalising prediction shapes from PaddleOCR-VL — and at *refusing to fake* test results when the underlying eval was small.

---

**License & data.** The CUAD sample contracts under `data/raw/cuad_sample/` are from the [Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad) (CC BY 4.0). All other code is original to this submission.
