# Slice 1 Smoke Test Results

## Commands
- `.venv/bin/python scripts/slice1_smoke_test.py`
- `python3 -m unittest discover -s tests`
- `python3 -m compileall ingestion indexing retrieval generation ui scripts tests`

## Inputs
- PDF contract: `contract_005`
- Scanned simulated contract: `contract_005`
- Queries: `q003, q004, q007` from `data/ground_truth/test_cases.json`

## Dependency Mode
- PyMuPDF available: `True`
- ChromaDB available: `True`
- rank_bm25 available: `True`
- PaddleOCR available: `True`
- LLM extractor: `not used`.
- LLM answer synthesis: `not used`.

## Summary
- Query checks passed: `3/3`
- Query checks failed: `0`
- Overall status: `PASS`

## Pipeline Status
- PDF ingest: data/raw/cuad_sample/contract_005.pdf routed as `text` with 3 parsed blocks.
- Scanned ingest: ran PaddleOCR-VL on 2 clean rendered images in `data/raw/scanned_simulated/contract_005` and produced 2 OCR blocks.
- Chroma index: built collection `slice1_smoke_chunks` with 10 chunks.
- Processed page text: `outputs/slice1_processed_page_text.jsonl`
- Chunks JSON: `outputs/slice1_chunks.json` (10 chunks)
- Structured JSON: `outputs/slice1_structured_fields.json`
- SQLite DB: `outputs/slice1_smoke_index/contracts.sqlite` with counts {'contracts': 1, 'parties': 0, 'clauses': 10}
- BM25 index pickle: `outputs/slice1_smoke_index/bm25_chunks.pkl`

## Query Results
### q003

- Expected contract: `contract_005`
- Expected page: `1`
- Retrieval query: `Agreement Date`
- Top citation: `[CONFLICTING TERMS, trang 1, contract_005]`
- Contract matches: `True`
- Page matches: `True`
- Contains expected term: `True`
- Status: `PASS`

**Answer**

HEREOF, the parties hereto have caused this Exhibit A to be executed by their duly authorized representatives as of the 26th day of March, 2020. Kubient Signature: By: /s/ Paul Roberts Its: President Date: 3/27/2020 Customer Signature: By: /s/ Ted Mendelsohn Its: VP, Commercial Mkts Date: 3/27/2020 [CONFLICTING TERMS, trang 1, contract_005]

### q004

- Expected contract: `contract_005`
- Expected page: `1`
- Retrieval query: `Effective Date`
- Top citation: `[Document, trang 1, contract_005]`
- Contract matches: `True`
- Page matches: `True`
- Contains expected term: `True`
- Status: `PASS`

**Answer**

EXHIBIT ‘B’ This Exhibit B is entered into as of the 26th day of March 2020 by and between Kubient, Inc. ("Kubient"), and The Associated Press ("Customer"). This Exhibit is hereby incorporated into and made a part of the Master Services Agreement (the "Agreement") between the Parties (Effective Date: February 5, 2020). [Document, trang 1, contract_005]

### q007

- Expected contract: `contract_005`
- Expected page: `1`
- Retrieval query: `No-Solicit Of Employees`
- Top citation: `[NON-SOLICITATION, trang 1, contract_005]`
- Contract matches: `True`
- Page matches: `True`
- Contains expected term: `True`
- Status: `PASS`

**Answer**

NON-SOLICITATION: During the Term and any renewal terms of the Agreement, and for a period of one (1) year following the expiration or earlier termination thereof, Customer shall not, without Kubient's prior written consent, directly or indirectly (i) solicit or encourage any person to leave the employment or other service of Kubient; or (ii) hire, on behalf o [NON-SOLICITATION, trang 1, contract_005]

## Remaining Gaps
- Structured extraction is deterministic unless `--use-llm-extractor` is enabled.
- Answer synthesis is extractive unless `--use-llm-answer` is enabled.
