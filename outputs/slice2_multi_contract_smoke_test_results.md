# Slice 2 Multi-Contract Smoke Test Results

## Commands
- `.venv/bin/python scripts/slice2_multi_contract_smoke_test.py`
- `.venv/bin/python -m unittest tests.test_slice2_multi_contract`
- `.venv/bin/python -m unittest discover -s tests`
- `.venv/bin/python -m compileall ingestion indexing retrieval generation ui scripts tests`

## Inputs
- Contracts: `contract_004, contract_005`
- Queries: `q004, q007, q013, q015, q016` from `data/ground_truth/test_cases.json`

## Dependency Mode
- PyMuPDF available: `True`
- ChromaDB available: `True`
- rank_bm25 available: `True`
- OCR: `not used`; Slice 2 foundation uses text PDFs/reference text for faster multi-contract indexing.
- LLM extractor: `not used`.
- LLM answer synthesis: `not used`.
- Intent router: deterministic local classifier from `retrieval/router.py`; retrieval still uses hybrid search while SQL is shown as diagnostics.
- Text-to-SQL: deterministic local translator from `retrieval/text_to_sql.py`; report-only until metadata extraction is richer.

## Summary
- Query checks passed: `5/5`
- Query checks failed: `0`
- Overall status: `PASS`

## Pipeline Status
- contract_004: PDF ingest routed `data/raw/cuad_sample/contract_004.pdf` as `text` with 5 parsed blocks.
- contract_004: chunked into 6 clause-aware chunks.
- contract_005: PDF ingest routed `data/raw/cuad_sample/contract_005.pdf` as `text` with 3 parsed blocks.
- contract_005: chunked into 10 clause-aware chunks.
- Chroma index: built collection `slice2_multi_contract_chunks` with 16 chunks.
- Chunks JSON: `outputs/slice2_multi_contract_chunks.json` (16 chunks, {'contract_004': 6, 'contract_005': 10})
- Structured JSON: `outputs/slice2_multi_contract_structured_fields.json`
- SQLite DB: `outputs/slice2_multi_contract_index/contracts.sqlite` with counts {'contracts': 2, 'parties': 4, 'clauses': 16}
- BM25 index pickle: `outputs/slice2_multi_contract_index/bm25_chunks.pkl`

## Query Results
### q004

- Expected contract: `contract_005`
- Expected page: `1`
- Retrieval query: `Effective Date. The date when the contract is effective`
- Intent: `structured`
- Structured SQL kind: `unsupported`
- Structured SQL row count: `None`
- Top citation: `[Document, trang 1, contract_005]`
- Matched citation: `[Document, trang 1, contract_005]`
- Contract matches: `True`
- Page matches: `True`
- Contains expected term: `True`
- Status: `PASS`

**Answer**

EXHIBIT ‘B’
This Exhibit B is entered into as of the 26th day of March 2020 by and between Kubient, Inc. ("Kubient"), and The Associated Press ("Customer").
This Exhibit is hereby incorporated into and made a part of the Master Services Agreement (the "Agreement") between the Parties (Effective Date:
February 5, 2020). [Document, trang 1, contract_005]

### q007

- Expected contract: `contract_005`
- Expected page: `1`
- Retrieval query: `No-Solicit Of Employees. Is there a restriction on a party’s soliciting or hiring employees and/or contractors from the  counterparty, whether during the contract or after the contract ends (or both)?`
- Intent: `semantic`
- Structured SQL kind: `None`
- Structured SQL row count: `None`
- Top citation: `[9, trang 3, contract_004]`
- Matched citation: `[NON-SOLICITATION, trang 1, contract_005]`
- Contract matches: `True`
- Page matches: `True`
- Contains expected term: `True`
- Status: `PASS`

**Answer**

NON-SOLICITATION: During the Term and any renewal terms of the Agreement, and for a period of one (1) year following the expiration or earlier
termination thereof, Customer shall not, without Kubient's prior written consent, directly or indirectly (i) solicit or encourage any person to leave
the employment or other service of Kubient; or (ii) hire, on behalf of Customer or any other person or entity, any person who h [NON-SOLICITATION, trang 1, contract_005]

### q013

- Expected contract: `contract_004`
- Expected page: `1`
- Retrieval query: `Expiration Date. On what date will the contract's initial term expire?`
- Intent: `structured`
- Structured SQL kind: `expiring_contracts`
- Structured SQL row count: `1`
- Top citation: `[CONFLICTING TERMS, trang 1, contract_005]`
- Matched citation: `[Document, trang 1-2, contract_004]`
- Contract matches: `True`
- Page matches: `True`
- Contains expected term: `True`
- Status: `PASS`

**Answer**

Exhibit 10.14 MASTER SERVICES AGREEMENT This Master Services Agreement (the "Agreement"), dated as of the 5th day of February, 2020 (the "Effective Date"), is by and between Kubient Inc., with offices located at 330 7th Avenue, 10th Floor, New York, NY 10001 ("Kubient") and The Associated Press, a New York not-for-profit corporation with principal place of business located at 200 Liberty Street, New York, NY 10281 (t [Document, trang 1-2, contract_004]

### q015

- Expected contract: `contract_004`
- Expected page: `4`
- Retrieval query: `Governing Law. Which state/country's law governs the interpretation of the contract?`
- Intent: `semantic`
- Structured SQL kind: `None`
- Structured SQL row count: `None`
- Top citation: `[9, trang 3, contract_004]`
- Matched citation: `[13, trang 4-5, contract_004]`
- Contract matches: `True`
- Page matches: `True`
- Contains expected term: `True`
- Status: `PASS`

**Answer**

13. Disclaimer. Except as expressly set forth herein, the Services are provided on an "as is," "where is," and "as available" basis, and, to the maximum extent permitted by Law, Kubient disclaims, and Customer hereby waives, all representations and warranties, express or implied, arising by operation of Law or otherwise, except for the representations and warranties set forth in this Agreement, including but not limi [13, trang 4-5, contract_004]

### q016

- Expected contract: `contract_004`
- Expected page: `1`
- Retrieval query: `Termination For Convenience. Can a party terminate this  contract without cause (solely by giving a notice and allowing a waiting  period to expire)?`
- Intent: `semantic`
- Structured SQL kind: `None`
- Structured SQL row count: `None`
- Top citation: `[Document, trang 1-2, contract_004]`
- Matched citation: `[Document, trang 1-2, contract_004]`
- Contract matches: `True`
- Page matches: `True`
- Contains expected term: `True`
- Status: `PASS`

**Answer**

Exhibit 10.14 MASTER SERVICES AGREEMENT This Master Services Agreement (the "Agreement"), dated as of the 5th day of February, 2020 (the "Effective Date"), is by and between Kubient Inc., with offices located at 330 7th Avenue, 10th Floor, New York, NY 10001 ("Kubient") and The Associated Press, a New York not-for-profit corporation with principal place of business located at 200 Liberty Street, New York, NY 10281 (t [Document, trang 1-2, contract_004]

## Remaining Gaps
- Query router and text-to-SQL are deterministic and report-only; routing behavior can now be wired into UI/search commands.
- Structured extraction remains deterministic unless `--use-llm-extractor` is enabled.
