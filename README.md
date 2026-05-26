# Digital Contract Hub

Digital Contract Hub ingests contract PDFs, extracts clause-aware chunks with page citations, indexes them for hybrid semantic and keyword retrieval, and answers questions using only cited source context.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/app.py
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` before using Gemini-backed features.

## Slice 1 Scope

- Route PDFs as text or scanned based on average extracted text per page.
- Parse text PDFs with PyMuPDF and convert detected tables to markdown.
- Provide a PaddleOCR-VL wrapper for scanned documents.
- Chunk by legal clause/article boundaries and preserve contract, clause, and page metadata.
- Store chunks in ChromaDB and BM25, with contract metadata in SQLite.
- Fuse vector and keyword results with reciprocal rank fusion.
- Rerank top hits with a local cross-encoder when available.
- Generate Gemini prompts that require citations in `[Điều X, trang Y, Hợp đồng Z]` form.
- Provide a Streamlit MVP with upload and search tabs.

## Limitations

- OCR speed depends on local GPU and PaddleOCR-VL availability.
- The local test suite avoids heavyweight model downloads and API calls.
- SQLite is intended for the prototype scale described in Slice 1.
