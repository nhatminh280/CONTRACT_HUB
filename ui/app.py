from __future__ import annotations

import os
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.answer import answer_with_citations
from indexing.bm25_store import BM25Store
from ingestion.chunker import Chunk, chunk_blocks
from ingestion.ocr import PaddleOCRVLRunner
from ingestion.parser import parse_pdf
from ingestion.router import route
from retrieval.hybrid_search import ScoredChunk
from retrieval.reranker import rerank


DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


def _load_demo_hits(query: str, chunks: list[Chunk]) -> list[ScoredChunk]:
    bm25 = BM25Store(chunks)
    hits = bm25.search(query, top_k=10)
    return rerank(query, hits, top_k=3)


st.set_page_config(page_title="Digital Contract Hub", layout="wide")
st.title("Digital Contract Hub")

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

upload_tab, search_tab = st.tabs(["Upload PDF", "Search"])

with upload_tab:
    uploaded = st.file_uploader("Contract PDF", type=["pdf"])
    contract_id = st.text_input("Contract ID", value="HD-2024-001")
    if uploaded and st.button("Ingest"):
        target = DATA_RAW / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        with st.status("Ingesting contract", expanded=True) as status:
            pdf_kind = route(str(target))
            st.write(f"Detected: {pdf_kind}")
            if pdf_kind == "text":
                blocks = parse_pdf(str(target))
            else:
                ocr = PaddleOCRVLRunner()
                blocks = ocr.parse(str(target))
                ocr.unload()
            chunks = chunk_blocks(blocks, contract_id=contract_id)
            st.session_state["chunks"] = chunks
            status.update(label=f"Ingested {len(chunks)} chunks", state="complete")
        st.dataframe(
            [{"citation": chunk.citation, "type": chunk.clause_type, "text": chunk.text[:240]} for chunk in st.session_state["chunks"]],
            use_container_width=True,
        )

with search_tab:
    query = st.text_input("Search contracts")
    if query:
        chunks = st.session_state.get("chunks", [])
        if not chunks:
            st.warning("Upload and ingest a contract first.")
        else:
            hits = _load_demo_hits(query, chunks)
            with st.spinner("Generating cited answer"):
                try:
                    answer = answer_with_citations(query, hits, api_key=os.getenv("ANTHROPIC_API_KEY"))
                except Exception:
                    answer = "\n\n".join(f"{hit.chunk.citation}\n{hit.chunk.text}" for hit in hits)
            st.markdown(answer)
            st.divider()
            for hit in hits:
                with st.expander(hit.chunk.citation):
                    st.write(hit.chunk.text)
