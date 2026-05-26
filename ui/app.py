from __future__ import annotations

import os
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import load_env_file
from generation.answer import answer_with_citations
from ingestion.chunker import Chunk, chunk_blocks
from ingestion.ocr import PaddleOCRVLRunner
from ingestion.parser import parse_pdf
from ingestion.router import route
from retrieval.contract_browser import list_clause_types, list_clauses, list_contracts, list_parties
from retrieval.query_engine import run_contract_query
from ui.export import rows_to_csv


DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)
load_env_file(ROOT / ".env")


st.set_page_config(page_title="Digital Contract Hub", layout="wide")
st.title("Digital Contract Hub")

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

sqlite_path = st.sidebar.text_input(
    "SQLite DB path",
    value=str(ROOT / "outputs" / "slice2_multi_contract_index" / "contracts.sqlite"),
)

upload_tab, search_tab, browser_tab = st.tabs(["Upload PDF", "Search", "Contracts"])

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
        result = run_contract_query(query, chunks, db_path=sqlite_path)
        st.caption(f"Intent: {result.intent}")
        if result.structured_rows:
            st.dataframe(result.structured_rows, use_container_width=True)
            csv_text = rows_to_csv(result.structured_rows)
            if csv_text:
                st.download_button(
                    "Export structured results CSV",
                    data=csv_text,
                    file_name="structured_search_results.csv",
                    mime="text/csv",
                )
        elif not chunks:
            st.warning("Upload and ingest a contract first, or use a structured query backed by SQLite.")
        else:
            with st.spinner("Generating cited answer"):
                try:
                    answer = answer_with_citations(query, result.hits, api_key=os.getenv("OPENAI_API_KEY"))
                except Exception:
                    answer = "\n\n".join(f"{hit.chunk.citation}\n{hit.chunk.text}" for hit in result.hits)
            st.markdown(answer)
            st.divider()
            hit_rows = [
                {
                    "citation": hit.chunk.citation,
                    "score": hit.score,
                    "contract_id": hit.chunk.contract_id,
                    "clause_number": hit.chunk.clause_number,
                    "page_start": hit.chunk.page_start,
                    "page_end": hit.chunk.page_end,
                    "text": hit.chunk.text,
                }
                for hit in result.hits
            ]
            csv_text = rows_to_csv(hit_rows)
            if csv_text:
                st.download_button(
                    "Export search results CSV",
                    data=csv_text,
                    file_name="search_results.csv",
                    mime="text/csv",
                )
            for hit in result.hits:
                with st.expander(hit.chunk.citation):
                    st.write(hit.chunk.text)
        if result.structured_error:
            st.caption(result.structured_error)

with browser_tab:
    col_party, col_expiry = st.columns([2, 1])
    with col_party:
        party_query = st.text_input("Party filter")
    with col_expiry:
        expiry_before = st.date_input("Expiry before", value=None)

    try:
        contracts = list_contracts(
            sqlite_path,
            party_query=party_query or None,
            expiry_before=expiry_before.isoformat() if expiry_before else None,
        )
    except Exception as exc:
        st.warning(f"Could not load contracts: {exc}")
        contracts = []

    st.dataframe(contracts, use_container_width=True)
    csv_text = rows_to_csv(contracts)
    if csv_text:
        st.download_button(
            "Export contracts CSV",
            data=csv_text,
            file_name="contracts.csv",
            mime="text/csv",
        )
    contract_ids = [row["id"] for row in contracts]
    if contract_ids:
        selected_contract = st.selectbox("Contract", contract_ids)
        parties = list_parties(sqlite_path, selected_contract)
        if parties:
            st.dataframe(parties, use_container_width=True)
            csv_text = rows_to_csv(parties)
            if csv_text:
                st.download_button(
                    "Export parties CSV",
                    data=csv_text,
                    file_name=f"{selected_contract}_parties.csv",
                    mime="text/csv",
                )
        clause_types = list_clause_types(sqlite_path, selected_contract)
        selected_type = st.selectbox("Clause type", ["All", *clause_types])
        clauses = list_clauses(
            sqlite_path,
            selected_contract,
            clause_type=None if selected_type == "All" else selected_type,
        )
        st.dataframe(clauses, use_container_width=True)
        csv_text = rows_to_csv(clauses)
        if csv_text:
            st.download_button(
                "Export clauses CSV",
                data=csv_text,
                file_name=f"{selected_contract}_clauses.csv",
                mime="text/csv",
            )
