import sqlite3
import tempfile
import unittest
from pathlib import Path

from indexing.bm25_store import BM25Store
from ingestion.chunker import Chunk
from retrieval.query_engine import run_contract_query


class SliceTwoQueryEngineTests(unittest.TestCase):
    def test_structured_query_returns_sql_rows_when_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "contracts.sqlite"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE contracts (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        value REAL,
                        currency TEXT,
                        effective_date DATE,
                        expiry_date DATE,
                        governing_law TEXT
                    );
                    INSERT INTO contracts
                        (id, title, value, currency, effective_date, expiry_date, governing_law)
                    VALUES
                        ('HD-1', 'Hợp đồng A', 100.0, 'VND', '2024-01-01', date('now', '+10 days'), 'Việt Nam');
                    """
                )

            result = run_contract_query("Hợp đồng nào sắp hết hạn trong 30 ngày?", [], db_path=str(db_path))

        self.assertEqual(result.intent, "structured")
        self.assertEqual(result.structured_kind, "expiring_contracts")
        self.assertEqual(len(result.structured_rows), 1)
        self.assertEqual(result.structured_rows[0]["id"], "HD-1")
        self.assertEqual(result.hits, [])

    def test_semantic_query_returns_ranked_clause_hits(self):
        chunks = [
            Chunk(
                id="c1",
                text="Payment shall be made net 30 days.",
                contract_id="HD-1",
                clause_number="Payment",
                page_start=2,
                page_end=2,
            ),
            Chunk(
                id="c2",
                text="Customer shall not solicit employees during the term.",
                contract_id="HD-1",
                clause_number="NON-SOLICITATION",
                page_start=4,
                page_end=4,
            ),
        ]

        result = run_contract_query("Explain the non-solicitation obligation", chunks)

        self.assertEqual(result.intent, "semantic")
        self.assertEqual(result.structured_rows, [])
        self.assertEqual(result.hits[0].chunk.clause_number, "NON-SOLICITATION")

    def test_unsupported_structured_query_falls_back_to_clause_search(self):
        chunks = [
            Chunk(
                id="c1",
                text="The Effective Date is February 5, 2020.",
                contract_id="HD-1",
                clause_number="Document",
                page_start=1,
                page_end=1,
            )
        ]

        result = run_contract_query("Effective Date. The date when the contract is effective", chunks, db_path="missing.sqlite")

        self.assertEqual(result.intent, "structured")
        self.assertEqual(result.structured_kind, "unsupported")
        self.assertEqual(result.hits[0].chunk.citation, "[Document, trang 1, HD-1]")

    def test_missing_sqlite_for_supported_structured_query_does_not_crash(self):
        result = run_contract_query("Hợp đồng nào sắp hết hạn trong 30 ngày?", [], db_path="missing.sqlite")

        self.assertEqual(result.intent, "structured")
        self.assertEqual(result.structured_kind, "unsupported")
        self.assertIn("no such table", result.structured_error.lower())

    def test_search_uses_persisted_bm25_chunks_when_session_chunks_are_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bm25_path = Path(tmpdir) / "bm25_chunks.pkl"
            chunks = [
                Chunk(
                    id="c1",
                    text="Customer shall not solicit employees during the term.",
                    contract_id="HD-1",
                    clause_number="NON-SOLICITATION",
                    page_start=4,
                    page_end=4,
                )
            ]
            BM25Store(chunks).save(str(bm25_path))

            result = run_contract_query(
                "Explain the non-solicitation obligation",
                [],
                bm25_path=str(bm25_path),
            )

        self.assertEqual(result.intent, "semantic")
        self.assertEqual(result.hits[0].chunk.citation, "[NON-SOLICITATION, trang 4, HD-1]")


if __name__ == "__main__":
    unittest.main()
