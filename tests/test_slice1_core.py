import os
import sqlite3
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class FakePage:
    def __init__(self, text, tables=None):
        self._text = text
        self._tables = tables or []

    def get_text(self, mode="text"):
        return self._text

    def find_tables(self):
        return SimpleNamespace(tables=self._tables)


class FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def extract(self):
        return self._rows


class SliceOneCoreTests(unittest.TestCase):
    def test_routes_pdf_by_average_text_per_page(self):
        from ingestion.router import route

        with patch("ingestion.router.fitz.open", return_value=[FakePage("x" * 150), FakePage("y" * 120)]):
            self.assertEqual(route("text.pdf"), "text")

        with patch("ingestion.router.fitz.open", return_value=[FakePage("short"), FakePage("")]):
            self.assertEqual(route("scan.pdf"), "scanned")

    def test_parser_returns_page_blocks_and_markdown_tables(self):
        from ingestion.parser import parse_pdf

        table = FakeTable([["Item", "Price"], ["Service", "500000000 VND"]])
        pages = [FakePage("Article 1\nPayment terms", tables=[table])]

        with patch("ingestion.parser.fitz.open", return_value=pages):
            blocks = parse_pdf("contract.pdf")

        self.assertEqual(blocks[0]["page"], 1)
        self.assertEqual(blocks[0]["type"], "text")
        self.assertIn("Payment terms", blocks[0]["text"])
        self.assertEqual(blocks[1]["type"], "table")
        self.assertIn("| Item | Price |", blocks[1]["text"])
        self.assertIn("| Service | 500000000 VND |", blocks[1]["text"])

    def test_clause_chunker_preserves_clause_and_page_citations(self):
        from ingestion.chunker import chunk_blocks

        blocks = [
            {"text": "Điều 1. Phạm vi\nBên A cung cấp dịch vụ.", "page": 1, "type": "text"},
            {"text": "Bổ sung nội dung điều một.", "page": 2, "type": "text"},
            {"text": "Article 2. Payment\nPayment within 30 days.", "page": 3, "type": "text"},
        ]

        chunks = chunk_blocks(blocks, contract_id="HD-2024-001", max_tokens=1000)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].contract_id, "HD-2024-001")
        self.assertEqual(chunks[0].clause_number, "Điều 1")
        self.assertEqual(chunks[0].page_start, 1)
        self.assertEqual(chunks[0].page_end, 2)
        self.assertEqual(chunks[1].clause_number, "Article 2")
        self.assertEqual(chunks[1].page_start, 3)

    def test_clause_chunker_splits_cuad_uppercase_headings(self):
        from ingestion.chunker import chunk_blocks

        blocks = [
            {
                "text": (
                    "EXHIBIT B\n"
                    "NATURE OF ENGAGEMENT: Customer retained Kubient.\n"
                    "SERVICES: Kubient shall provide advertising consultation.\n"
                    "REVENUE SHARE/FEE: The Parties shall share revenue."
                ),
                "page": 1,
                "type": "text",
            },
            {
                "text": (
                    "SCHEDULE 1 TO EXHIBIT B\n"
                    "Monthly Revenue\n"
                    "ACKNOWLEDGEMENT: Customer acknowledges the calculations."
                ),
                "page": 2,
                "type": "text",
            },
        ]

        chunks = chunk_blocks(blocks, contract_id="contract_005", max_tokens=1000)
        numbers = [chunk.clause_number for chunk in chunks]

        self.assertIn("NATURE OF ENGAGEMENT", numbers)
        self.assertIn("SERVICES", numbers)
        self.assertIn("REVENUE SHARE/FEE", numbers)
        self.assertIn("SCHEDULE 1 TO EXHIBIT B", numbers)
        self.assertIn("ACKNOWLEDGEMENT", numbers)
        services = next(chunk for chunk in chunks if chunk.clause_number == "SERVICES")
        self.assertEqual(services.page_start, 1)
        self.assertEqual(services.page_end, 1)
        acknowledgement = next(chunk for chunk in chunks if chunk.clause_number == "ACKNOWLEDGEMENT")
        self.assertEqual(acknowledgement.page_start, 2)
        self.assertEqual(acknowledgement.citation, "[ACKNOWLEDGEMENT, trang 2, contract_005]")
        schedule = next(chunk for chunk in chunks if chunk.clause_number == "SCHEDULE 1 TO EXHIBIT B")
        self.assertEqual(schedule.page_start, 2)
        self.assertEqual(schedule.page_end, 2)

    def test_hybrid_search_fuses_vector_and_keyword_hits_with_citations(self):
        from ingestion.chunker import Chunk
        from retrieval.hybrid_search import ScoredChunk, fuse

        first = Chunk(
            id="c1",
            text="Payment within 30 days",
            contract_id="HD-2024-001",
            clause_number="Điều 5",
            page_start=3,
            page_end=3,
            clause_type="payment_terms",
        )
        second = Chunk(
            id="c2",
            text="Penalty is 0.05 percent per day",
            contract_id="HD-2024-001",
            clause_number="Điều 9",
            page_start=7,
            page_end=7,
            clause_type="penalty",
        )

        results = fuse(
            vector_hits=[ScoredChunk(chunk=first, score=0.9), ScoredChunk(chunk=second, score=0.8)],
            bm25_hits=[ScoredChunk(chunk=second, score=3.0), ScoredChunk(chunk=first, score=1.0)],
        )

        self.assertEqual(results[0].chunk.id, "c1")
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(results[0].chunk.citation, "[Điều 5, trang 3, HD-2024-001]")

    def test_prompt_context_requires_citations_and_fallback(self):
        from generation.prompts import build_answer_messages, format_context
        from ingestion.chunker import Chunk
        from retrieval.hybrid_search import ScoredChunk

        chunk = Chunk(
            id="c1",
            text="Bên A thanh toán trong vòng 30 ngày.",
            contract_id="HD-2024-001",
            clause_number="Điều 5",
            page_start=3,
            page_end=3,
            clause_type="payment_terms",
        )
        hit = ScoredChunk(chunk=chunk, score=1.0)

        context = format_context([hit])
        self.assertIn("[Điều 5, trang 3, HD-2024-001]", context)
        self.assertIn("Bên A thanh toán", context)

        messages = build_answer_messages("Khi nào thanh toán?", [hit])
        self.assertIn("Không có trong tài liệu", messages[0]["content"])
        self.assertIn("Khi nào thanh toán?", messages[1]["content"])

    def test_sql_store_persists_contract_parties_and_clauses(self):
        from indexing.sql_store import ContractRecord, SQLStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "contracts.sqlite")
            store = SQLStore(db_path)
            store.initialize()
            store.upsert_contract(
                ContractRecord(
                    contract_id="HD-2024-001",
                    title="Hợp đồng cung cấp dịch vụ IT",
                    value=500000000,
                    currency="VND",
                    effective_date="2024-01-01",
                    expiry_date="2025-01-01",
                    governing_law="Việt Nam",
                    parties=[{"name": "Công ty A", "role": "bên_a"}],
                    clauses=[{"number": "Điều 5", "type": "payment_terms", "page": 3, "summary": "Thanh toán 30 ngày"}],
                )
            )

            with sqlite3.connect(db_path) as conn:
                contract = conn.execute("SELECT title, value FROM contracts WHERE id = ?", ("HD-2024-001",)).fetchone()
                party = conn.execute("SELECT name, role FROM parties WHERE contract_id = ?", ("HD-2024-001",)).fetchone()
                clause = conn.execute("SELECT number, page FROM clauses WHERE contract_id = ?", ("HD-2024-001",)).fetchone()

        self.assertEqual(contract, ("Hợp đồng cung cấp dịch vụ IT", 500000000.0))
        self.assertEqual(party, ("Công ty A", "bên_a"))
        self.assertEqual(clause, ("Điều 5", 3))

    def test_ocr_runner_parses_image_folder_with_page_numbers(self):
        from ingestion.ocr import PaddleOCRVLRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            first = os.path.join(tmpdir, "page_001.png")
            second = os.path.join(tmpdir, "page_002.png")
            noisy = os.path.join(tmpdir, "page_001_noisy.png")
            open(first, "wb").close()
            open(second, "wb").close()
            open(noisy, "wb").close()

            runner = PaddleOCRVLRunner()
            runner._pipeline = SimpleNamespace(
                predict=lambda path: [{"text": f"text for {os.path.basename(path)}"}]
            )

            blocks = runner.parse_image_folder(tmpdir)

        self.assertEqual([block["page"] for block in blocks], [1, 2])
        self.assertEqual([block["type"] for block in blocks], ["ocr_image", "ocr_image"])
        self.assertIn("page_001.png", blocks[0]["text"])
        self.assertIn("page_002.png", blocks[1]["text"])

    def test_ocr_runner_preserves_paddleocrvl_kwargs(self):
        from ingestion.ocr import PaddleOCRVLRunner

        captured = {}

        class FakePaddleOCRVL:
            def __init__(self, pipeline_version="v1.5", **kwargs):
                captured["pipeline_version"] = pipeline_version
                captured["kwargs"] = kwargs

            def predict(self, _path):
                return []

        fake_module = SimpleNamespace(PaddleOCRVL=FakePaddleOCRVL)
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            runner = PaddleOCRVLRunner(
                vl_rec_model_name="PaddleOCR-VL-0.9B",
                vl_rec_model_dir="/models/PaddleOCR-VL",
                layout_detection_model_name="PP-DocLayoutV2",
                layout_detection_model_dir="/models/PP-DocLayoutV2",
                device="gpu:0",
                vl_rec_model_kwargs={"attn_implementation": "flash_attention_2"},
            )
            runner.parse("contract.pdf")

        self.assertEqual(captured["pipeline_version"], "v1.5")
        self.assertEqual(captured["kwargs"]["vl_rec_model_name"], "PaddleOCR-VL-0.9B")
        self.assertEqual(captured["kwargs"]["vl_rec_model_dir"], "/models/PaddleOCR-VL")
        self.assertEqual(captured["kwargs"]["layout_detection_model_name"], "PP-DocLayoutV2")
        self.assertEqual(captured["kwargs"]["layout_detection_model_dir"], "/models/PP-DocLayoutV2")
        self.assertEqual(captured["kwargs"]["device"], "gpu:0")
        self.assertEqual(captured["kwargs"]["vl_rec_model_kwargs"], {"attn_implementation": "flash_attention_2"})


if __name__ == "__main__":
    unittest.main()
