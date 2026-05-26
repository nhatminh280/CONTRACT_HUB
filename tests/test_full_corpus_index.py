import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


class FullCorpusIndexTests(unittest.TestCase):
    def test_build_full_corpus_index_persists_all_requested_contracts(self):
        from indexing.bm25_store import BM25Store
        from scripts.build_full_corpus_index import build_full_corpus_index

        def block_loader(contract_id, _manifest):
            return [
                {
                    "text": f"MASTER AGREEMENT {contract_id}\nPayment terms are net 30 days.",
                    "page": 1,
                    "type": "text",
                }
            ]

        manifest = {
            "contract_a": {"pdf_path": "a.pdf"},
            "contract_b": {"pdf_path": "b.pdf"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_full_corpus_index(
                manifest=manifest,
                contract_ids=["contract_a", "contract_b"],
                output_dir=Path(tmpdir),
                block_loader=block_loader,
            )

            with sqlite3.connect(result.sqlite_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
            chunks_payload = json.loads(result.chunks_path.read_text(encoding="utf-8"))
            bm25_chunks = BM25Store.load(str(result.bm25_path)).chunks

        self.assertEqual(count, 2)
        self.assertEqual(result.contract_count, 2)
        self.assertEqual({row["contract_id"] for row in chunks_payload}, {"contract_a", "contract_b"})
        self.assertEqual({chunk.contract_id for chunk in bm25_chunks}, {"contract_a", "contract_b"})


if __name__ == "__main__":
    unittest.main()
