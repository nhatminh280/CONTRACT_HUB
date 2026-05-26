import sqlite3
import tempfile
import unittest
from pathlib import Path

from retrieval.text_to_sql import UnsupportedStructuredQuery, build_structured_query, execute_structured_query


class SliceTwoTextToSqlTests(unittest.TestCase):
    def test_builds_expiry_window_query(self):
        structured = build_structured_query("Hợp đồng nào sắp hết hạn trong 30 ngày?")

        self.assertIn("expiry_date <= date('now', ?)", structured.sql)
        self.assertEqual(structured.params, ("+30 days",))
        self.assertEqual(structured.kind, "expiring_contracts")

    def test_builds_party_value_sum_query(self):
        structured = build_structured_query("Tổng giá trị hợp đồng với Công ty A?")

        self.assertIn("SUM(c.value)", structured.sql)
        self.assertEqual(structured.params, ("%Công ty A%",))
        self.assertEqual(structured.kind, "party_contract_value")

    def test_rejects_unsupported_structured_query(self):
        with self.assertRaises(UnsupportedStructuredQuery):
            build_structured_query("Có bao nhiêu điều khoản bảo mật?")

    def test_executes_structured_query_against_sqlite(self):
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
                    CREATE TABLE parties (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        contract_id TEXT,
                        name TEXT,
                        role TEXT
                    );
                    INSERT INTO contracts
                        (id, title, value, currency, effective_date, expiry_date, governing_law)
                    VALUES
                        ('HD-1', 'Hợp đồng A', 100.0, 'VND', '2024-01-01', date('now', '+10 days'), 'Việt Nam'),
                        ('HD-2', 'Hợp đồng B', 200.0, 'VND', '2024-01-01', date('now', '+60 days'), 'Việt Nam');
                    INSERT INTO parties (contract_id, name, role)
                    VALUES ('HD-1', 'Công ty A', 'vendor'), ('HD-2', 'Công ty B', 'vendor');
                    """
                )

            rows = execute_structured_query(str(db_path), "Hợp đồng nào sắp hết hạn trong 30 ngày?")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "HD-1")
        self.assertEqual(rows[0]["title"], "Hợp đồng A")


if __name__ == "__main__":
    unittest.main()
