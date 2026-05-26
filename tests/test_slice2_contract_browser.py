import sqlite3
import tempfile
import unittest
from pathlib import Path

from retrieval.contract_browser import list_clauses, list_contracts, list_parties


class SliceTwoContractBrowserTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "contracts.sqlite"
        with sqlite3.connect(self.db_path) as conn:
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
                CREATE TABLE clauses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT,
                    number TEXT,
                    type TEXT,
                    page INTEGER,
                    summary TEXT
                );
                INSERT INTO contracts
                    (id, title, value, currency, effective_date, expiry_date, governing_law)
                VALUES
                    ('HD-1', 'Supply Agreement', 100.0, 'USD', '2024-01-01', '2024-12-31', 'New York'),
                    ('HD-2', 'Services Agreement', 200.0, 'USD', '2024-02-01', '2025-02-01', 'Delaware');
                INSERT INTO parties (contract_id, name, role)
                VALUES
                    ('HD-1', 'Acme Corp', 'buyer'),
                    ('HD-2', 'Beta LLC', 'customer');
                INSERT INTO clauses (contract_id, number, type, page, summary)
                VALUES
                    ('HD-1', '1', 'payment_terms', 2, 'Pay within 30 days'),
                    ('HD-1', '2', 'termination', 4, 'Either party may terminate'),
                    ('HD-2', '1', 'general', 1, 'Services scope');
                """
            )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_lists_contracts_with_party_names(self):
        rows = list_contracts(str(self.db_path))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "HD-1")
        self.assertEqual(rows[0]["parties"], "Acme Corp")

    def test_filters_contracts_by_party_and_expiry(self):
        rows = list_contracts(str(self.db_path), party_query="beta", expiry_before="2025-12-31")

        self.assertEqual([row["id"] for row in rows], ["HD-2"])

    def test_lists_clauses_for_contract_and_type(self):
        rows = list_clauses(str(self.db_path), contract_id="HD-1", clause_type="termination")

        self.assertEqual(rows, [{"number": "2", "type": "termination", "page": 4, "summary": "Either party may terminate"}])

    def test_lists_parties(self):
        rows = list_parties(str(self.db_path), contract_id="HD-1")

        self.assertEqual(rows, [{"name": "Acme Corp", "role": "buyer"}])


if __name__ == "__main__":
    unittest.main()
