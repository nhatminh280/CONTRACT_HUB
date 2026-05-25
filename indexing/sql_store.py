from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3
from typing import Any


@dataclass
class ContractRecord:
    contract_id: str
    title: str
    value: float | None = None
    currency: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    governing_law: str | None = None
    parties: list[dict[str, Any]] = field(default_factory=list)
    clauses: list[dict[str, Any]] = field(default_factory=list)


class SQLStore:
    def __init__(self, db_path: str = "data/index/contracts.sqlite") -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS contracts (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    value REAL,
                    currency TEXT,
                    effective_date DATE,
                    expiry_date DATE,
                    governing_law TEXT
                );
                CREATE TABLE IF NOT EXISTS parties (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT,
                    name TEXT,
                    role TEXT
                );
                CREATE TABLE IF NOT EXISTS clauses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id TEXT,
                    number TEXT,
                    type TEXT,
                    page INTEGER,
                    summary TEXT
                );
                """
            )

    def upsert_contract(self, record: ContractRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO contracts (id, title, value, currency, effective_date, expiry_date, governing_law)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    value = excluded.value,
                    currency = excluded.currency,
                    effective_date = excluded.effective_date,
                    expiry_date = excluded.expiry_date,
                    governing_law = excluded.governing_law
                """,
                (
                    record.contract_id,
                    record.title,
                    record.value,
                    record.currency,
                    record.effective_date,
                    record.expiry_date,
                    record.governing_law,
                ),
            )
            conn.execute("DELETE FROM parties WHERE contract_id = ?", (record.contract_id,))
            conn.execute("DELETE FROM clauses WHERE contract_id = ?", (record.contract_id,))
            conn.executemany(
                "INSERT INTO parties (contract_id, name, role) VALUES (?, ?, ?)",
                [(record.contract_id, party.get("name"), party.get("role")) for party in record.parties],
            )
            conn.executemany(
                "INSERT INTO clauses (contract_id, number, type, page, summary) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        record.contract_id,
                        clause.get("number") or clause.get("clause_number"),
                        clause.get("type") or clause.get("clause_type"),
                        clause.get("page"),
                        clause.get("summary"),
                    )
                    for clause in record.clauses
                ],
            )
