from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3
from typing import Any


@dataclass(frozen=True)
class StructuredQuery:
    kind: str
    sql: str
    params: tuple[Any, ...] = ()


class UnsupportedStructuredQuery(ValueError):
    pass


def _expiry_days(query: str) -> int:
    match = re.search(r"(\d+)\s*(?:ngày|day|days)", query.lower())
    return int(match.group(1)) if match else 30


def _party_name(query: str) -> str | None:
    match = re.search(r"(?:với|with)\s+(.+?)(?:\?|$)", query, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def build_structured_query(query: str) -> StructuredQuery:
    lowered = query.lower()
    if any(token in lowered for token in ["hết hạn", "expiry", "expiration"]):
        days = _expiry_days(query)
        return StructuredQuery(
            kind="expiring_contracts",
            sql=(
                "SELECT id, title, expiry_date "
                "FROM contracts "
                "WHERE expiry_date IS NOT NULL "
                "AND expiry_date <= date('now', ?) "
                "ORDER BY expiry_date ASC"
            ),
            params=(f"+{days} days",),
        )
    if any(token in lowered for token in ["tổng giá trị", "sum", "total value"]):
        party = _party_name(query)
        if not party:
            raise UnsupportedStructuredQuery("Missing party name for value aggregation query")
        return StructuredQuery(
            kind="party_contract_value",
            sql=(
                "SELECT SUM(c.value) AS total_value, c.currency "
                "FROM contracts c "
                "JOIN parties p ON p.contract_id = c.id "
                "WHERE p.name LIKE ? "
                "GROUP BY c.currency"
            ),
            params=(f"%{party}%",),
        )
    raise UnsupportedStructuredQuery(f"Unsupported structured query: {query}")


def execute_structured_query(db_path: str, query: str) -> list[dict[str, Any]]:
    structured = build_structured_query(query)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(structured.sql, structured.params).fetchall()
    return [dict(row) for row in rows]
