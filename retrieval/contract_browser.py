from __future__ import annotations

import sqlite3
from typing import Any


def _rows(db_path: str, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def list_contracts(
    db_path: str,
    party_query: str | None = None,
    expiry_before: str | None = None,
) -> list[dict[str, Any]]:
    filters = []
    params: list[Any] = []
    if party_query:
        filters.append("EXISTS (SELECT 1 FROM parties p WHERE p.contract_id = c.id AND lower(p.name) LIKE ?)")
        params.append(f"%{party_query.lower()}%")
    if expiry_before:
        filters.append("c.expiry_date IS NOT NULL AND c.expiry_date <= ?")
        params.append(expiry_before)

    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    return _rows(
        db_path,
        f"""
        SELECT
            c.id,
            c.title,
            c.value,
            c.currency,
            c.effective_date,
            c.expiry_date,
            c.governing_law,
            COALESCE(group_concat(p.name, ', '), '') AS parties
        FROM contracts c
        LEFT JOIN parties p ON p.contract_id = c.id
        {where}
        GROUP BY c.id, c.title, c.value, c.currency, c.effective_date, c.expiry_date, c.governing_law
        ORDER BY c.id
        """,
        tuple(params),
    )


def list_parties(db_path: str, contract_id: str) -> list[dict[str, Any]]:
    return _rows(
        db_path,
        """
        SELECT name, role
        FROM parties
        WHERE contract_id = ?
        ORDER BY name
        """,
        (contract_id,),
    )


def list_clause_types(db_path: str, contract_id: str | None = None) -> list[str]:
    params: tuple[Any, ...] = ()
    where = ""
    if contract_id:
        where = "WHERE contract_id = ?"
        params = (contract_id,)
    rows = _rows(db_path, f"SELECT DISTINCT type FROM clauses {where} ORDER BY type", params)
    return [row["type"] for row in rows if row["type"]]


def list_clauses(
    db_path: str,
    contract_id: str,
    clause_type: str | None = None,
) -> list[dict[str, Any]]:
    filters = ["contract_id = ?"]
    params: list[Any] = [contract_id]
    if clause_type:
        filters.append("type = ?")
        params.append(clause_type)
    return _rows(
        db_path,
        f"""
        SELECT number, type, page, summary
        FROM clauses
        WHERE {' AND '.join(filters)}
        ORDER BY page, number
        """,
        tuple(params),
    )
