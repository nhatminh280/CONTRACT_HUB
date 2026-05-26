from __future__ import annotations

from datetime import date
import json
import re
from typing import Any

from config.llm import llm_api_key, llm_base_url, llm_model, llm_provider
from indexing.sql_store import ContractRecord
from ingestion.chunker import Chunk


EXTRACTION_PROMPT = """Extract contract metadata as strict JSON with:
contract_id, title, parties, effective_date, expiry_date, contract_value,
currency, governing_law, clauses.
Each clause must include clause_number, clause_type, page, summary.
Return one valid JSON object only. Do not use markdown, bullet lists, or code fences."""


def extract_structured_json(text: str, api_key: str | None = None, model: str | None = None) -> dict[str, Any]:
    """Extract structured contract data through the configured OpenAI-compatible API."""
    from openai import OpenAI

    provider = llm_provider()
    if provider == "anthropic":
        from config.anthropic_client import create_anthropic_text

        content = create_anthropic_text(
            system=EXTRACTION_PROMPT,
            user_content=text,
            max_tokens=4000,
            model=model,
        )
        return json.loads(content or "{}")

    client = OpenAI(api_key=llm_api_key(api_key, provider=provider), base_url=llm_base_url(provider=provider))
    response = client.chat.completions.create(
        model=llm_model(model, provider=provider),
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=4000,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content or "{}")


def format_chunks_for_llm_extraction(chunks: list[Chunk], max_chars: int = 16000) -> str:
    parts = []
    for chunk in chunks:
        excerpt = " ".join(chunk.text.split())[:1500]
        page = str(chunk.page_start) if chunk.page_start == chunk.page_end else f"{chunk.page_start}-{chunk.page_end}"
        parts.append(
            "\n".join(
                [
                    f"Contract: {chunk.contract_id}",
                    f"Citation: {chunk.citation}",
                    f"Clause: {chunk.clause_number}",
                    f"Page: {page}",
                    f"Text: {excerpt}",
                ]
            )
        )
    text = "\n\n".join(parts)
    return text[:max_chars]


def _chunk_clauses(chunks: list[Chunk]) -> list[dict[str, Any]]:
    return [
        {
            "number": chunk.clause_number,
            "type": chunk.clause_type,
            "page": chunk.page_start,
            "summary": chunk.text[:240].replace("\n", " "),
        }
        for chunk in chunks
    ]


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_parties(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    parties = []
    for party in value:
        if isinstance(party, str):
            name = party.strip()
            if name:
                parties.append({"name": name, "role": "party"})
            continue
        if isinstance(party, dict):
            name = str(party.get("name") or party.get("party") or "").strip()
            if name:
                parties.append({"name": name, "role": party.get("role") or "party"})
    return parties


def _normalize_clauses(value: Any, chunks: list[Chunk]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        return _chunk_clauses(chunks)

    clauses = []
    for clause in value:
        if not isinstance(clause, dict):
            continue
        number = clause.get("number") or clause.get("clause_number")
        summary = clause.get("summary") or clause.get("text") or ""
        clauses.append(
            {
                "number": number,
                "type": clause.get("type") or clause.get("clause_type"),
                "page": clause.get("page") or clause.get("page_start"),
                "summary": str(summary),
            }
        )
    return clauses or _chunk_clauses(chunks)


def contract_record_from_llm_json(contract_id: str, payload: dict[str, Any], chunks: list[Chunk]) -> ContractRecord:
    return ContractRecord(
        contract_id=contract_id,
        title=str(payload.get("title") or contract_id).strip(),
        value=_coerce_float(payload.get("contract_value", payload.get("value"))),
        currency=payload.get("currency"),
        effective_date=payload.get("effective_date"),
        expiry_date=payload.get("expiry_date"),
        governing_law=payload.get("governing_law"),
        parties=_normalize_parties(payload.get("parties")),
        clauses=_normalize_clauses(payload.get("clauses"), chunks),
    )


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _first_line(text: str, fallback: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), fallback)


def _extract_title(text: str, fallback: str) -> str:
    for pattern in [
        r"EXHIBIT\s+\S?B\S?",
        r"MASTER SERVICES AGREEMENT",
        r"GLOBAL MASTER SUPPLY AGREEMENT",
    ]:
        match = re.search(pattern, text[:1200], flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    first_line = _first_line(text, fallback)
    return first_line[:160].strip()


def _parse_written_date(text: str) -> str | None:
    match = re.search(
        r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+(?P<month>[A-Za-z]+),?\s+(?P<year>\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        month = MONTHS.get(match.group("month").lower())
        if month:
            return date(int(match.group("year")), month, int(match.group("day"))).isoformat()
    match = re.search(
        r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"(?P<day>\d{1,2}),?\s+(?P<year>\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        month = MONTHS[match.group("month").lower()]
        return date(int(match.group("year")), month, int(match.group("day"))).isoformat()
    match = re.search(r"(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})", text)
    if match:
        return date(int(match.group("year")), int(match.group("month")), int(match.group("day"))).isoformat()
    return None


def _add_years(iso_date: str, years: int) -> str:
    parsed = date.fromisoformat(iso_date)
    try:
        return parsed.replace(year=parsed.year + years).isoformat()
    except ValueError:
        return parsed.replace(month=2, day=28, year=parsed.year + years).isoformat()


def _extract_effective_date(text: str) -> str | None:
    effective_match = re.search(r"Effective Date\W{0,40}", text, flags=re.IGNORECASE)
    if effective_match:
        window = text[max(effective_match.start() - 140, 0) : effective_match.end() + 140]
        parsed = _parse_written_date(window)
        if parsed:
            return parsed
    return _parse_written_date(text[:1000])


def _extract_expiry_date(text: str, effective_date: str | None) -> str | None:
    termination_match = re.search(r"Termination Date:\s*(?P<date>[^\n.]+)", text, flags=re.IGNORECASE)
    if termination_match:
        parsed = _parse_written_date(termination_match.group("date"))
        if parsed:
            return parsed
    if effective_date and re.search(r"initial term .*?one\s+\(1\)\s+year from the Effective Date", text, re.IGNORECASE | re.DOTALL):
        return _add_years(effective_date, 1)
    return None


def _extract_parties(text: str) -> list[dict[str, str]]:
    match = re.search(
        r"by and between\s+(?P<first>.+?)(?:\s+\(\"[^\"]+\"\))?\s+and\s+(?P<second>.+?)(?:\s+\(\"[^\"]+\"\))?[.\n]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    parties = []
    for key in ("first", "second"):
        name = " ".join(match.group(key).split())
        name = re.sub(r"\s+\(\"[^\"]+\"\)", "", name)
        name = re.sub(r",?\s+with offices.*$", "", name, flags=re.IGNORECASE)
        name = re.sub(r",?\s+a\s+[^,]+(?:corporation|company|limited liability company).*$", "", name, flags=re.IGNORECASE)
        name = name.strip(" ,")
        if name:
            parties.append({"name": name, "role": "party"})
    return parties


def _extract_governing_law(text: str) -> str | None:
    match = re.search(r"laws of the (?P<law>State of [A-Za-z ]+?)(?:\s+without|\.|,)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return " ".join(match.group("law").split())


def extract_deterministic_record(contract_id: str, chunks: list[Chunk]) -> ContractRecord:
    text = "\n\n".join(chunk.text for chunk in chunks)
    clauses = [
        {
            "number": chunk.clause_number,
            "type": chunk.clause_type,
            "page": chunk.page_start,
            "summary": chunk.text[:240].replace("\n", " "),
        }
        for chunk in chunks
    ]
    effective_date = _extract_effective_date(text)
    return ContractRecord(
        contract_id=contract_id,
        title=_extract_title(text, contract_id),
        parties=_extract_parties(text),
        effective_date=effective_date,
        expiry_date=_extract_expiry_date(text, effective_date),
        governing_law=_extract_governing_law(text),
        clauses=clauses,
    )
