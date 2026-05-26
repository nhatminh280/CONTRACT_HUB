from __future__ import annotations

from datetime import date
import json
import re
from typing import Any

from config.env import load_env_file
from indexing.sql_store import ContractRecord
from ingestion.chunker import Chunk


EXTRACTION_PROMPT = """Extract contract metadata as strict JSON with:
contract_id, title, parties, effective_date, expiry_date, contract_value,
currency, governing_law, clauses.
Each clause must include clause_number, clause_type, page, summary.
Return JSON only."""


def extract_structured_json(text: str, api_key: str | None = None, model: str = "gpt-5-mini") -> dict[str, Any]:
    """Extract structured contract data through OpenAI."""
    from openai import OpenAI

    load_env_file()
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=EXTRACTION_PROMPT,
        input=text,
        max_output_tokens=2000,
    )
    return json.loads(response.output_text)


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
