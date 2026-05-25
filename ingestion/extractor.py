from __future__ import annotations

import json
from typing import Any


EXTRACTION_PROMPT = """Extract contract metadata as strict JSON with:
contract_id, title, parties, effective_date, expiry_date, contract_value,
currency, governing_law, clauses.
Each clause must include clause_number, clause_type, page, summary.
Return JSON only."""


def extract_structured_json(text: str, api_key: str | None = None, model: str = "claude-3-5-haiku-latest") -> dict[str, Any]:
    """Extract structured contract data through Claude Haiku."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=2000,
        system=EXTRACTION_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    content = message.content[0].text
    return json.loads(content)
