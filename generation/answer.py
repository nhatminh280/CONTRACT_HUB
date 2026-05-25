from __future__ import annotations

from retrieval.hybrid_search import ScoredChunk
from generation.prompts import build_answer_messages


def answer_with_citations(query: str, hits: list[ScoredChunk], api_key: str | None = None, model: str = "claude-3-5-sonnet-latest") -> str:
    if not hits:
        return 'Không có trong tài liệu.'

    import anthropic

    messages = build_answer_messages(query, hits)
    system = messages[0]["content"]
    user_message = messages[1]
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1200,
        system=system,
        messages=[user_message],
    )
    return response.content[0].text
