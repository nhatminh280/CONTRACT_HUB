from __future__ import annotations

from config.llm import gemini_api_key, gemini_base_url, gemini_model
from retrieval.hybrid_search import ScoredChunk
from generation.prompts import build_answer_messages


def answer_with_citations(query: str, hits: list[ScoredChunk], api_key: str | None = None, model: str = "gemini-3.5-flash") -> str:
    if not hits:
        return 'Không có trong tài liệu.'

    from openai import OpenAI

    messages = build_answer_messages(query, hits)
    client = OpenAI(api_key=gemini_api_key(api_key), base_url=gemini_base_url())
    response = client.chat.completions.create(
        model=gemini_model(model),
        messages=messages,
        max_tokens=1200,
    )
    return response.choices[0].message.content or ""
