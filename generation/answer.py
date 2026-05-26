from __future__ import annotations

from config.llm import llm_api_key, llm_base_url, llm_model, llm_provider
from retrieval.hybrid_search import ScoredChunk
from generation.prompts import build_answer_messages


def answer_with_citations(query: str, hits: list[ScoredChunk], api_key: str | None = None, model: str | None = None) -> str:
    if not hits:
        return 'Không có trong tài liệu.'

    from openai import OpenAI

    messages = build_answer_messages(query, hits)
    provider = llm_provider()
    if provider == "anthropic":
        from config.anthropic_client import create_anthropic_text

        return create_anthropic_text(
            system=messages[0]["content"],
            user_content=messages[1]["content"],
            max_tokens=1200,
            model=model,
        )

    client = OpenAI(api_key=llm_api_key(api_key, provider=provider), base_url=llm_base_url(provider=provider))
    response = client.chat.completions.create(
        model=llm_model(model, provider=provider),
        messages=messages,
        max_tokens=1200,
    )
    return response.choices[0].message.content or ""
