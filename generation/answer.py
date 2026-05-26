from __future__ import annotations

from config.env import load_env_file
from retrieval.hybrid_search import ScoredChunk
from generation.prompts import build_answer_messages


def answer_with_citations(query: str, hits: list[ScoredChunk], api_key: str | None = None, model: str = "gpt-5-mini") -> str:
    if not hits:
        return 'Không có trong tài liệu.'

    from openai import OpenAI

    load_env_file()
    messages = build_answer_messages(query, hits)
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=messages[0]["content"],
        input=messages[1]["content"],
        max_output_tokens=1200,
    )
    return response.output_text
