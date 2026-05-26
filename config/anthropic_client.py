from __future__ import annotations

from typing import Any

from config.llm import llm_api_key, llm_model, llm_ocr_model


def _text_from_response(response: Any) -> str:
    parts = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def create_anthropic_text(
    system: str,
    user_content: str | list[dict[str, Any]],
    max_tokens: int,
    model: str | None = None,
    ocr: bool = False,
) -> str:
    from anthropic import Anthropic

    selected_model = llm_ocr_model(model, provider="anthropic") if ocr else llm_model(model, provider="anthropic")
    client = Anthropic(api_key=llm_api_key(provider="anthropic"))
    response = client.messages.create(
        model=selected_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    return _text_from_response(response)
