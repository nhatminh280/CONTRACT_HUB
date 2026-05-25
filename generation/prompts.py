from retrieval.hybrid_search import ScoredChunk


SYSTEM_PROMPT = """Bạn là trợ lý phân tích hợp đồng pháp lý.
Chỉ trả lời dựa trên context được cung cấp.
Mỗi claim PHẢI cite [Điều X, trang Y, Hợp đồng Z].
Không tìm thấy -> nói rõ "Không có trong tài liệu"."""


def format_context(hits: list[ScoredChunk]) -> str:
    return "\n\n".join(f"{hit.chunk.citation}\n{hit.chunk.text}" for hit in hits)


def build_answer_messages(query: str, hits: list[ScoredChunk]) -> list[dict[str, str]]:
    context = format_context(hits)
    user_content = f"Context:\n{context}\n\nUser: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
