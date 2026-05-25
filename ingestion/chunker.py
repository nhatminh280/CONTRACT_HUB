from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any


CLAUSE_PATTERNS = [
    re.compile(r"^(?P<number>(?:Điều|ĐIỀU)\s+\d+(?:[.\-]\d+)*)", re.MULTILINE),
    re.compile(r"^(?P<number>(?:Article|Section|Clause)\s+\d+(?:[.\-]\d+)*)", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(?P<number>\d+\.)\s+", re.MULTILINE),
]
CUAD_HEADING_PATTERN = re.compile(r"^(?P<number>[A-Z][A-Z0-9/&()' -]{2,80}):\s*(?P<body>.*)$")
CUAD_STANDALONE_HEADING_PATTERN = re.compile(r"^(?P<number>(?:SCHEDULE|EXHIBIT|ATTACHMENT)\s+[A-Z0-9][A-Z0-9/&()' -]{0,80})$")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    contract_id: str
    clause_number: str
    page_start: int
    page_end: int
    clause_type: str = "general"

    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            page = f"trang {self.page_start}"
        else:
            page = f"trang {self.page_start}-{self.page_end}"
        return f"[{self.clause_number}, {page}, {self.contract_id}]"


def _detect_clause_number(text: str) -> str | None:
    stripped = text.lstrip()
    for pattern in CLAUSE_PATTERNS:
        match = pattern.search(stripped)
        if match and match.start() == 0:
            return match.group("number").rstrip(".")
    return None


def _is_cuad_heading(line: str) -> re.Match[str] | None:
    stripped = line.strip()
    match = CUAD_HEADING_PATTERN.match(stripped)
    if not match:
        return None
    heading = match.group("number").strip()
    if heading in {"BY", "ITS", "DATE"}:
        return None
    return match


def _detect_standalone_heading(line: str) -> str | None:
    stripped = line.strip()
    match = CUAD_STANDALONE_HEADING_PATTERN.match(stripped)
    if match:
        return match.group("number").strip()
    return None


def _split_block_into_segments(text: str, page: int) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current_number: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        heading = _is_cuad_heading(line)
        if heading:
            if current_lines:
                segments.append(
                    {
                        "text": "\n".join(current_lines).strip(),
                        "page": page,
                        "clause_number": current_number,
                    }
                )
            current_number = heading.group("number").strip()
            body = heading.group("body").strip()
            current_lines = [line.strip() if body else f"{current_number}:"]
            continue
        standalone_heading = _detect_standalone_heading(line)
        if standalone_heading:
            if current_lines:
                segments.append(
                    {
                        "text": "\n".join(current_lines).strip(),
                        "page": page,
                        "clause_number": current_number,
                    }
                )
            current_number = standalone_heading
            current_lines = [line.strip()]
            continue
        current_lines.append(line)

    if current_lines:
        segments.append(
            {
                "text": "\n".join(current_lines).strip(),
                "page": page,
                "clause_number": current_number,
            }
        )
    return [segment for segment in segments if segment["text"]]


def _chunk_id(contract_id: str, clause_number: str, page_start: int, offset: int, text: str) -> str:
    digest = hashlib.sha1(f"{contract_id}|{clause_number}|{page_start}|{offset}|{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def _split_long_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]
    step = max(max_tokens - overlap, 1)
    windows = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + max_tokens]
        if not window:
            break
        windows.append(" ".join(window))
        if start + max_tokens >= len(tokens):
            break
    return windows


def _infer_clause_type(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ["payment", "thanh toán", "giá trị"]):
        return "payment_terms"
    if any(term in lower for term in ["penalty", "phạt", "vi phạm"]):
        return "penalty"
    if any(term in lower for term in ["termination", "chấm dứt"]):
        return "termination"
    return "general"


def chunk_blocks(
    blocks: list[dict[str, Any]],
    contract_id: str,
    max_tokens: int = 1000,
    overlap: int = 128,
) -> list[Chunk]:
    """Chunk parsed blocks by clause boundaries while preserving page citations."""
    clauses: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for block in blocks:
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        page = int(block.get("page", 1))
        for segment in _split_block_into_segments(text, page):
            segment_text = segment["text"]
            clause_number = segment["clause_number"] or _detect_clause_number(segment_text)
            if clause_number or current is None:
                if current is not None:
                    clauses.append(current)
                current = {
                    "clause_number": clause_number or "Document",
                    "texts": [segment_text],
                    "page_start": page,
                    "page_end": page,
                }
            else:
                current["texts"].append(segment_text)
                current["page_end"] = page

    if current is not None:
        clauses.append(current)

    chunks: list[Chunk] = []
    for clause in clauses:
        combined = "\n\n".join(clause["texts"]).strip()
        windows = _split_long_text(combined, max_tokens=max_tokens, overlap=overlap)
        for offset, window in enumerate(windows):
            clause_number = str(clause["clause_number"])
            chunks.append(
                Chunk(
                    id=_chunk_id(contract_id, clause_number, int(clause["page_start"]), offset, window),
                    text=window,
                    contract_id=contract_id,
                    clause_number=clause_number,
                    page_start=int(clause["page_start"]),
                    page_end=int(clause["page_end"]),
                    clause_type=_infer_clause_type(window),
                )
            )

    return chunks
