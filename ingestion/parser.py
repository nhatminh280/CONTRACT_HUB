from typing import Any

try:
    import fitz
except ImportError:  # pragma: no cover - exercised only without dependency
    class _MissingFitz:
        def open(self, *_args, **_kwargs):
            raise ImportError("pymupdf is required for PDF parsing")

    fitz = _MissingFitz()


def _table_to_markdown(rows: list[list[Any]]) -> str:
    clean_rows = [["" if cell is None else str(cell) for cell in row] for row in rows if row]
    if not clean_rows:
        return ""

    width = max(len(row) for row in clean_rows)
    normalized = [row + [""] * (width - len(row)) for row in clean_rows]
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(row) + " |"

    lines = [render(header), render(separator)]
    lines.extend(render(row) for row in body)
    return "\n".join(lines)


def _extract_tables(page: Any) -> list[str]:
    if not hasattr(page, "find_tables"):
        return []
    tables_obj = page.find_tables()
    tables = getattr(tables_obj, "tables", []) or []
    markdown_tables = []
    for table in tables:
        markdown = _table_to_markdown(table.extract())
        if markdown:
            markdown_tables.append(markdown)
    return markdown_tables


def parse_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """Extract text blocks and detected tables from a text PDF with page numbers."""
    doc = fitz.open(pdf_path)
    blocks: list[dict[str, Any]] = []

    for index, page in enumerate(doc, start=1):
        text = (page.get_text("text") or "").strip()
        if text:
            blocks.append({"text": text, "page": index, "type": "text"})
        for markdown_table in _extract_tables(page):
            blocks.append({"text": markdown_table, "page": index, "type": "table"})

    return blocks
