from typing import Literal

try:
    import fitz
except ImportError:  # pragma: no cover - exercised only without dependency
    class _MissingFitz:
        def open(self, *_args, **_kwargs):
            raise ImportError("pymupdf is required for PDF routing")

    fitz = _MissingFitz()


def route(pdf_path: str, threshold_chars_per_page: int = 100) -> Literal["text", "scanned"]:
    """Classify a PDF as text-native or scanned using average extractable chars/page."""
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    if page_count == 0:
        return "scanned"
    total_chars = sum(len(page.get_text() or "") for page in doc)
    avg_chars = total_chars / page_count
    return "text" if avg_chars >= threshold_chars_per_page else "scanned"
