from __future__ import annotations

from pathlib import Path


def demo_index_dir(root: Path) -> Path:
    return root / "outputs" / "full_corpus_index"


def demo_sqlite_path(root: Path) -> Path:
    return demo_index_dir(root) / "contracts.sqlite"


def demo_bm25_path(root: Path) -> Path:
    return demo_index_dir(root) / "bm25_chunks.pkl"
