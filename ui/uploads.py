from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Literal

from ingestion.chunker import Chunk


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]

_WIDGET_KEY_DISALLOWED = re.compile(r"[^A-Za-z0-9_-]")


def classify_uploads(file_names: list[str]) -> Literal["pdf", "images"]:
    suffixes = [Path(name).suffix.lower() for name in file_names if name]
    if suffixes and all(suffix == ".pdf" for suffix in suffixes):
        return "pdf"
    if suffixes and all(suffix in IMAGE_EXTENSIONS for suffix in suffixes):
        return "images"
    raise ValueError("Upload either PDFs or contract images, not mixed file types.")


def reset_upload_dir(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def normalize_contract_id(raw: str) -> str:
    if raw is None:
        raise ValueError("Contract ID is required.")
    stripped = raw.strip()
    if not stripped:
        raise ValueError("Contract ID is required.")
    return stripped


def next_contract_id(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    for suffix in range(2, 1000):
        candidate = f"{base}-{suffix:03d}"
        if candidate not in taken:
            return candidate
    raise ValueError(f"Could not allocate a unique contract id from base {base}.")


def remove_contract_chunks(chunks: list[Chunk], contract_id: str) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk.contract_id != contract_id]


def replace_contract_chunks(
    existing: list[Chunk], new_chunks: list[Chunk], contract_id: str
) -> list[Chunk]:
    return remove_contract_chunks(existing, contract_id) + list(new_chunks)


def sanitize_widget_key(contract_id: str) -> str:
    if not contract_id:
        raise ValueError("Cannot sanitize an empty contract id.")
    return _WIDGET_KEY_DISALLOWED.sub("_", contract_id)
