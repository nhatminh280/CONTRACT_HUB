from __future__ import annotations

import os
from pathlib import Path


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
            continue
        if char == "#" and quote is None:
            return value[:index].rstrip()
    return value.strip()


def _clean_value(value: str) -> str:
    cleaned = _strip_inline_comment(value)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned


def load_env_file(path: str | Path = ".env", override: bool = False) -> bool:
    env_path = Path(path)
    if not env_path.exists():
        return False

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = _clean_value(value.strip())
    return True
