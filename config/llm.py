from __future__ import annotations

import os

from config.env import load_env_file


DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


def gemini_api_key(explicit_api_key: str | None = None) -> str | None:
    load_env_file()
    return explicit_api_key or os.getenv("GEMINI_API_KEY")


def gemini_base_url() -> str:
    load_env_file()
    return os.getenv("GEMINI_BASE_URL") or DEFAULT_GEMINI_BASE_URL


def gemini_model(explicit_model: str | None = None) -> str:
    load_env_file()
    return explicit_model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
