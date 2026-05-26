from __future__ import annotations

import os

from config.env import load_env_file


DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_OCR_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_OCR_MODEL = "gpt-4.1-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_ANTHROPIC_OCR_MODEL = "claude-haiku-4-5-20251001"


def gemini_api_key(explicit_api_key: str | None = None) -> str | None:
    load_env_file()
    return explicit_api_key or os.getenv("GEMINI_API_KEY")


def gemini_base_url() -> str:
    load_env_file()
    return os.getenv("GEMINI_BASE_URL") or DEFAULT_GEMINI_BASE_URL


def gemini_model(explicit_model: str | None = None) -> str:
    load_env_file()
    return explicit_model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def gemini_ocr_model(explicit_model: str | None = None) -> str:
    load_env_file()
    return (
        explicit_model
        or os.getenv("GEMINI_OCR_MODEL")
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_GEMINI_OCR_MODEL
    )


def openai_api_key(explicit_api_key: str | None = None) -> str | None:
    load_env_file()
    return explicit_api_key or os.getenv("OPENAI_API_KEY")


def openai_base_url() -> str | None:
    load_env_file()
    return os.getenv("OPENAI_BASE_URL") or None


def openai_model(explicit_model: str | None = None) -> str:
    load_env_file()
    return explicit_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def openai_ocr_model(explicit_model: str | None = None) -> str:
    load_env_file()
    return (
        explicit_model
        or os.getenv("OPENAI_OCR_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_OPENAI_OCR_MODEL
    )


def anthropic_api_key(explicit_api_key: str | None = None) -> str | None:
    load_env_file()
    return explicit_api_key or os.getenv("ANTHROPIC_API_KEY")


def anthropic_model(explicit_model: str | None = None) -> str:
    load_env_file()
    return explicit_model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL


def anthropic_ocr_model(explicit_model: str | None = None) -> str:
    load_env_file()
    return (
        explicit_model
        or os.getenv("ANTHROPIC_OCR_MODEL")
        or os.getenv("ANTHROPIC_MODEL")
        or DEFAULT_ANTHROPIC_OCR_MODEL
    )


def llm_provider() -> str:
    load_env_file()
    configured = os.getenv("LLM_PROVIDER")
    if configured:
        provider = configured.strip().lower()
        if provider in {"openai", "gemini", "anthropic"}:
            return provider
        raise ValueError("LLM_PROVIDER must be 'openai', 'gemini', or 'anthropic'")
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "gemini"


def llm_api_key(explicit_api_key: str | None = None, provider: str | None = None) -> str | None:
    selected = (provider or llm_provider()).strip().lower()
    if selected == "openai":
        return openai_api_key(explicit_api_key)
    if selected == "anthropic":
        return anthropic_api_key(explicit_api_key)
    if selected == "gemini":
        return gemini_api_key(explicit_api_key)
    raise ValueError("provider must be 'openai', 'gemini', or 'anthropic'")


def llm_base_url(provider: str | None = None) -> str | None:
    selected = (provider or llm_provider()).strip().lower()
    if selected == "openai":
        return openai_base_url()
    if selected == "anthropic":
        return None
    if selected == "gemini":
        return gemini_base_url()
    raise ValueError("provider must be 'openai', 'gemini', or 'anthropic'")


def llm_model(explicit_model: str | None = None, provider: str | None = None) -> str:
    selected = (provider or llm_provider()).strip().lower()
    if selected == "openai":
        return openai_model(explicit_model)
    if selected == "anthropic":
        return anthropic_model(explicit_model)
    if selected == "gemini":
        return gemini_model(explicit_model)
    raise ValueError("provider must be 'openai', 'gemini', or 'anthropic'")


def llm_ocr_model(explicit_model: str | None = None, provider: str | None = None) -> str:
    selected = (provider or llm_provider()).strip().lower()
    if selected == "openai":
        return openai_ocr_model(explicit_model)
    if selected == "anthropic":
        return anthropic_ocr_model(explicit_model)
    if selected == "gemini":
        return gemini_ocr_model(explicit_model)
    raise ValueError("provider must be 'openai', 'gemini', or 'anthropic'")
