from pathlib import Path
import base64
import inspect
import os
import re
import warnings
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SAFE_EXTRA_KWARGS: set[str] = set()
os.environ.setdefault("FLAGS_use_cuda_managed_memory", "true")
os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.999")


OCR_API_PROMPT = """You are an OCR engine for legal contract documents.

Extract all visible text from this contract page.

Rules:
- Preserve clause numbers, section titles, party names, dates, amounts, currency symbols, percentages, and legal terms exactly.
- Preserve line breaks when they help readability.
- Do not summarize.
- Do not translate.
- Do not explain.
- If text is unclear, mark it as [unclear].
- Return only the extracted text."""


_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _resolve_ocr_device() -> str:
    device = os.environ.get("OCR_DEVICE", "cpu").strip().lower()
    if device == "gpu":
        device = "gpu:0"
    return device or "cpu"


def _ocr_fallback_enabled() -> bool:
    return os.environ.get("OCR_FALLBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _ocr_fallback_provider() -> str:
    configured = os.environ.get("OCR_API_FALLBACK_PROVIDER")
    if configured:
        return configured.strip().lower()
    from config.llm import llm_provider

    return llm_provider()


def _api_ocr_image(image_path: Path, provider: str) -> str:
    from config.llm import llm_api_key, llm_base_url, llm_ocr_model

    mime = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/png")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")

    if provider == "anthropic":
        from config.anthropic_client import create_anthropic_text

        return create_anthropic_text(
            system=OCR_API_PROMPT,
            user_content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": encoded,
                    },
                },
                {"type": "text", "text": "Extract the contract text from this page."},
            ],
            max_tokens=4096,
            ocr=True,
        )

    from openai import OpenAI

    client = OpenAI(api_key=llm_api_key(provider=provider), base_url=llm_base_url(provider=provider))
    response = client.chat.completions.create(
        model=llm_ocr_model(provider=provider),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_API_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return (response.choices[0].message.content or "").strip()


class PaddleOCRVLRunner:
    """Thin PaddleOCR-VL wrapper with optional LLM vision API fallback."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("device", None)
        self.kwargs = kwargs
        self._pipeline = None
        self._init_error: str | None = None

    def _load(self) -> Any:
        if self._pipeline is not None or self._init_error is not None:
            return self._pipeline

        try:
            import paddle
            from paddleocr import PaddleOCRVL

            paddle.set_device(_resolve_ocr_device())

            defaults = {
                "pipeline_version": "v1.5",
                "use_layout_detection": False,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_chart_recognition": False,
                "use_seal_recognition": False,
                "use_ocr_for_image_block": False,
            }
            defaults.update(self.kwargs)
            signature = inspect.signature(PaddleOCRVL)
            accepts_var_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if accepts_var_kwargs:
                accepted = defaults
            else:
                accepted = {
                    key: value
                    for key, value in defaults.items()
                    if key in signature.parameters or key in SAFE_EXTRA_KWARGS
                }
            self._pipeline = PaddleOCRVL(**accepted)
        except Exception as exc:
            self._init_error = f"PaddleOCR-VL initialization failed: {exc}"
            warnings.warn(self._init_error, RuntimeWarning, stacklevel=2)
        return self._pipeline

    def _api_fallback_image_block(self, path: Path, page_number: int) -> dict[str, Any]:
        provider = _ocr_fallback_provider()
        if provider not in {"gemini", "openai", "anthropic"}:
            return {
                "text": f"OCR fallback provider '{provider}' is not supported",
                "page": page_number,
                "type": "ocr_error",
                "source": "fallback",
            }
        try:
            text = _api_ocr_image(path, provider)
        except Exception as exc:
            return {
                "text": f"{provider} OCR fallback failed: {exc}",
                "page": page_number,
                "type": "ocr_error",
                "source": "fallback",
            }
        warnings.warn(
            f"{provider} OCR fallback used for {path.name} (page {page_number})",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "text": text,
            "page": page_number,
            "type": "ocr_api",
            "source": provider,
        }

    def _ocr_error_block(self, page: int, detail: str | None = None) -> dict[str, Any]:
        message = detail or self._init_error or "PaddleOCR-VL unavailable"
        return {"text": message, "page": page, "type": "ocr_error", "source": "fallback"}

    def _can_extract_text(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, dict)):
            return bool(value)
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return False
        return any(getattr(value, attr, None) is not None for attr in ("markdown", "text", "rec_text", "ocr_text"))

    def _prediction_text(self, prediction: Any) -> str:
        if prediction is None:
            return ""
        if isinstance(prediction, str):
            return prediction
        if isinstance(prediction, dict):
            if "res" in prediction and self._can_extract_text(prediction["res"]):
                return self._prediction_text(prediction["res"])
            if "parsing_res_list" in prediction:
                texts = []
                for block in prediction.get("parsing_res_list") or []:
                    content = getattr(block, "content", None)
                    if content is None and isinstance(block, dict):
                        content = block.get("block_content")
                    if self._can_extract_text(content):
                        texts.append(self._prediction_text(content))
                if texts:
                    return "\n\n".join(text for text in texts if text)
            for key in ("markdown_texts", "block_content", "markdown", "text", "rec_text", "ocr_text"):
                value = prediction.get(key)
                if self._can_extract_text(value):
                    text = self._prediction_text(value)
                    if text:
                        return text
            return "\n".join(
                self._prediction_text(value)
                for value in prediction.values()
                if self._can_extract_text(value)
            )
        if isinstance(prediction, (list, tuple)):
            return "\n".join(
                self._prediction_text(item)
                for item in prediction
                if self._can_extract_text(item)
            )
        for attr in ("content", "markdown", "json", "text", "rec_text", "ocr_text"):
            value = getattr(prediction, attr, None)
            if self._can_extract_text(value):
                return self._prediction_text(value)
        if hasattr(prediction, "shape") and hasattr(prediction, "dtype"):
            return ""
        return str(prediction)

    def parse(self, pdf_path: str) -> list[dict[str, Any]]:
        pipeline = self._load()
        if pipeline is None:
            return [self._ocr_error_block(page=1)]
        try:
            result = pipeline.predict(pdf_path)
            blocks: list[dict[str, Any]] = []
            for page_index, page in enumerate(result, start=1):
                text = self._prediction_text(page)
                if text.strip():
                    blocks.append({"text": text.strip(), "page": page_index, "type": "ocr"})
            return blocks
        except Exception as exc:
            return [self._ocr_error_block(page=1, detail=f"PaddleOCR-VL predict failed: {exc}")]

    def parse_image_folder(self, folder_path: str, include_noisy: bool = False) -> list[dict[str, Any]]:
        pipeline = self._load()
        image_paths: list[Path] = []
        for path in sorted(Path(folder_path).iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not include_noisy and "_noisy" in path.stem:
                continue
            image_paths.append(path)

        fallback_enabled = _ocr_fallback_enabled()
        blocks: list[dict[str, Any]] = []
        for fallback_index, path in enumerate(image_paths, start=1):
            match = re.search(r"page_(\d+)", path.stem)
            page_number = int(match.group(1)) if match else fallback_index

            paddle_error: str | None = self._init_error
            if pipeline is not None:
                try:
                    result = pipeline.predict(str(path))
                    text = self._prediction_text(result).strip()
                    if text:
                        blocks.append({
                            "text": text,
                            "page": page_number,
                            "type": "ocr_image",
                            "source": "paddle",
                        })
                        continue
                    paddle_error = "PaddleOCR-VL returned empty text"
                except Exception as exc:
                    paddle_error = f"PaddleOCR-VL predict failed: {exc}"
                    warnings.warn(
                        f"{paddle_error} for {path.name}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            if fallback_enabled:
                blocks.append(self._api_fallback_image_block(path, page_number))
            else:
                blocks.append(self._ocr_error_block(page=page_number, detail=paddle_error))

        return sorted(blocks, key=lambda block: block["page"])

    def unload(self) -> None:
        self._pipeline = None
        self._init_error = None
