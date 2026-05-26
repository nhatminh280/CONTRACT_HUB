from pathlib import Path
import inspect
import os
import re
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SAFE_EXTRA_KWARGS = {"device"}
os.environ.setdefault("FLAGS_use_cuda_managed_memory", "true")
os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.999")


class PaddleOCRVLRunner:
    """Thin PaddleOCR-VL wrapper that can release model memory after use."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._pipeline = None

    def _load(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

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
        return self._pipeline

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
        result = pipeline.predict(pdf_path)
        blocks: list[dict[str, Any]] = []
        for page_index, page in enumerate(result, start=1):
            text = self._prediction_text(page)
            if text.strip():
                blocks.append({"text": text.strip(), "page": page_index, "type": "ocr"})
        return blocks

    def parse_image_folder(self, folder_path: str, include_noisy: bool = False) -> list[dict[str, Any]]:
        pipeline = self._load()
        image_paths = []
        for path in sorted(Path(folder_path).iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not include_noisy and "_noisy" in path.stem:
                continue
            image_paths.append(path)

        blocks: list[dict[str, Any]] = []
        for fallback_index, path in enumerate(image_paths, start=1):
            match = re.search(r"page_(\d+)", path.stem)
            page_number = int(match.group(1)) if match else fallback_index
            result = pipeline.predict(str(path))
            text = self._prediction_text(result).strip()
            if text:
                blocks.append({"text": text, "page": page_number, "type": "ocr_image"})
        return sorted(blocks, key=lambda block: block["page"])

    def unload(self) -> None:
        self._pipeline = None
