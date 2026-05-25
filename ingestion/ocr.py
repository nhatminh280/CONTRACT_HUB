from pathlib import Path
import inspect
import re
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class PaddleOCRVLRunner:
    """Thin PaddleOCR-VL wrapper that can release model memory after use."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._pipeline = None

    def _load(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

            defaults = {"pipeline_version": "v1.5"}
            defaults.update(self.kwargs)
            signature = inspect.signature(PaddleOCRVL)
            accepts_extra_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if accepts_extra_kwargs:
                accepted = defaults
            else:
                accepted = {key: value for key, value in defaults.items() if key in signature.parameters}
            self._pipeline = PaddleOCRVL(**accepted)
        return self._pipeline

    def _prediction_text(self, prediction: Any) -> str:
        if prediction is None:
            return ""
        if isinstance(prediction, str):
            return prediction
        if isinstance(prediction, dict):
            preferred = []
            for key in ("markdown", "text", "rec_text", "ocr_text"):
                value = prediction.get(key)
                if value:
                    preferred.append(self._prediction_text(value))
            if preferred:
                return "\n".join(text for text in preferred if text)
            return "\n".join(self._prediction_text(value) for value in prediction.values() if value)
        if isinstance(prediction, (list, tuple)):
            return "\n".join(self._prediction_text(item) for item in prediction if item)
        for attr in ("markdown", "text", "rec_text", "ocr_text"):
            value = getattr(prediction, attr, None)
            if value:
                return self._prediction_text(value)
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
