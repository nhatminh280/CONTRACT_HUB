import re
from typing import Literal


Intent = Literal["semantic", "structured", "keyword"]


def classify_intent(query: str) -> Intent:
    lowered = query.lower()
    if re.search(r"\b(?:điều|article|section|clause)\s+\d+(?:[.\-]\d+)*\b", lowered):
        return "keyword"
    if re.search(r"\b(?:hđ|hd)-[\w.-]+\b", lowered):
        return "keyword"
    if any(
        token in lowered
        for token in [
            "ngày",
            "date",
            "giá trị",
            "value",
            "tổng",
            "sum",
            "hết hạn",
            "expiry",
            "expiration",
        ]
    ):
        return "structured"
    return "semantic"
