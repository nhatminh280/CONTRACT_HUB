from typing import Literal


Intent = Literal["semantic", "structured", "keyword"]


def classify_intent(query: str) -> Intent:
    lowered = query.lower()
    if any(token in lowered for token in ["điều ", "article", "section", "clause", "hđ-", "hd-"]):
        return "keyword"
    if any(token in lowered for token in ["ngày", "date", "giá trị", "value", "tổng", "sum", "hết hạn"]):
        return "structured"
    return "semantic"
