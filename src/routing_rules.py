"""Deterministic routing helpers shared by MockLLM and the fallback router."""

from __future__ import annotations

from .semantic import TIME_ALIASES, SemanticRegistry

ATTRIBUTION_KEYWORDS = (
    "为什么",
    "为啥",
    "原因",
    "归因",
    "拆解",
    "贡献",
    "拖累",
    "跌",
    "涨",
    "下降",
    "下滑",
    "增长",
    "why",
    "attribute",
)


def is_attribution(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in ATTRIBUTION_KEYWORDS)


def detect_time_range(question: str, registry: SemanticRegistry) -> str | None:
    """Longest alias wins, so '过去30天' beats any shorter overlapping alias."""
    lowered = question.lower()
    candidates: list[tuple[int, str]] = []
    for alias, key in TIME_ALIASES.items():
        if alias in lowered and key in registry.time_range_specs:
            candidates.append((len(alias), key))
    for key in registry.time_range_specs:
        if key in lowered:
            candidates.append((len(key), key))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
