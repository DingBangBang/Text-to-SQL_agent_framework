"""Tiny bilingual helper shared by the agent and the dashboard."""

from __future__ import annotations

SUPPORTED_LANGUAGES = ("zh", "en")
LANGUAGE_LABELS = {"zh": "中文", "en": "English"}

DIMENSION_LABELS: dict[str, dict[str, str]] = {
    "symbol": {"zh": "币种", "en": "asset"},
    "chain_type": {"zh": "链类型", "en": "chain type"},
    "trade_date": {"zh": "日期", "en": "date"},
}


def normalize_language(value: str | None) -> str:
    return "en" if str(value).lower().startswith("en") else "zh"


def tr(language: str, zh: str, en: str) -> str:
    return en if normalize_language(language) == "en" else zh


def dimension_label(key: str, language: str = "zh") -> str:
    return DIMENSION_LABELS.get(key, {}).get(normalize_language(language), key)
