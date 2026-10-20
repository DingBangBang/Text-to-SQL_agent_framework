"""Token guardrail tests — the 3,000-token ceiling must be mathematically hard."""

from __future__ import annotations

from src.token_budget import (
    TokenBudget,
    count_message_tokens,
    count_tokens,
    truncate_text_to_tokens,
)


def test_token_counter_is_non_zero():
    assert count_tokens("为什么上周链上交易量跌了？") >= 5
    assert count_tokens("") == 0


def test_small_payload_is_untouched():
    messages = [
        {"role": "system", "content": "你是分析师。"},
        {"role": "user", "content": "过去7天链上交易量是多少？"},
    ]
    safe, report = TokenBudget(3000).enforce(messages)
    assert report.truncated is False
    assert safe == messages


def test_oversized_payload_is_truncated_to_limit():
    huge = "BTC 链上交易明细 " * 4000
    messages = [
        {"role": "system", "content": "只输出 JSON。"},
        {"role": "user", "content": huge},
    ]
    safe, report = TokenBudget(3000).enforce(messages)

    assert report.truncated is True
    assert report.original_tokens > 3000
    assert count_message_tokens(safe) <= 3000
    assert safe[0]["content"] == "只输出 JSON。"


def test_many_messages_still_fit_budget():
    messages = [{"role": "system", "content": "规则" * 200}]
    for index in range(40):
        messages.append(
            {"role": "user", "content": f"第{index}批数据：" + "1,234.56 " * 120}
        )
    safe, report = TokenBudget(3000).enforce(messages)

    assert report.truncated is True
    assert count_message_tokens(safe) <= 3000


def test_truncate_text_keeps_head_and_tail():
    text = "HEAD-" + "数据" * 3000 + "-TAIL"
    truncated = truncate_text_to_tokens(text, 200)
    assert truncated.startswith("HEAD-")
    assert "TAIL" in truncated
    assert count_tokens(truncated) <= 220
