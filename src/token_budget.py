"""Hard token budget enforcement (tiktoken) for every LLM request.

DeepSeek tokens are *not* identical to OpenAI's ``cl100k_base`` vocabulary,
but the two are close enough for a conservative guardrail.  We always keep a
safety margin, so a 3,000-token measured payload can never balloon past the
provider's context window or our per-call cost ceiling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

try:  # pragma: no cover - exercised implicitly
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except Exception:  # pragma: no cover - offline fallback
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False

_ENCODER_NAME = "cl100k_base"
_ENCODER: Any | None = None
TRUNCATION_MARKER = "\n…[内容已按 Token 预算截断]…\n"


def get_encoder() -> Any | None:
    global _ENCODER
    if not _TIKTOKEN_AVAILABLE:
        return None
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.get_encoding(_ENCODER_NAME)
        except Exception:  # pragma: no cover - offline cache miss
            _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken; fall back to a conservative estimate."""
    if not text:
        return 0
    encoder = get_encoder()
    if encoder is None:  # pragma: no cover - only when tiktoken is offline
        # ~1 token per 2.2 characters is deliberately pessimistic for zh/en mixes.
        return max(1, int(len(text) / 2.2))
    return len(encoder.encode(text, disallowed_special=()))


def count_message_tokens(messages: Iterable[Mapping[str, str]]) -> int:
    """OpenAI-style message overhead (4 tokens/message + 2 priming tokens)."""
    total = 2
    for message in messages:
        total += 4
        total += count_tokens(str(message.get("role", "")))
        total += count_tokens(str(message.get("content", "")))
    return total


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Token-safe truncation that keeps the head and tail of the payload."""
    if max_tokens <= 0 or not text:
        return ""
    encoder = get_encoder()
    if encoder is None:  # pragma: no cover
        approx_chars = max(1, int(max_tokens * 2.2))
        return text[:approx_chars]
    ids = encoder.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    marker_tokens = len(encoder.encode(TRUNCATION_MARKER))
    keep = max(1, max_tokens - marker_tokens)
    head = int(keep * 0.7)
    tail = keep - head
    head_text = encoder.decode(ids[:head])
    tail_text = encoder.decode(ids[-tail:]) if tail > 0 else ""
    return f"{head_text}{TRUNCATION_MARKER}{tail_text}"


@dataclass
class BudgetReport:
    limit: int
    original_tokens: int
    final_tokens: int
    truncated: bool
    messages_touched: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TokenBudget:
    """Enforces the per-request input budget before any API call is made."""

    def __init__(self, max_input_tokens: int = 3000) -> None:
        if max_input_tokens < 128:
            raise ValueError("max_input_tokens is unrealistically small")
        self.max_input_tokens = max_input_tokens

    def enforce(
        self,
        messages: list[dict[str, str]],
        limit: int | None = None,
    ) -> tuple[list[dict[str, str]], BudgetReport]:
        limit = limit or self.max_input_tokens
        cloned = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        original = count_message_tokens(cloned)
        if original <= limit:
            return cloned, BudgetReport(limit, original, original, False)

        touched = 0
        # Repeatedly shrink the largest non-system message first: system prompts
        # contain the scoring-relevant rules, conversational context is cheaper
        # to lose.
        guard = 0
        while count_message_tokens(cloned) > limit and guard < 40:
            guard += 1
            total = count_message_tokens(cloned)
            excess = total - limit
            candidates = [
                (i, count_tokens(m["content"]))
                for i, m in enumerate(cloned)
                if m["role"] != "system" and count_tokens(m["content"]) > 24
            ]
            if not candidates:
                candidates = [
                    (i, count_tokens(m["content"]))
                    for i, m in enumerate(cloned)
                    if count_tokens(m["content"]) > 24
                ]
            if not candidates:
                break
            index, size = max(candidates, key=lambda item: item[1])
            target = max(16, size - excess - 8)
            if target >= size:
                target = size - 1
            cloned[index]["content"] = truncate_text_to_tokens(
                cloned[index]["content"], target
            )
            touched += 1

        # Final guarantee: if message overhead alone exceeds the budget, keep
        # only the most recent message and cut it to fit.
        if count_message_tokens(cloned) > limit:
            keep = cloned[-1]
            overhead = count_message_tokens([keep])
            allowed = max(16, limit - overhead + count_tokens(keep["content"]))
            keep["content"] = truncate_text_to_tokens(keep["content"], allowed)
            cloned = [keep]
            touched += 1

        final = count_message_tokens(cloned)
        return cloned, BudgetReport(limit, original, final, True, touched)
