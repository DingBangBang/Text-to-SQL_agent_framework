"""LLM access layer: DeepSeek client + strict token/cost guardrails.

The LLM is only ever asked to do three things:
1. intent routing (small JSON),
2. SQL generation / repair (small prompt, schema + metric semantics),
3. final narrative summarisation (markdown tables already computed locally).

Raw rows never leave the machine: the insight prompt only receives aggregated
query results, and the token budget truncates anything oversized.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from .config import Settings
from .token_budget import BudgetReport, TokenBudget, count_tokens


class LLMError(RuntimeError):
    """Any non-retryable LLM failure."""


class MissingAPIKeyError(LLMError):
    pass


class CostBudgetExceeded(LLMError):
    pass


class LLMFormatError(LLMError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_cny: float = 0.0
    model: str = "mock"
    mock: bool = False
    latency_ms: int = 0
    input_truncated: bool = False
    budget: BudgetReport | None = None


class BaseLLM(Protocol):
    model: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        task: str = "generic",
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a possibly fenced LLM answer."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMFormatError(f"LLM 未返回 JSON 对象：{text[:200]!r}")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMFormatError(f"LLM 返回的 JSON 无法解析：{exc}") from exc
    if not isinstance(payload, dict):
        raise LLMFormatError("LLM 返回的 JSON 顶层必须是对象")
    return payload



class DeepSeekLLM:
    """Minimal DeepSeek chat-completions client with a hard cost meter."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        budget: TokenBudget | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.model = self.settings.model
        self.budget = budget or TokenBudget(self.settings.max_input_tokens)
        self.session = session or requests.Session()
        self.spent_cny = 0.0
        self.calls: list[dict[str, Any]] = []
        self._last_budget_report: BudgetReport | None = None

    # -- metering ---------------------------------------------------------
    def reset_meter(self) -> None:
        self.spent_cny = 0.0
        self.calls = []

    @property
    def total_input_tokens(self) -> int:
        return sum(c["input_tokens"] for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c["output_tokens"] for c in self.calls)

    # -- core -------------------------------------------------------------
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        task: str = "generic",
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if not self.settings.api_key:
            raise MissingAPIKeyError(
                "未检测到 DEEPSEEK_API_KEY，请在 environment.env 中配置后重试。"
            )

        safe_messages, report = self.budget.enforce(messages)
        self._last_budget_report = report
        projected_cost = self.settings.cost_cny(
            count_tokens(json.dumps(safe_messages, ensure_ascii=False)), 1200
        )
        if self.spent_cny + projected_cost > self.settings.max_run_cost_cny:
            raise CostBudgetExceeded(
                f"预计本次调用将超出单次运行成本上限 "
                f"¥{self.settings.max_run_cost_cny:.2f}（已花费 ¥{self.spent_cny:.4f}）。"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": safe_messages,
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
            "max_tokens": 1200,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        response = self._post(payload)
        latency_ms = int((time.perf_counter() - started) * 1000)

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {}) or {}
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"DeepSeek 响应格式异常：{response.text[:300]}") from exc

        prompt_tokens = int(usage.get("prompt_tokens", 0)) or count_tokens(
            json.dumps(safe_messages, ensure_ascii=False)
        )
        output_tokens = int(usage.get("completion_tokens", 0)) or count_tokens(content)
        cached_tokens = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        cost = self.settings.cost_cny(prompt_tokens, output_tokens, cached_tokens)
        self.spent_cny += cost
        self.calls.append(
            {
                "task": task,
                "model": self.model,
                "input_tokens": prompt_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "cost_cny": cost,
                "latency_ms": latency_ms,
            }
        )

        return LLMResponse(
            text=content,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            cost_cny=cost,
            model=self.model,
            mock=False,
            latency_ms=latency_ms,
            input_truncated=report.truncated,
            budget=report,
        )


    def _post(self, payload: dict[str, Any]) -> requests.Response:
        """One transport-level retry for 429/5xx only; never loops on bad output."""
        url = f"{self.settings.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.settings.request_timeout_s,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                raise LLMError(f"DeepSeek 网络请求失败：{exc}") from exc

            if response.status_code == 200:
                return response
            if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                time.sleep(2.0)
                continue
            raise LLMError(
                f"DeepSeek API 返回 {response.status_code}：{response.text[:300]}"
            )
        raise LLMError(f"DeepSeek 请求失败：{last_error}")


def build_llm(settings: Settings | None = None, **kwargs: Any) -> BaseLLM:
    """Factory: real DeepSeek client or the deterministic local MockLLM."""
    settings = settings or Settings.from_env()
    if settings.llm_mode == "deepseek":
        return DeepSeekLLM(settings, **kwargs)
    from .mock_llm import MockLLM

    return MockLLM(settings, **kwargs)
