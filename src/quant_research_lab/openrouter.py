"""OpenRouter 结构化输出适配器：与 OllamaModel 相同的委员会协议、调用与费用上限。

密钥只从参数或 OPENROUTER_API_KEY 环境变量读取，不写入审计日志、缓存或异常信息。
OpenRouter 可能把同一模型路由到不同服务商；实际服务商逐次记录在审计日志里。
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .agents import committee_system_prompt
from .models import Proposal

T = TypeVar("T", bound=BaseModel)
API = "https://openrouter.ai/api/v1"
Transport = Callable[[str, dict[str, Any] | None, str], dict[str, Any]]


def http_transport(path: str, body: dict[str, Any] | None, api_key: str) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if body is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        API + path,
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read(20_000_001)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:200]
        raise ValueError(f"OpenRouter HTTP {error.code}：{detail}") from None
    if len(raw) > 20_000_000:
        raise ValueError("OpenRouter 响应超出大小限制")
    return dict(json.loads(raw))


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_round(item) for item in value]
    return value


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """发给模型的紧凑形式：证据改为一行一条、浮点保留4位；信息内容不变。

    原始证据摘要是"代码 已知收盘价 X，N条历史"，可得时点都不晚于本次截止时点，
    因此压缩为 "证据ID|收盘价|历史条数"。审计日志仍保存原始 payload。
    """
    result: dict[str, Any] = _round(
        {key: value for key, value in payload.items() if key != "evidence"}
    )
    rows = []
    for item in payload.get("evidence", []):
        close, count = item["summary"].split("已知收盘价 ")[1].split("，")
        rows.append(f"{item['id']}|{close}|{count.removesuffix('条历史')}")
    result["evidence_format"] = "evidence_id|known_close|history_rows (all known at as_of)"
    result["evidence"] = rows
    return result


def strict_schema(role: str, payload: dict[str, Any], manager: bool) -> dict[str, Any]:
    """严格结构化输出：全部字段必填、禁止额外字段；权重用 [{symbol, weight}] 列表表示。"""
    ids = [item["id"] for item in payload.get("evidence", [])]
    item: dict[str, Any] = {"type": "string", "enum": ids} if ids else {"type": "string"}
    # 与协议一致：至少引用1条、至多30条证据。
    cited = {"type": "array", "items": item, "minItems": 1, "maxItems": 30}
    if manager:
        symbols = sorted(payload.get("momentum") or {})
        symbol: dict[str, Any] = (
            {"type": "string", "enum": symbols} if symbols else {"type": "string"}
        )
        return {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "array",
                    "maxItems": int(payload.get("top_k", 20)),
                    "items": {
                        "type": "object",
                        "properties": {"symbol": symbol, "weight": {"type": "number"}},
                        "required": ["symbol", "weight"],
                        "additionalProperties": False,
                    },
                },
                "evidence_ids": cited,
                "summary": {"type": "string"},
            },
            "required": ["weights", "evidence_ids", "summary"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": [role]},
            "summary": {"type": "string"},
            "evidence_ids": cited,
            "concerns": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["role", "summary", "evidence_ids", "concerns"],
        "additionalProperties": False,
    }


def from_strict(output: dict[str, Any], manager: bool) -> dict[str, Any]:
    """把列表形式的权重转回 Proposal 的字典；同一股票重复给权重视为协议错误。"""
    if not manager:
        return output
    pairs = output.get("weights") or []
    symbols = [str(pair["symbol"]) for pair in pairs]
    if len(symbols) != len(set(symbols)):
        raise ValueError("模型对同一股票给出多个权重")
    return {**output, "weights": {str(p["symbol"]): float(p["weight"]) for p in pairs}}


class OpenRouterModel:
    def __init__(
        self,
        model: str,
        max_calls: int = 4,
        seed: int = 42,
        max_cost_usd: float = 1.0,
        audit_path: Path | None = None,
        api_key: str | None = None,
        transport: Transport = http_transport,
        reasoning_effort: str = "low",
        provider: str | None = None,
        max_repairs: int = 2,
    ) -> None:
        if not model.strip() or "/" not in model or max_calls < 1 or not 0 < max_cost_usd <= 5:
            raise ValueError(
                "需要 OpenRouter 模型ID（如 openai/gpt-oss-120b）、正调用预算与费用上限"
            )
        self.model, self.max_calls, self.seed = model, max_calls, seed
        self.max_cost_usd, self.audit_path = max_cost_usd, audit_path
        self.reasoning_effort = reasoning_effort
        # 服务商或端点标签（如 openai/flex）：固定单一端点且禁止回退，便于复现。
        self.provider = provider
        # 输出不符合协议时，把错误反馈给模型重答的最多次数；每次都计入调用与费用。
        self.max_repairs = max_repairs
        self._api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")
        self.transport = transport
        self.calls = 0
        self.cost_usd = 0.0
        self.expected_digest: str | None = None
        # 由 metadata() 从公开目录读取；推理型模型（如 gpt-5-nano）不接受 temperature。
        self.supported: set[str] | None = None
        self.audit: list[dict[str, Any]] = []

    def metadata(self) -> dict[str, Any]:
        """公开模型目录中的条目。OpenRouter 不提供权重哈希；digest 是目录 slug，不是权重指纹。"""
        for item in self.transport("/models", None, "")["data"]:
            if item["id"] == self.model:
                slug = str(item.get("canonical_slug") or item["id"])
                if item.get("supported_parameters"):
                    self.supported = set(item["supported_parameters"])
                return {
                    "name": item["id"],
                    "digest": "openrouter-slug:" + slug,
                    "details": {
                        "canonical_slug": slug,
                        "created": item.get("created"),
                        "context_length": item.get("context_length"),
                        "pricing": item.get("pricing"),
                        "digest_kind": "provider_catalog_slug_not_weight_hash",
                        "supported_parameters": sorted(self.supported or []),
                    },
                }
        raise ValueError("OpenRouter 目录中没有该模型；不会自动换模型")

    def _journal(self, event: dict[str, Any]) -> None:
        self.audit.append(event)
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def generate(self, role: str, payload: dict[str, Any], schema: type[T]) -> T:
        if not self._api_key:
            raise ValueError("缺少 OPENROUTER_API_KEY 环境变量；不会改用规则结果")
        manager = schema is Proposal
        routing: dict[str, Any] = {"require_parameters": True}
        if self.provider:
            # OpenRouter 的服务商名或端点标签（如 openai/flex、deepinfra/bf16），原样传递。
            routing.update(only=[self.provider], allow_fallbacks=False)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": committee_system_prompt(role)},
            {
                "role": "user",
                "content": json.dumps(
                    compact_payload(payload), ensure_ascii=False, allow_nan=False
                ),
            },
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": role,
                "strict": True,
                "schema": strict_schema(role, payload, manager),
            },
        }
        for repair in range(self.max_repairs + 1):
            if self.calls >= self.max_calls:
                raise RuntimeError("模型调用预算已用完")
            if self.cost_usd >= self.max_cost_usd:
                raise RuntimeError(f"模型费用达到上限 ${self.max_cost_usd}，停止运行")
            self.calls += 1
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4000,
                "reasoning": {"effort": self.reasoning_effort},
                "response_format": response_format,
                "provider": routing,
                "usage": {"include": True},
            }
            for name, value in (("temperature", 0), ("seed", self.seed)):
                if self.supported is None or name in self.supported:
                    body[name] = value
            started = time.monotonic()
            event: dict[str, Any] = {
                "role": role,
                "model": self.model,
                "backend": "openrouter",
                "call": self.calls,
                "repair": repair,
                "recorded_at": datetime.now(UTC).isoformat(),
                "input": payload,
                "request_sha256": hashlib.sha256(
                    json.dumps(body, sort_keys=True).encode()
                ).hexdigest(),
                "options": {
                    "temperature": body.get("temperature"),
                    "seed": body.get("seed"),
                    "reasoning": self.reasoning_effort,
                    "provider": routing,
                },
            }
            content = ""
            try:
                envelope = self.transport("/chat/completions", body, self._api_key)
                usage = envelope.get("usage") or {}
                self.cost_usd += float(usage.get("cost") or 0.0)
                event.update(
                    provider=envelope.get("provider"),
                    served_model=envelope.get("model"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    cost_usd=usage.get("cost"),
                )
                content = str(envelope["choices"][0]["message"]["content"] or "")
                event["raw_content"] = content
                result = schema.model_validate(from_strict(json.loads(content), manager))
                event.update(status="ok", output=result.model_dump(mode="json"))
                return result
            except (
                ValidationError,
                json.JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
            ) as error:
                # 只有"返回了内容但不符合协议"才重试；把错误告诉模型，重试次数与费用全部计入。
                event.update(status="invalid_output", error_type=type(error).__name__)
                if repair == self.max_repairs:
                    raise
                # 空回复（推理耗尽输出长度）不附上助手消息；两种情况都改变提示，避免同seed重复。
                previous = [{"role": "assistant", "content": content}] if content else []
                problem = (
                    f"did not match the required JSON schema ({type(error).__name__})"
                    if content
                    else "was empty"
                )
                messages = [
                    *messages,
                    *previous,
                    {
                        "role": "user",
                        "content": f"Your previous reply {problem}. Reply now with exactly one "
                        "JSON object that has only the schema's fields; keep reasoning brief.",
                    },
                ]
            except Exception as error:
                event.update(status="error", error_type=type(error).__name__)
                raise
            finally:
                event["elapsed_seconds"] = time.monotonic() - started
                self._journal(event)
        raise AssertionError("unreachable")
