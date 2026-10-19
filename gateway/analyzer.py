"""Turns an incoming chat request into a scheduling Request: tenant, class, deadline, tokens."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from gateway.config import ClassConfig, GatewayConfig, TenantConfig

GATEWAY_FIELDS = frozenset({"service_class", "task_type", "session_id"})
CHAT_TEMPLATE_TOKENS_PER_MESSAGE = 4


class BadRequest(Exception):
    pass


@dataclass(eq=False)
class Request:
    id: str
    tenant: str
    sensitive: bool
    cls: ClassConfig
    rank: int  # current queue rank; aging can lower it by one
    vllm_priority: int
    body: dict  # what is forwarded upstream (gateway-only fields removed)
    prompt_tokens: int
    max_tokens: int
    output_estimate: int  # min(max_tokens, class p90): used for admission and queue-wait estimates
    task_type: str
    session_id: str | None
    stream: bool
    arrival: float
    deadline: float  # arrival + class TTFT target
    backend: str | None = None
    promoted: bool = False
    rerouted: bool = False
    dispatched_at: float | None = None


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def qwen_token_counter(model: str = "Qwen/Qwen3-8B") -> Callable[[str], int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(model)
    return lambda text: len(tokenizer.encode(text).ids)


def prompt_text(messages: list) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    return "\n".join(parts)


def resolve_class(tenant: TenantConfig, requested: str | None) -> str:
    if requested is None:
        return tenant.default_class
    if requested not in tenant.allowed_classes:
        raise BadRequest(f"class {requested!r} is not allowed for this API key")
    return requested


def analyze(
    body: dict,
    tenant: TenantConfig,
    cfg: GatewayConfig,
    count_tokens: Callable[[str], int],
    now: float,
    request_id: str,
) -> Request:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a non-empty list")
    cls = cfg.class_by_name(resolve_class(tenant, body.get("service_class")))
    raw_max = body.get("max_tokens") or body.get("max_completion_tokens")
    max_tokens = int(raw_max) if raw_max is not None else cls.output_p90
    if max_tokens <= 0:
        raise BadRequest("max_tokens must be positive")
    prompt_tokens = count_tokens(prompt_text(messages)) + CHAT_TEMPLATE_TOKENS_PER_MESSAGE * len(
        messages
    )
    session = body.get("session_id")
    return Request(
        id=request_id,
        tenant=tenant.name,
        sensitive=tenant.sensitive,
        cls=cls,
        rank=cls.rank,
        vllm_priority=cls.vllm_priority,
        body={k: v for k, v in body.items() if k not in GATEWAY_FIELDS},
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        output_estimate=min(max_tokens, cls.output_p90),
        task_type=str(body.get("task_type") or "chat"),
        session_id=str(session) if session else None,
        stream=bool(body.get("stream", False)),
        arrival=now,
        deadline=now + cls.ttft_target_s,
    )
