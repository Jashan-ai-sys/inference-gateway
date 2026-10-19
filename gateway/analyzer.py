"""Turns an incoming chat request into a scheduling Request: tenant, class, deadline, tokens."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from gateway.config import ClassConfig, GatewayConfig, TenantConfig

# Client fields that go upstream. Anything else (model, priority, stream, max_tokens, ...) is
# dropped: the dispatcher sets those, or the gateway consumes them. Tool calls and logprobs are
# dropped too until non-streaming responses reassemble them (out of v1 scope).
FORWARDED_FIELDS = frozenset(
    {
        "messages",
        "temperature",
        "top_p",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "response_format",
        "user",
    }
)
MESSAGE_FIELDS = frozenset({"role", "content"})
MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant"})
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
    body: dict  # what is forwarded upstream (allowlisted fields only)
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
    """All forwarded message text. Anything the gateway cannot count is rejected, so admission,
    fairness and rate-limit charges cover everything sent upstream (v1 is text-only)."""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or set(message) - MESSAGE_FIELDS:
            raise BadRequest("each message must be an object with only role and content")
        if message.get("role") not in MESSAGE_ROLES:
            raise BadRequest(f"message role must be one of {sorted(MESSAGE_ROLES)}")
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if (
                    not isinstance(part, dict)
                    or set(part) != {"type", "text"}
                    or part["type"] != "text"
                    or not isinstance(part["text"], str)
                ):
                    raise BadRequest("only text content parts are supported")
                parts.append(part["text"])
        else:
            raise BadRequest("message content must be a string or a list of text parts")
    return "\n".join(parts)


def resolve_class(tenant: TenantConfig, requested: str | None) -> str:
    if requested is None:
        return tenant.default_class
    if requested not in tenant.allowed_classes:
        raise BadRequest(f"class {requested!r} is not allowed for this API key")
    return requested


def _max_tokens(body: dict, cls: ClassConfig) -> int:
    raw = body.get("max_completion_tokens")
    if raw is None:
        raw = body.get("max_tokens")
    if raw is None:
        return cls.output_p90
    if type(raw) is not int or raw <= 0:
        raise BadRequest("max_tokens must be a positive integer")
    return raw


def _check_scalars(body: dict) -> None:
    n = body.get("n")
    if n is not None and (type(n) is not int or n != 1):
        raise BadRequest("n must be 1")
    if not isinstance(body.get("stream", False), bool):
        raise BadRequest("stream must be a boolean")
    if body.get("service_class") is not None and not isinstance(body["service_class"], str):
        raise BadRequest("service_class must be a string")


def analyze(
    body: dict,
    tenant: TenantConfig,
    cfg: GatewayConfig,
    count_tokens: Callable[[str], int],
    now: float,
    request_id: str,
) -> Request:
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a non-empty list")
    _check_scalars(body)
    cls = cfg.class_by_name(resolve_class(tenant, body.get("service_class")))
    max_tokens = _max_tokens(body, cls)
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
        body={k: v for k, v in body.items() if k in FORWARDED_FIELDS},
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        output_estimate=min(max_tokens, cls.output_p90),
        task_type=str(body.get("task_type") or "chat"),
        session_id=str(session) if session else None,
        stream=body.get("stream", False),
        arrival=now,
        deadline=now + cls.ttft_target_s,
    )
