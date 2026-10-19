"""Streams one request to one OpenAI-compatible backend and classifies failures for the engine.

The gateway always streams upstream, even for non-streaming clients, so TTFT is measured the
same way for every request. Cancelling the calling task closes the upstream connection, which
makes vLLM abort the request and free its KV blocks."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from gateway.analyzer import Request
from gateway.config import BackendConfig

BACKPRESSURE_STATUSES = frozenset({429, 503})


class Backpressure(Exception):
    def __init__(self, status: int, retry_after: float | None) -> None:
        super().__init__(f"backend busy ({status})")
        self.status = status
        self.retry_after = retry_after


class UpstreamError(Exception):
    def __init__(self, status: int | None, detail: str) -> None:
        super().__init__(f"{status}: {detail}" if status else detail)
        self.status = status


@dataclass
class StreamResult:
    output_tokens: int
    prompt_tokens: int | None
    first_token_at: float | None
    finished_at: float
    finish_reason: str | None


def build_body(req: Request, backend: BackendConfig) -> dict:
    body = {**req.body, **backend.extra_body}
    body.pop("max_completion_tokens", None)
    body.pop("priority", None)
    body.update(
        model=backend.model,
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=req.max_tokens,
    )
    if backend.send_priority:
        body["priority"] = req.vllm_priority
    return body


def count_chunk_tokens(chunk: dict) -> int:
    """vLLM and most providers stream about one token per content delta; usage corrects it."""
    count = 0
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content"):
            count += 1
    return count


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


async def stream_chat(
    client: httpx.AsyncClient,
    backend: BackendConfig,
    body: dict,
    on_chunk: Callable[[dict], Awaitable[None]],
    clock: Callable[[], float],
) -> StreamResult:
    key = backend.api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    counted, usage, first_token_at, finish = 0, None, None, None
    try:
        async with client.stream(
            "POST", f"{backend.base_url}/chat/completions", json=body, headers=headers
        ) as response:
            if response.status_code in BACKPRESSURE_STATUSES:
                raise Backpressure(response.status_code, _retry_after(response))
            if response.status_code >= 400:
                detail = (await response.aread()).decode(errors="replace")[:500]
                raise UpstreamError(response.status_code, detail)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise UpstreamError(None, json.dumps(chunk["error"])[:500])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                n = count_chunk_tokens(chunk)
                if n and first_token_at is None:
                    first_token_at = clock()
                counted += n
                for choice in chunk.get("choices") or []:
                    finish = choice.get("finish_reason") or finish
                await on_chunk(chunk)
    except httpx.TransportError as exc:
        raise UpstreamError(None, f"{type(exc).__name__}: {exc}") from exc
    output_tokens = usage.get("completion_tokens", counted) if usage else counted
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    return StreamResult(output_tokens, prompt_tokens, first_token_at, clock(), finish)
