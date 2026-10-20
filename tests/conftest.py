import asyncio
import inspect
import itertools
import json
from pathlib import Path

import httpx
import pytest

from gateway.config import GatewayConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = ROOT / "config" / "sim.yaml"
_ids = itertools.count()


@pytest.fixture
def cfg(monkeypatch) -> GatewayConfig:
    for letter in "abc":
        monkeypatch.setenv(f"GW_KEY_{letter.upper()}", f"key-{letter}")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return load_config(SIM_CONFIG)


def with_backend(cfg: GatewayConfig, name: str, **changes) -> GatewayConfig:
    backends = [b.model_copy(update=changes) if b.name == name else b for b in cfg.backends]
    return cfg.model_copy(update={"backends": backends})


def with_class(cfg: GatewayConfig, name: str, **changes) -> GatewayConfig:
    classes = [c.model_copy(update=changes) if c.name == name else c for c in cfg.classes]
    return cfg.model_copy(update={"classes": classes})


def only_backends(cfg: GatewayConfig, *names: str) -> GatewayConfig:
    return cfg.model_copy(update={"backends": [b for b in cfg.backends if b.name in names]})


def make_request(
    cfg: GatewayConfig,
    *,
    tenant: str = "tenant-a",
    cls: str = "standard",
    prompt: int = 100,
    max_tokens: int = 50,
    now: float = 0.0,
    task: str = "chat",
    session: str | None = None,
    sensitive: bool = False,
    content: str = "hi",
):
    from gateway.analyzer import Request  # lazy: conftest is used from Task 2, analyzer from Task 3

    c = cfg.class_by_name(cls)
    return Request(
        id=f"r{next(_ids)}",
        tenant=tenant,
        sensitive=sensitive,
        cls=c,
        rank=c.rank,
        vllm_priority=c.vllm_priority,
        body={"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens},
        prompt_tokens=prompt,
        max_tokens=max_tokens,
        output_estimate=min(max_tokens, c.output_p90),
        task_type=task,
        session_id=session,
        stream=True,
        arrival=now,
        deadline=now + c.ttft_target_s,
    )


def chat_chunk(content: str | None = None, finish: str | None = None, usage: dict | None = None):
    chunk: dict = {"id": "c", "object": "chat.completion.chunk", "model": "m", "choices": []}
    if content is not None or finish is not None:
        delta = {"content": content} if content else {}
        chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
    if usage:
        chunk["usage"] = usage
    return chunk


def sse_bytes(chunks: list[dict], done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


def sse_response(tokens=("Hello", " world"), usage: dict | None = None) -> httpx.Response:
    chunks = [chat_chunk(t) for t in tokens] + [chat_chunk(finish="stop")]
    if usage:
        chunks.append(chat_chunk(usage=usage))
    return httpx.Response(
        200, content=sse_bytes(chunks), headers={"content-type": "text/event-stream"}
    )


class SlowStream(httpx.AsyncByteStream):
    """An upstream that keeps streaming tokens and records whether the gateway closed it."""

    def __init__(self, n_tokens: int = 1000, delay: float = 0.01) -> None:
        self.n_tokens = n_tokens
        self.delay = delay
        self.closed = False

    async def __aiter__(self):
        for i in range(self.n_tokens):
            await asyncio.sleep(self.delay)
            yield sse_bytes([chat_chunk(f"t{i}")], done=False)
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def mock_transport(chat) -> httpx.MockTransport:
    """Routes /metrics polls to an empty 200 and everything else to `chat` (sync or async)."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/metrics"):
            return httpx.Response(200, text="")
        result = chat(request)
        return await result if inspect.isawaitable(result) else result

    return httpx.MockTransport(handler)


async def drain(job, timeout: float = 5.0) -> list:
    """Collect a job's output until Done or Failed."""
    from gateway.engine import Done, Failed

    items = []
    while True:
        item = await asyncio.wait_for(job.out.get(), timeout)
        items.append(item)
        if isinstance(item, Done | Failed):
            return items
