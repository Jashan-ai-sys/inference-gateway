import asyncio
import json

import httpx
import pytest

from gateway.dispatcher import Backpressure, UpstreamError, build_body, stream_chat
from tests.conftest import FakeClock, SlowStream, make_request, sse_response, with_backend


async def _noop(chunk):
    return None


def test_build_body_sets_upstream_fields(cfg):
    backend = cfg.backend_by_name("sim-strong")
    req = make_request(cfg, cls="interactive", max_tokens=42)
    req.body["max_completion_tokens"] = 7
    body = build_body(req, backend)
    assert body["model"] == "Qwen/Qwen3-8B-AWQ"
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 42 and "max_completion_tokens" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "priority" not in body


def test_priority_only_for_backends_that_want_it(cfg):
    cfg = with_backend(cfg, "sim-strong", send_priority=True)
    body = build_body(make_request(cfg, cls="background"), cfg.backend_by_name("sim-strong"))
    assert body["priority"] == 2


async def test_stream_counts_tokens_and_prefers_reported_usage(cfg):
    clock = FakeClock(5.0)
    seen = []

    async def on_chunk(chunk):
        seen.append(chunk)

    usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    transport = httpx.MockTransport(lambda r: sse_response(("a", "b"), usage=usage))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await stream_chat(client, cfg.backend_by_name("sim-strong"), {}, on_chunk, clock)
    assert result.output_tokens == 3  # usage wins over the 2 counted deltas
    assert result.prompt_tokens == 12
    assert result.first_token_at == 5.0
    assert result.finish_reason == "stop"
    assert len(seen) == 4


async def test_stream_without_usage_uses_counted_tokens(cfg):
    transport = httpx.MockTransport(lambda r: sse_response(("a", "b", "c")))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await stream_chat(
            client, cfg.backend_by_name("sim-strong"), {}, _noop, FakeClock()
        )
    assert result.output_tokens == 3 and result.prompt_tokens is None


async def test_sends_bearer_key_for_hosted_backends(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    captured = {}

    def handler(request):
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return sse_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await stream_chat(
            client, cfg.backend_by_name("groq-gpt-oss-20b"), {"x": 1}, _noop, FakeClock()
        )
    assert captured == {"auth": "Bearer secret", "body": {"x": 1}}


@pytest.mark.parametrize("status", [429, 503])
async def test_busy_statuses_are_backpressure(cfg, status):
    transport = httpx.MockTransport(lambda r: httpx.Response(status, headers={"Retry-After": "2"}))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(Backpressure) as err:
            await stream_chat(client, cfg.backend_by_name("sim-strong"), {}, _noop, FakeClock())
    assert err.value.retry_after == 2.0


async def test_server_error_and_connection_error_are_upstream_errors(cfg):
    backend = cfg.backend_by_name("sim-strong")
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(UpstreamError) as err:
            await stream_chat(client, backend, {}, _noop, FakeClock())
    assert err.value.status == 500

    def refuse(request):
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        with pytest.raises(UpstreamError):
            await stream_chat(client, backend, {}, _noop, FakeClock())


async def test_cancelling_closes_the_upstream_stream(cfg):
    stream = SlowStream(n_tokens=1000, delay=0.01)
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        task = asyncio.create_task(
            stream_chat(client, cfg.backend_by_name("sim-strong"), {}, _noop, FakeClock())
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert stream.closed


def _raw(body: bytes) -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize(
    "body",
    [b"data: not-json\n\n", b"data: [1,2]\n\n", b"data: null\n\n", b'data: {"choices": [\n\n'],
)
async def test_malformed_stream_chunk_is_upstream_error(cfg, body):
    transport = httpx.MockTransport(lambda r: _raw(body))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(UpstreamError):
            await stream_chat(client, cfg.backend_by_name("sim-strong"), {}, _noop, FakeClock())


async def test_in_stream_error_chunk_is_upstream_error(cfg):
    transport = httpx.MockTransport(lambda r: _raw(b'data: {"error": {"message": "boom"}}\n\n'))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(UpstreamError) as err:
            await stream_chat(client, cfg.backend_by_name("sim-strong"), {}, _noop, FakeClock())
    assert "boom" in str(err.value)


@pytest.mark.parametrize("usage", [{"completion_tokens": None}, "junk"])
async def test_unusable_usage_falls_back_to_counted_tokens(cfg, usage):
    transport = httpx.MockTransport(lambda r: sse_response(("a", "b"), usage=usage))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await stream_chat(
            client, cfg.backend_by_name("sim-strong"), {}, _noop, FakeClock()
        )
    assert result.output_tokens == 2 and result.prompt_tokens is None


async def test_httpx_request_errors_are_upstream_errors(cfg):
    def broken(request):
        raise httpx.DecodingError("bad encoding")

    async with httpx.AsyncClient(transport=httpx.MockTransport(broken)) as client:
        with pytest.raises(UpstreamError):
            await stream_chat(client, cfg.backend_by_name("sim-strong"), {}, _noop, FakeClock())
