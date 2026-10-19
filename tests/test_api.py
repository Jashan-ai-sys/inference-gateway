from contextlib import asynccontextmanager

import httpx

from gateway.api import create_app
from gateway.engine import Engine
from tests.conftest import mock_transport, only_backends, sse_response, with_class

AUTH_A = {"Authorization": "Bearer key-a"}
BODY = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}


@asynccontextmanager
async def gateway(
    cfg, chat=lambda r: sse_response(usage={"prompt_tokens": 5, "completion_tokens": 2})
):
    engine = Engine(cfg, httpx.AsyncClient(transport=mock_transport(chat)))
    app = create_app(cfg, engine)
    await engine.start()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
    try:
        yield client, engine
    finally:
        await client.aclose()
        await engine.stop()
        await engine.client.aclose()


async def test_rejects_unknown_key(cfg):
    async with gateway(cfg) as (client, _):
        response = await client.post("/v1/chat/completions", json=BODY)
    assert response.status_code == 401


async def test_non_streaming_response(cfg):
    async with gateway(only_backends(cfg, "sim-strong")) as (client, _):
        response = await client.post("/v1/chat/completions", json=BODY, headers=AUTH_A)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello world"
    assert data["usage"]["completion_tokens"] == 2
    assert response.headers["x-gateway-backend"] == "sim-strong"


async def test_streaming_response(cfg):
    async with gateway(cfg) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json={**BODY, "stream": True}, headers=AUTH_A
        )
    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]" and len(lines) >= 3


async def test_class_comes_from_key_not_header(cfg):
    spoof = {"Authorization": "Bearer key-c", "X-Service-Class": "interactive"}
    async with gateway(cfg) as (client, _):
        ok = await client.post("/v1/chat/completions", json=BODY, headers=spoof)
        bad = await client.post(
            "/v1/chat/completions", json={**BODY, "service_class": "interactive"}, headers=spoof
        )
    assert ok.headers["x-gateway-class"] == "background"
    assert bad.status_code == 400


async def test_queue_full_is_429_with_retry_after(cfg):
    cfg = only_backends(with_class(cfg, "standard", queue_limit=0), "sim-strong")
    async with gateway(cfg) as (client, _):
        response = await client.post("/v1/chat/completions", json=BODY, headers=AUTH_A)
    assert response.status_code == 429 and response.headers["retry-after"] == "1"


async def test_bad_body_is_400(cfg):
    async with gateway(cfg) as (client, _):
        response = await client.post("/v1/chat/completions", content=b"not json", headers=AUTH_A)
    assert response.status_code == 400


async def test_upstream_failure_maps_to_status(cfg):
    async with gateway(only_backends(cfg, "sim-strong"), chat=lambda r: httpx.Response(500)) as (
        client,
        _,
    ):
        response = await client.post("/v1/chat/completions", json=BODY, headers=AUTH_A)
    assert response.status_code == 502


async def test_metrics_and_health(cfg):
    async with gateway(only_backends(cfg, "sim-strong")) as (client, _):
        await client.post("/v1/chat/completions", json=BODY, headers=AUTH_A)
        metrics = await client.get("/metrics")
        health = await client.get("/healthz")
    sample = 'gateway_requests_total{backend="sim-strong",cls="standard",outcome="ok"} 1.0'
    assert sample in metrics.text  # exposition sorts labels alphabetically
    assert health.json() == {"status": "ok", "backends": ["sim-strong"]}
