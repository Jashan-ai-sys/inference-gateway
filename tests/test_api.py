import asyncio
from contextlib import asynccontextmanager

import httpx

from gateway.api import create_app
from gateway.engine import Engine
from tests.conftest import (
    mock_transport,
    only_backends,
    sse_response,
    with_backend,
    with_class,
)

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


async def test_rejects_bare_key_without_bearer_scheme(cfg):
    async with gateway(cfg) as (client, _):
        response = await client.post(
            "/v1/chat/completions", json=BODY, headers={"Authorization": "key-a"}
        )
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


async def test_metrics_and_health(cfg, monkeypatch):
    monkeypatch.setenv("GW_METRICS_TOKEN", "ops-token")
    async with gateway(only_backends(cfg, "sim-strong")) as (client, _):
        await client.post("/v1/chat/completions", json=BODY, headers=AUTH_A)
        metrics = await client.get("/metrics", headers={"Authorization": "Bearer ops-token"})
        tenant_key = await client.get("/metrics", headers=AUTH_A)
        anonymous = await client.get("/metrics")
        health = await client.get("/healthz")
    sample = 'gateway_requests_total{backend="sim-strong",cls="standard",outcome="ok"} 1.0'
    assert sample in metrics.text  # exposition sorts labels alphabetically
    assert tenant_key.status_code == 401 and anonymous.status_code == 401
    assert health.json() == {"status": "ok"}


async def test_metrics_disabled_without_token(cfg, monkeypatch):
    monkeypatch.delenv("GW_METRICS_TOKEN", raising=False)
    async with gateway(cfg) as (client, _):
        response = await client.get("/metrics", headers=AUTH_A)
    assert response.status_code == 404


async def test_streaming_upstream_failure_maps_to_status(cfg):
    async with gateway(only_backends(cfg, "sim-strong"), chat=lambda r: httpx.Response(500)) as (
        client,
        _,
    ):
        response = await client.post(
            "/v1/chat/completions", json={**BODY, "stream": True}, headers=AUTH_A
        )
    assert response.status_code == 502


async def test_streaming_deadline_expiry_maps_to_503(cfg):
    cfg = with_class(
        with_backend(only_backends(cfg, "sim-strong"), "sim-strong", max_in_flight=0),
        "interactive",
        ttft_target_s=0.3,
    )
    async with gateway(cfg) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={**BODY, "stream": True},
            headers={"Authorization": "Bearer key-b"},
        )
    assert response.status_code == 503


async def test_backend_header_reports_the_backend_after_a_reroute(cfg):
    """sim-strong wins routing but cannot admit; at aging_fraction of the 1 s target the
    request moves to sim-small, and the headers must name sim-small."""
    cfg = only_backends(cfg, "sim-strong", "sim-small")
    cfg = with_backend(cfg, "sim-strong", max_in_flight=0, base_ttft_s=0.1)
    cfg = with_backend(cfg, "sim-small", base_ttft_s=0.3, prefill_tokens_per_s=4000)
    interactive = {"Authorization": "Bearer key-b"}
    async with gateway(cfg) as (client, engine):
        collected, streamed = await asyncio.gather(
            client.post("/v1/chat/completions", json=BODY, headers=interactive),
            client.post("/v1/chat/completions", json={**BODY, "stream": True}, headers=interactive),
        )
        reroutes = engine.metrics.registry.get_sample_value(
            "gateway_reroutes_total", {"cls": "interactive", "backend": "sim-small"}
        )
    assert reroutes == 2
    assert collected.status_code == streamed.status_code == 200
    assert collected.headers["x-gateway-backend"] == "sim-small"
    assert streamed.headers["x-gateway-backend"] == "sim-small"


async def test_pre_chunk_errors_carry_gateway_headers(cfg):
    async with gateway(only_backends(cfg, "sim-strong"), chat=lambda r: httpx.Response(500)) as (
        client,
        _,
    ):
        collected = await client.post("/v1/chat/completions", json=BODY, headers=AUTH_A)
        streamed = await client.post(
            "/v1/chat/completions", json={**BODY, "stream": True}, headers=AUTH_A
        )
    for response in (collected, streamed):
        assert response.status_code == 502
        assert response.headers["x-gateway-backend"] == "sim-strong"
        assert response.headers["x-gateway-class"] == "standard"


async def test_no_backend_capacity_429_has_retry_after_and_permanent_400_does_not(cfg):
    cfg = only_backends(cfg, "sim-strong")
    too_long = {**BODY, "messages": [{"role": "user", "content": "x" * 40000}]}
    async with gateway(cfg) as (client, engine):
        permanent = await client.post("/v1/chat/completions", json=too_long, headers=AUTH_A)
        engine.health["sim-strong"].open_until = engine.clock() + 100
        capacity = await client.post("/v1/chat/completions", json=BODY, headers=AUTH_A)
    assert permanent.status_code == 400 and "retry-after" not in permanent.headers
    assert capacity.status_code == 429 and capacity.headers["retry-after"] == "1"
