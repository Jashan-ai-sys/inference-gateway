"""Needs `make sim-up`. Functional checks only: the simulator is FIFO, so no latency claims."""

import asyncio
import json

import httpx
import pytest

from gateway.api import create_app
from gateway.engine import Engine
from gateway.telemetry import parse_vllm_metrics
from tests.conftest import make_request, only_backends

pytestmark = pytest.mark.integration
AUTH = {"Authorization": "Bearer key-a"}


async def _running(port: int) -> float:
    async with httpx.AsyncClient() as client:
        text = (await client.get(f"http://localhost:{port}/metrics")).text
    return parse_vllm_metrics(text).running


async def _gateway(cfg):
    engine = Engine(cfg, httpx.AsyncClient(timeout=httpx.Timeout(30.0)))
    app = create_app(cfg, engine)
    await engine.start()
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw", timeout=30.0
    )
    return client, engine


async def _close(client, engine) -> None:
    await client.aclose()
    await engine.stop()
    await engine.client.aclose()


async def test_simulator_exposes_vllm_metric_names():
    async with httpx.AsyncClient() as client:
        text = (await client.get("http://localhost:8001/metrics")).text
    assert "vllm:num_requests_running" in text
    assert "vllm:num_requests_waiting" in text


async def test_gateway_streams_through_simulator(cfg):
    client, engine = await _gateway(only_backends(cfg, "sim-strong"))
    try:
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20, "stream": True}
        response = await client.post("/v1/chat/completions", json=body, headers=AUTH)
    finally:
        await _close(client, engine)
    lines = [line for line in response.text.splitlines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    assert any(json.loads(line[5:]).get("choices") for line in lines[:-1])


@pytest.mark.xfail(
    raises=AssertionError,
    reason="llm-d-inference-sim does not abort on disconnect; checked on real vLLM in Plan 3",
)
async def test_cancel_frees_the_simulator_slot(cfg):
    cfg = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    engine = Engine(cfg, httpx.AsyncClient(timeout=httpx.Timeout(60.0)))
    try:
        job = engine.submit(make_request(cfg, max_tokens=2000, now=engine.clock()))
        await asyncio.wait_for(job.out.get(), 10)  # first chunk
        assert await _running(8001) >= 1
        engine.cancel(job)
        for _ in range(50):
            if await _running(8001) == 0:
                break
            await asyncio.sleep(0.1)
    finally:
        await engine.stop()
        await engine.client.aclose()
    assert await _running(8001) == 0


async def test_gateway_absorbs_simulator_429s(cfg):
    """sim-small has 1 worker and a 1-slot queue: a burst makes it return 429. In gateway mode
    the engine treats 429 as backpressure, requeues, and every request still completes."""
    cfg = only_backends(cfg, "sim-small")
    client, engine = await _gateway(cfg)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    try:
        responses = await asyncio.gather(
            *[client.post("/v1/chat/completions", json=body, headers=AUTH) for _ in range(6)]
        )
        saturated_until = engine.schedulers["sim-small"].saturated_until
    finally:
        await _close(client, engine)
    assert saturated_until > 0, "the burst never drew a 429 from sim-small"
    assert [r.status_code for r in responses] == [200] * 6
