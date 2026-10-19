import asyncio
import io
import json

import httpx
import pytest

from gateway.engine import Done, Engine, Failed
from gateway.scheduler import QueueFull
from tests.conftest import (
    FakeClock,
    SlowStream,
    drain,
    make_request,
    mock_transport,
    only_backends,
    sse_response,
    with_backend,
    with_class,
)

USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


async def test_passthrough_streams_and_accounts(cfg):
    cfg = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    log = io.StringIO()
    async with httpx.AsyncClient(
        transport=mock_transport(lambda r: sse_response(usage=USAGE))
    ) as c:
        engine = Engine(cfg, c, decision_log=log)
        job = engine.submit(make_request(cfg, prompt=100, session="s1", now=engine.clock()))
        items = await drain(job)
    assert isinstance(items[-1], Done) and items[-1].result.output_tokens == 2
    assert engine.vtc.counter("tenant-a") == 100 + 2 * 2
    assert engine.sessions.get("tenant-a", "s1").tokens == 12
    assert engine.jobs == {}
    assert json.loads(log.getvalue())["backend"] == "sim-strong"
    sample = engine.metrics.registry.get_sample_value(
        "gateway_requests_total", {"cls": "standard", "backend": "sim-strong", "outcome": "ok"}
    )
    assert sample == 1.0


async def test_quiet_tenant_is_not_starved_by_noisy_tenant(cfg):
    cfg = only_backends(with_backend(cfg, "sim-strong", max_in_flight=1), "sim-strong")
    order = []

    async def handler(request):
        order.append(json.loads(request.content)["messages"][0]["content"])
        await asyncio.sleep(0.01)
        return sse_response(("x",) * 5, usage={"prompt_tokens": 10, "completion_tokens": 5})

    async with httpx.AsyncClient(transport=mock_transport(handler)) as c:
        engine = Engine(cfg, c)
        await engine.start()
        jobs = [
            engine.submit(make_request(cfg, tenant="tenant-a", content="a", now=engine.clock()))
            for _ in range(6)
        ]
        jobs += [
            engine.submit(make_request(cfg, tenant="tenant-b", content="b", now=engine.clock()))
            for _ in range(2)
        ]
        for job in jobs:
            await drain(job)
        await engine.stop()
    assert order[:4] == ["a", "b", "a", "b"]


async def test_backpressure_requeues_and_retries(cfg):
    cfg = only_backends(cfg, "sim-strong")
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, headers={"Retry-After": "0.05"})
        return sse_response(usage=USAGE)

    async with httpx.AsyncClient(transport=mock_transport(handler)) as c:
        engine = Engine(cfg, c)
        await engine.start()
        items = await drain(engine.submit(make_request(cfg, now=engine.clock())))
        await engine.stop()
    assert isinstance(items[-1], Done) and len(calls) == 2


async def test_upstream_error_fails_job_and_counts_failure(cfg):
    cfg = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    async with httpx.AsyncClient(transport=mock_transport(lambda r: httpx.Response(500))) as c:
        engine = Engine(cfg, c)
        items = await drain(engine.submit(make_request(cfg, now=engine.clock())))
    assert isinstance(items[-1], Failed) and items[-1].status == 502
    assert engine.health["sim-strong"].consecutive_failures == 1


async def test_queued_request_expires_with_503(cfg):
    cfg = with_class(cfg, "interactive", ttft_target_s=0.3)
    cfg = only_backends(with_backend(cfg, "sim-strong", max_in_flight=1), "sim-strong")
    slow = SlowStream(n_tokens=1000, delay=0.01)
    transport = mock_transport(
        lambda r: httpx.Response(200, stream=slow, headers={"content-type": "text/event-stream"})
    )
    async with httpx.AsyncClient(transport=transport) as c:
        engine = Engine(cfg, c)
        await engine.start()
        busy = engine.submit(make_request(cfg, cls="background", now=engine.clock()))
        await asyncio.sleep(0.05)
        late = engine.submit(make_request(cfg, cls="interactive", now=engine.clock()))
        items = await drain(late, timeout=2.0)
        engine.cancel(busy)
        await engine.stop()
    assert isinstance(items[-1], Failed) and items[-1].status == 503


async def test_cancel_queued_and_running_jobs(cfg):
    cfg = only_backends(with_backend(cfg, "sim-strong", max_in_flight=0), "sim-strong")
    async with httpx.AsyncClient(transport=mock_transport(lambda r: sse_response())) as c:
        engine = Engine(cfg, c)
        queued = engine.submit(make_request(cfg, now=engine.clock()))
        engine.cancel(queued)
        assert engine.schedulers["sim-strong"].queued() == [] and engine.jobs == {}

    passthrough = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    slow = SlowStream(n_tokens=1000, delay=0.01)
    transport = mock_transport(
        lambda r: httpx.Response(200, stream=slow, headers={"content-type": "text/event-stream"})
    )
    async with httpx.AsyncClient(transport=transport) as c:
        engine = Engine(passthrough, c)
        running = engine.submit(make_request(passthrough, now=engine.clock()))
        await running.out.get()  # first chunk arrived
        engine.cancel(running)
        await asyncio.sleep(0.05)
    assert slow.closed and engine.jobs == {}


async def test_at_risk_interactive_request_is_rerouted_once(cfg):
    cfg = only_backends(cfg, "sim-strong", "sim-small")
    cfg = with_backend(cfg, "sim-strong", max_in_flight=0, base_ttft_s=0.1)
    cfg = with_backend(cfg, "sim-small", base_ttft_s=0.3, prefill_tokens_per_s=4000)
    clock = FakeClock()
    async with httpx.AsyncClient(
        transport=mock_transport(lambda r: sse_response(usage=USAGE))
    ) as c:
        engine = Engine(cfg, c, clock=clock)
        job = engine.submit(
            make_request(cfg, tenant="tenant-b", cls="interactive", now=engine.clock())
        )
        assert job.req.backend == "sim-strong"
        clock.advance(0.6)
        engine._tick()
        assert job.req.backend == "sim-small" and job.req.rerouted
        items = await drain(job)
    assert isinstance(items[-1], Done)


def test_queue_full_is_raised_and_nothing_is_left_behind(cfg):
    cfg = only_backends(with_class(cfg, "standard", queue_limit=0), "sim-strong")
    engine = Engine(cfg, httpx.AsyncClient())
    with pytest.raises(QueueFull):
        engine.submit(make_request(cfg, now=engine.clock()))
    assert engine.jobs == {}


async def test_unexpected_upstream_exception_still_fails_the_job(cfg, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr("gateway.engine.stream_chat", boom)
    cfg = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    async with httpx.AsyncClient(transport=mock_transport(lambda r: sse_response())) as c:
        engine = Engine(cfg, c)
        items = await drain(engine.submit(make_request(cfg, now=engine.clock())))
    assert isinstance(items[-1], Failed) and items[-1].status == 502
    assert engine.health["sim-strong"].consecutive_failures == 1
    assert engine.jobs == {}


async def test_cancel_before_task_starts_releases_passthrough_reservation(cfg):
    cfg = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    async with httpx.AsyncClient(transport=mock_transport(lambda r: sse_response())) as c:
        engine = Engine(cfg, c)
        engine.cancel(engine.submit(make_request(cfg, now=engine.clock())))
        await asyncio.sleep(0.01)
    assert engine.jobs == {} and engine.schedulers["sim-strong"].in_flight == {}


async def test_cancel_before_task_starts_releases_gateway_reservation(cfg):
    cfg = only_backends(cfg, "sim-strong")
    async with httpx.AsyncClient(transport=mock_transport(lambda r: sse_response())) as c:
        engine = Engine(cfg, c)
        job = engine.submit(make_request(cfg, now=engine.clock()))
        engine._tick()
        engine.cancel(job)
        await asyncio.sleep(0.01)
    assert engine.jobs == {} and engine.schedulers["sim-strong"].in_flight == {}


async def test_dispatch_loop_survives_a_failing_tick(cfg, monkeypatch):
    cfg = only_backends(cfg, "sim-strong")
    async with httpx.AsyncClient(transport=mock_transport(lambda r: sse_response())) as c:
        engine = Engine(cfg, c)
        original, calls = engine._tick, []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("disk full")
            original()

        monkeypatch.setattr(engine, "_tick", flaky)
        await engine.start()
        items = await drain(engine.submit(make_request(cfg, now=engine.clock())))
        await engine.stop()
    assert isinstance(items[-1], Done)


async def test_upstream_error_message_does_not_leak_internal_detail(cfg):
    cfg = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    transport = mock_transport(lambda r: httpx.Response(500, text="failed http://10.0.0.5/x"))
    async with httpx.AsyncClient(transport=transport) as c:
        engine = Engine(cfg, c)
        items = await drain(engine.submit(make_request(cfg, now=engine.clock())))
    assert isinstance(items[-1], Failed) and "10.0.0.5" not in items[-1].message


async def test_stop_fails_unfinished_jobs_and_leaves_client_open(cfg):
    cfg = only_backends(with_backend(cfg, "sim-strong", max_in_flight=0), "sim-strong")
    async with httpx.AsyncClient(transport=mock_transport(lambda r: sse_response())) as c:
        engine = Engine(cfg, c)
        job = engine.submit(make_request(cfg, now=engine.clock()))
        await engine.stop()
        assert not c.is_closed
    assert job.out.get_nowait() == Failed(503, "gateway shutting down")


async def test_prompt_is_charged_once_across_backpressure_retries(cfg):
    cfg = only_backends(cfg, "sim-strong")
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, headers={"Retry-After": "0.02"})
        return sse_response(usage=USAGE)

    async with httpx.AsyncClient(transport=mock_transport(handler)) as c:
        engine = Engine(cfg, c)
        await engine.start()
        await drain(engine.submit(make_request(cfg, prompt=100, now=engine.clock())))
        await engine.stop()
    assert engine.vtc.counter("tenant-a") == 100 + 2 * 2


async def test_queue_full_is_logged_with_error(cfg):
    cfg = only_backends(with_class(cfg, "standard", queue_limit=0), "sim-strong")
    log = io.StringIO()
    engine = Engine(cfg, httpx.AsyncClient(), decision_log=log)
    with pytest.raises(QueueFull):
        engine.submit(make_request(cfg, now=engine.clock()))
    entries = [json.loads(line) for line in log.getvalue().splitlines()]
    assert len(entries) == 1 and entries[0]["error"] == "queue_full"
