import httpx
import pytest

from gateway.telemetry import BackendHealth, Ewma, parse_vllm_metrics, poll_once

SAMPLE = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_running{engine="1",model_name="m"} 2.0
vllm:num_requests_waiting{engine="0",model_name="m"} 4.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.61
vllm:prompt_tokens_total{engine="0",model_name="m"} 1.234e+03
"""


def test_parse_sums_requests_and_takes_max_kv_usage():
    stats = parse_vllm_metrics(SAMPLE)
    assert (stats.running, stats.waiting) == (5.0, 4.0)
    assert stats.kv_usage == pytest.approx(0.61)


def test_parse_missing_metrics_are_zero():
    stats = parse_vllm_metrics("other_metric 1\n")
    assert (stats.running, stats.waiting, stats.kv_usage) == (0.0, 0.0, 0.0)


def test_ewma():
    e = Ewma(alpha=0.5)
    assert e.value is None
    e.update(1.0)
    e.update(3.0)
    assert e.value == 2.0


def test_base_ttft_falls_back_to_config_prior(cfg):
    h = BackendHealth(cfg.backend_by_name("sim-strong"))
    assert h.base_ttft() == 0.2
    h.record_success(0.5)
    assert h.base_ttft() == 0.5


def test_circuit_breaker_opens_after_consecutive_failures(cfg):
    h = BackendHealth(cfg.backend_by_name("sim-strong"), failure_threshold=3, open_s=10.0)
    h.record_failure(0.0)
    h.record_failure(0.0)
    assert h.available(0.0)
    h.record_failure(1.0)
    assert not h.available(5.0)
    assert h.available(11.0)
    assert h.error_rate() > 0


def test_success_resets_failure_streak(cfg):
    h = BackendHealth(cfg.backend_by_name("sim-strong"), failure_threshold=2)
    h.record_failure(0.0)
    h.record_success(0.1)
    h.record_failure(0.0)
    assert h.available(0.0)


async def test_poll_once_updates_stats_and_survives_errors(cfg):
    backend = cfg.backend_by_name("sim-strong")
    h = BackendHealth(backend)
    ok = httpx.MockTransport(lambda r: httpx.Response(200, text=SAMPLE))
    async with httpx.AsyncClient(transport=ok) as client:
        await poll_once(client, backend, h)
    assert h.stats.running == 5.0
    broken = httpx.MockTransport(lambda r: httpx.Response(500))
    async with httpx.AsyncClient(transport=broken) as client:
        await poll_once(client, backend, h)
    assert h.stats.running == 5.0  # last good reading kept
