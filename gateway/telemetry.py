"""Backend health: vLLM /metrics polling, latency and error EWMAs, and a circuit breaker."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import httpx

from gateway.config import BackendConfig

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
_GAUGES = frozenset(
    ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc")
)


def _finite(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


@dataclass
class EngineStats:
    running: float = 0.0
    waiting: float = 0.0
    kv_usage: float = 0.0  # 0-1 fraction


def parse_vllm_metrics(text: str) -> EngineStats:
    """Read the three gauges routing needs; sums across engines (data-parallel ranks)."""
    stats = EngineStats()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        name = match.group(1)
        if name not in _GAUGES:
            continue
        value = _finite(match.group(3))
        if value is None:
            continue
        if name == "vllm:num_requests_running":
            stats.running += value
        elif name == "vllm:num_requests_waiting":
            stats.waiting += value
        elif name == "vllm:kv_cache_usage_perc":
            stats.kv_usage = max(stats.kv_usage, value)
    return stats


class Ewma:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> None:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value


class BackendHealth:
    def __init__(
        self, backend: BackendConfig, failure_threshold: int = 3, open_s: float = 10.0
    ) -> None:
        self.backend = backend
        self.failure_threshold = failure_threshold
        self.open_s = open_s
        self.ttft = Ewma()
        self.errors = Ewma()
        self.consecutive_failures = 0
        self.open_until = 0.0
        self.stats = EngineStats()

    def base_ttft(self) -> float:
        """Dispatch-to-first-token minus estimated prefill, measured; the config prior until the
        first success."""
        return self.ttft.value if self.ttft.value is not None else self.backend.base_ttft_s

    def error_rate(self) -> float:
        return self.errors.value or 0.0

    def available(self, now: float) -> bool:
        return now >= self.open_until

    def record_success(self, ttft_s: float) -> None:
        self.ttft.update(ttft_s)
        self.errors.update(0.0)
        self.consecutive_failures = 0

    def record_failure(self, now: float) -> None:
        self.errors.update(1.0)
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.open_until = now + self.open_s
            self.consecutive_failures = 0


async def poll_once(
    client: httpx.AsyncClient, backend: BackendConfig, health: BackendHealth
) -> None:
    if not backend.metrics_url:
        return
    try:
        response = await client.get(backend.metrics_url, timeout=2.0)
        response.raise_for_status()
    except (httpx.HTTPError, httpx.InvalidURL):
        return
    health.stats = parse_vllm_metrics(response.text)
