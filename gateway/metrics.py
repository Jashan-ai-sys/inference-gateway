"""Gateway Prometheus metrics. Each Engine owns a registry so tests stay isolated."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from gateway.analyzer import Request
from gateway.config import BackendConfig
from gateway.dispatcher import StreamResult

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)
TPOT_BUCKETS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28)


class GatewayMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        by_cls = ["cls", "backend"]
        self.requests = Counter(
            "gateway_requests", "Finished requests", ["cls", "backend", "outcome"], registry=r
        )
        self.rejections = Counter(
            "gateway_rejections", "Rejected before dispatch", ["cls", "reason"], registry=r
        )
        self.goodput = Counter(
            "gateway_goodput_requests", "Requests meeting TTFT and TPOT targets", by_cls, registry=r
        )
        self.queue_wait = Histogram(
            "gateway_queue_wait_seconds",
            "Time queued in the gateway",
            by_cls,
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "gateway_ttft_seconds",
            "Arrival to first token",
            by_cls,
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.engine_ttft = Histogram(
            "gateway_engine_ttft_seconds",
            "Dispatch to first token",
            by_cls,
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.tpot = Histogram(
            "gateway_tpot_seconds",
            "Time per output token",
            by_cls,
            buckets=TPOT_BUCKETS,
            registry=r,
        )
        self.tokens = Counter("gateway_tokens", "Tokens", ["backend", "kind"], registry=r)
        self.cost = Counter(
            "gateway_cost_usd", "Estimated spend", ["backend", "tenant"], registry=r
        )
        self.queue_depth = Gauge("gateway_queue_depth", "Queued requests", by_cls, registry=r)
        self.reroutes = Counter(
            "gateway_reroutes", "Re-routes of at-risk requests", ["cls"], registry=r
        )

    def record_finished(self, req: Request, result: StreamResult, backend: BackendConfig) -> bool:
        """Record one successful request; returns whether it met both SLO targets (goodput)."""
        cls, name = req.cls.name, backend.name
        prompt_tokens = result.prompt_tokens or req.prompt_tokens
        self.requests.labels(cls, name, "ok").inc()
        self.tokens.labels(name, "prompt").inc(prompt_tokens)
        self.tokens.labels(name, "output").inc(result.output_tokens)
        self.cost.labels(name, req.tenant).inc(backend.cost(prompt_tokens, result.output_tokens))
        self.queue_wait.labels(cls, name).observe(req.dispatched_at - req.arrival)
        if result.first_token_at is None:
            return False
        ttft = result.first_token_at - req.arrival
        self.ttft.labels(cls, name).observe(ttft)
        self.engine_ttft.labels(cls, name).observe(result.first_token_at - req.dispatched_at)
        met = ttft <= req.cls.ttft_target_s
        if result.output_tokens > 1:
            tpot = (result.finished_at - result.first_token_at) / (result.output_tokens - 1)
            self.tpot.labels(cls, name).observe(tpot)
            met = met and tpot <= req.cls.tpot_target_s
        if met:
            self.goodput.labels(cls, name).inc()
        return met
