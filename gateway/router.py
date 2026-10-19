"""Model Router: hard-constraint filter, predicted TTFT from the scheduler's queue state and the
uncached prompt tokens, then a routing policy."""

from __future__ import annotations

from dataclasses import dataclass

from gateway.analyzer import Request
from gateway.config import BackendConfig, GatewayConfig
from gateway.scheduler import BackendScheduler
from gateway.sessions import SessionStore
from gateway.telemetry import BackendHealth

PERMANENT_REJECTIONS = frozenset(
    {"context", "quality", "privacy", "kv_capacity", "rate_limit_size"}
)


@dataclass
class Candidate:
    backend: str
    cost: float
    predicted_ttft: float
    quality: float
    error_rate: float
    uncached_tokens: int
    score: float | None = None


@dataclass
class Decision:
    backend: str
    policy: str
    candidates: list[Candidate]
    rejected: dict[str, str]
    reason: str


class NoBackend(Exception):
    def __init__(self, status: int, reason: str, rejected: dict[str, str]) -> None:
        super().__init__(reason)
        self.status = status
        self.rejected = rejected


def uncached_tokens(req: Request, backend_name: str, sessions: SessionStore) -> int:
    """Prompt tokens the backend must prefill. KV caches are model-specific, so a conversation
    moved to another backend re-prefills everything (what cross-model KV reuse would shrink)."""
    if req.session_id:
        state = sessions.get(req.tenant, req.session_id)
        if state is not None and state.backend == backend_name:
            return max(req.prompt_tokens - state.tokens, 0)
    return req.prompt_tokens


def normalise(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


def route(
    req: Request,
    cfg: GatewayConfig,
    backends: dict[str, BackendConfig],
    schedulers: dict[str, BackendScheduler],
    health: dict[str, BackendHealth],
    sessions: SessionStore,
    now: float,
    exclude: frozenset[str] = frozenset(),
) -> Decision:
    candidates: list[Candidate] = []
    rejected: dict[str, str] = {}
    for name, backend in backends.items():
        if name in exclude:
            rejected[name] = "excluded"
            continue
        if req.prompt_tokens + req.max_tokens > backend.context_limit:
            rejected[name] = "context"
            continue
        quality = backend.quality.get(req.task_type, 0.0)
        if quality < req.cls.min_quality:
            rejected[name] = "quality"
            continue
        if req.sensitive and not backend.allows_sensitive:
            rejected[name] = "privacy"
            continue
        scheduler = schedulers[name]
        if backend.kv_capacity_tokens and (
            scheduler.reservation(req) > scheduler.kv_headroom * backend.kv_capacity_tokens
        ):
            rejected[name] = "kv_capacity"
            continue
        if (
            scheduler.limiter
            and scheduler.limiter.tokens
            and req.prompt_tokens + req.output_estimate > scheduler.limiter.tokens.capacity
        ):
            rejected[name] = "rate_limit_size"
            continue
        if not health[name].available(now):
            rejected[name] = "circuit_open"
            continue
        if scheduler.limiter and not scheduler.limiter.has_room(
            req.prompt_tokens + req.output_estimate, now
        ):
            rejected[name] = "rate_limit"
            continue
        uncached = uncached_tokens(req, name, sessions)
        ttft = (
            scheduler.queue_wait(req.rank)
            + uncached / backend.prefill_tokens_per_s
            + health[name].base_ttft()
        )
        # Passthrough arms leave scheduling to the engine (FIFO or vLLM priority): no shedding here.
        if cfg.scheduling != "passthrough" and now + ttft > req.deadline:
            rejected[name] = "deadline"
            continue
        candidates.append(
            Candidate(
                backend=name,
                cost=backend.cost(req.prompt_tokens, req.output_estimate),
                predicted_ttft=ttft,
                quality=quality,
                error_rate=health[name].error_rate(),
                uncached_tokens=uncached,
            )
        )
    if not candidates:
        permanent = bool(rejected) and set(rejected.values()) <= PERMANENT_REJECTIONS
        raise NoBackend(400 if permanent else 429, "no eligible backend", rejected)
    chosen, reason = _apply_policy(req, cfg, candidates, rejected)
    return Decision(chosen.backend, cfg.routing, candidates, rejected, reason)


def _apply_policy(
    req: Request, cfg: GatewayConfig, candidates: list[Candidate], rejected: dict[str, str]
) -> tuple[Candidate, str]:
    if cfg.routing == "strongest":
        best = max(candidates, key=lambda c: (c.quality, -c.predicted_ttft))
        return best, f"highest {req.task_type} quality"
    if cfg.routing == "cheapest":
        best = min(candidates, key=lambda c: (c.cost, c.predicted_ttft))
        return best, "lowest estimated cost"
    if cfg.routing == "static":
        target = cfg.static_routes.get(req.task_type)
        for candidate in candidates:
            if candidate.backend == target:
                return candidate, f"static route for {req.task_type}"
        raise NoBackend(429, f"static route {target!r} is not eligible", rejected)
    weights = req.cls.weights
    costs = normalise([c.cost for c in candidates])
    ttfts = normalise([c.predicted_ttft for c in candidates])
    risks = normalise([1.0 - c.quality for c in candidates])
    errors = normalise([c.error_rate for c in candidates])
    for i, candidate in enumerate(candidates):
        candidate.score = (
            weights.cost * costs[i]
            + weights.latency * ttfts[i]
            + weights.quality_risk * risks[i]
            + weights.error_rate * errors[i]
        )
    best = min(candidates, key=lambda c: (c.score, c.predicted_ttft))
    return best, f"lowest weighted score {best.score:.3f}"
