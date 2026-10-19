"""Per-backend queues: strict class priority, VTC fairness inside a class, aging, expiry and
KV-token admission control."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from gateway.analyzer import Request
from gateway.config import BackendConfig, ClassConfig
from gateway.kv import reservation_tokens
from gateway.ratelimit import HostedLimiter
from gateway.vtc import VTC


class QueueFull(Exception):
    def __init__(self, class_name: str) -> None:
        super().__init__(f"{class_name} queue is full")
        self.class_name = class_name


class ClassQueue:
    """One service class on one backend: a FIFO lane per tenant, VTC picks the lane."""

    def __init__(self, cls: ClassConfig, vtc: VTC) -> None:
        self.cls = cls
        self.vtc = vtc
        self._lanes: dict[str, deque[Request]] = {}

    def __len__(self) -> int:
        return sum(len(lane) for lane in self._lanes.values())

    def __iter__(self) -> Iterator[Request]:
        for lane in self._lanes.values():
            yield from lane

    def backlogged(self) -> list[str]:
        return [tenant for tenant, lane in self._lanes.items() if lane]

    def push(self, req: Request, *, front: bool = False, force: bool = False) -> None:
        if not force and len(self) >= self.cls.queue_limit:
            raise QueueFull(self.cls.name)
        lane = self._lanes.setdefault(req.tenant, deque())
        if not lane:
            self.vtc.on_backlogged(req.tenant, self.backlogged())
        if front:
            lane.appendleft(req)
        else:
            lane.append(req)

    def head(self) -> Request | None:
        tenant = self.vtc.pick(self.backlogged())
        return self._lanes[tenant][0] if tenant else None

    def remove(self, req: Request) -> bool:
        lane = self._lanes.get(req.tenant)
        if lane is not None and req in lane:
            lane.remove(req)
            return True
        return False


class BackendScheduler:
    def __init__(
        self,
        backend: BackendConfig,
        classes: list[ClassConfig],
        vtc: VTC,
        kv_headroom: float = 0.9,
    ) -> None:
        self.backend = backend
        self.queues = [ClassQueue(c, vtc) for c in sorted(classes, key=lambda c: c.rank)]
        self.kv_headroom = kv_headroom
        self.limiter = HostedLimiter(backend.rpm, backend.tpm) if backend.kind == "hosted" else None
        self.in_flight: dict[str, int] = {}  # request id -> reserved KV tokens
        self.engine_kv_usage = 0.0  # 0-1 fraction from vllm:kv_cache_usage_perc
        self.saturated_until = 0.0  # set after a 429/503 from the backend

    def _queue(self, rank: int) -> ClassQueue:
        return next(q for q in self.queues if q.cls.rank == rank)

    def enqueue(self, req: Request) -> None:
        self._queue(req.rank).push(req)

    def requeue_front(self, req: Request) -> None:
        self._queue(req.rank).push(req, front=True, force=True)

    def remove(self, req: Request) -> bool:
        return any(q.remove(req) for q in self.queues)

    def queued(self) -> list[Request]:
        return [r for q in self.queues for r in q]

    def reservation(self, req: Request) -> int:
        return reservation_tokens(req.prompt_tokens, req.output_estimate, self.backend.block_size)

    def kv_reserved(self) -> int:
        ledger = sum(self.in_flight.values())
        capacity = self.backend.kv_capacity_tokens
        if capacity is None:
            return ledger
        return max(ledger, int(self.engine_kv_usage * capacity))

    def can_admit(self, req: Request, now: float) -> bool:
        if now < self.saturated_until or len(self.in_flight) >= self.backend.max_in_flight:
            return False
        if self.limiter and not self.limiter.has_room(req.prompt_tokens + req.output_estimate, now):
            return False
        capacity = self.backend.kv_capacity_tokens
        if capacity is None:
            return True
        return self.kv_reserved() + self.reservation(req) <= self.kv_headroom * capacity

    def admit(self, req: Request, now: float) -> None:
        self.in_flight[req.id] = self.reservation(req) if self.backend.kv_capacity_tokens else 0
        if self.limiter:
            self.limiter.take(req.prompt_tokens + req.output_estimate, now)
        req.dispatched_at = now

    def next_admissible(self, now: float) -> Request | None:
        """Strict priority: if the most urgent non-empty class cannot be admitted, nothing is."""
        for queue in self.queues:
            head = queue.head()
            if head is None:
                continue
            if not self.can_admit(head, now):
                return None
            queue.remove(head)
            self.admit(head, now)
            return head
        return None

    def release(self, req: Request) -> None:
        self.in_flight.pop(req.id, None)

    def promote_aged(self, now: float, aging_fraction: float) -> list[Request]:
        """Move a request up one class, once, after it waited aging_fraction of its target."""
        moved = []
        for upper, lower in zip(self.queues, self.queues[1:], strict=False):
            for req in list(lower):
                if not req.promoted and now - req.arrival > aging_fraction * req.cls.ttft_target_s:
                    lower.remove(req)
                    req.rank = upper.cls.rank
                    req.promoted = True
                    upper.push(req, force=True)
                    moved.append(req)
        return moved

    def at_risk(self, now: float, aging_fraction: float) -> list[Request]:
        """Top-class requests that have used aging_fraction of their target and not re-routed."""
        top = self.queues[0]
        return [
            r
            for r in top
            if not r.rerouted and now - r.arrival > aging_fraction * r.cls.ttft_target_s
        ]

    def expire(self, now: float) -> list[Request]:
        expired = [r for r in self.queued() if now > r.deadline]
        for req in expired:
            self.remove(req)
        return expired

    def queue_wait(self, rank: int) -> float:
        """Seconds of queued work at this rank or more urgent, at the backend's throughput."""
        work = sum(
            r.prompt_tokens + r.output_estimate
            for q in self.queues
            if q.cls.rank <= rank
            for r in q
        )
        return work / self.backend.throughput_tokens_per_s
