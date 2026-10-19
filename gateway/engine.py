"""Connects router, schedulers, dispatcher and telemetry with one dispatch loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TextIO

import httpx

from gateway.analyzer import Request, approx_tokens
from gateway.config import GatewayConfig
from gateway.dispatcher import (
    Backpressure,
    StreamResult,
    UpstreamError,
    build_body,
    count_chunk_tokens,
    stream_chat,
)
from gateway.metrics import GatewayMetrics
from gateway.router import Decision, NoBackend, route
from gateway.scheduler import BackendScheduler, QueueFull
from gateway.sessions import SessionStore
from gateway.telemetry import BackendHealth, poll_once
from gateway.vtc import VTC

log = logging.getLogger(__name__)

BACKPRESSURE_PAUSE_S = 0.2
LOOP_TICK_S = 0.02


@dataclass
class Done:
    result: StreamResult


@dataclass
class Failed:
    status: int
    message: str


@dataclass(eq=False)
class Job:
    req: Request
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    finished: bool = False


class Engine:
    def __init__(
        self,
        cfg: GatewayConfig,
        client: httpx.AsyncClient,
        *,
        clock: Callable[[], float] = time.monotonic,
        count_tokens: Callable[[str], int] = approx_tokens,
        decision_log: TextIO | None = None,
        metrics: GatewayMetrics | None = None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.clock = clock
        self.count_tokens = count_tokens
        self.decision_log = decision_log
        self.metrics = metrics or GatewayMetrics()
        self.vtc = VTC(cfg.vtc_prompt_weight, cfg.vtc_output_weight)
        self.backends = {b.name: b for b in cfg.active_backends()}
        self.schedulers = {
            name: BackendScheduler(b, cfg.classes, self.vtc, cfg.kv_headroom)
            for name, b in self.backends.items()
        }
        self.health = {name: BackendHealth(b) for name, b in self.backends.items()}
        self.sessions = SessionStore()
        self.jobs: dict[str, Job] = {}
        self._wake = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._dispatch_loop()),
            asyncio.create_task(self._telemetry_loop()),
        ]

    async def stop(self) -> None:
        for job in list(self.jobs.values()):
            self.cancel(job)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.client.aclose()

    def submit(self, req: Request) -> Job:
        decision = self._route(req)
        job = Job(req)
        scheduler = self.schedulers[decision.backend]
        if self.cfg.scheduling == "passthrough":
            self.jobs[req.id] = job
            scheduler.admit(req, self.clock())
            self._launch(job)
            return job
        try:
            scheduler.enqueue(req)
        except QueueFull:
            self.metrics.rejections.labels(req.cls.name, "queue_full").inc()
            raise
        self.jobs[req.id] = job
        self._wake.set()
        return job

    def cancel(self, job: Job) -> None:
        if job.finished:
            return
        job.finished = True
        self.metrics.requests.labels(job.req.cls.name, job.req.backend, "cancelled").inc()
        if job.task is not None:
            job.task.cancel()  # _run's finally releases the reservation
            return
        scheduler = self.schedulers.get(job.req.backend)
        if scheduler is not None:
            scheduler.remove(job.req)
        self.jobs.pop(job.req.id, None)

    def _route(self, req: Request) -> Decision:
        try:
            decision = route(
                req,
                self.cfg,
                self.backends,
                self.schedulers,
                self.health,
                self.sessions,
                self.clock(),
            )
        except NoBackend as exc:
            self.metrics.rejections.labels(req.cls.name, "no_backend").inc()
            self._log_decision(req, None, exc)
            raise
        req.backend = decision.backend
        self._log_decision(req, decision, None)
        return decision

    async def _dispatch_loop(self) -> None:
        while True:
            self._wake.clear()
            self._tick()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), LOOP_TICK_S)

    def _tick(self) -> None:
        """One scheduling pass over every backend: age, re-route, expire, admit."""
        now = self.clock()
        for name, scheduler in self.schedulers.items():
            scheduler.promote_aged(now, self.cfg.aging_fraction)
            self._reroute_at_risk(name, scheduler, now)
            for req in scheduler.expire(now):
                self._expire(req)
            while (req := scheduler.next_admissible(now)) is not None:
                job = self.jobs.get(req.id)
                if job is None or job.finished:
                    scheduler.release(req)
                    continue
                self._launch(job)
            for queue in scheduler.queues:
                self.metrics.queue_depth.labels(queue.cls.name, name).set(len(queue))

    def _reroute_at_risk(self, name: str, scheduler: BackendScheduler, now: float) -> None:
        for req in scheduler.at_risk(now, self.cfg.aging_fraction):
            req.rerouted = True
            try:
                decision = route(
                    req,
                    self.cfg,
                    self.backends,
                    self.schedulers,
                    self.health,
                    self.sessions,
                    now,
                    exclude=frozenset({name}),
                )
            except NoBackend:
                continue
            scheduler.remove(req)
            req.backend = decision.backend
            self.schedulers[decision.backend].requeue_front(req)
            self.metrics.reroutes.labels(req.cls.name).inc()
            self._log_decision(req, decision, None)

    def _expire(self, req: Request) -> None:
        job = self.jobs.pop(req.id, None)
        if job is None or job.finished:
            return
        job.finished = True
        self.metrics.rejections.labels(req.cls.name, "deadline").inc()
        job.out.put_nowait(Failed(503, "deadline exceeded while queued"))

    def _launch(self, job: Job) -> None:
        job.task = asyncio.create_task(self._run(job))

    async def _run(self, job: Job) -> None:
        req = job.req
        backend = self.backends[req.backend]
        scheduler = self.schedulers[req.backend]
        health = self.health[req.backend]
        requeued = False
        streamed = 0
        self.vtc.charge_prompt(req.tenant, req.prompt_tokens)

        async def on_chunk(chunk: dict) -> None:
            nonlocal streamed
            n = count_chunk_tokens(chunk)
            if n:
                streamed += n
                self.vtc.charge_output(req.tenant, n)
            await job.out.put(chunk)

        try:
            async with asyncio.timeout(req.cls.timeout_s):
                result = await stream_chat(
                    self.client, backend, build_body(req, backend), on_chunk, self.clock
                )
        except Backpressure as exc:
            scheduler.saturated_until = self.clock() + (exc.retry_after or BACKPRESSURE_PAUSE_S)
            if self.cfg.scheduling == "gateway" and streamed == 0:
                scheduler.release(req)
                scheduler.requeue_front(req)
                job.task = None
                requeued = True
                self._wake.set()
                return
            self._fail(job, 503, "backend busy")
        except TimeoutError:
            health.record_failure(self.clock())
            self._fail(job, 504, "upstream timeout")
        except UpstreamError as exc:
            health.record_failure(self.clock())
            self._fail(job, 502, f"upstream error: {exc}")
        except Exception:
            health.record_failure(self.clock())
            log.exception("unexpected error from %s", req.backend)
            self._fail(job, 502, "internal upstream error")
        else:
            if result.output_tokens > streamed:
                self.vtc.charge_output(req.tenant, result.output_tokens - streamed)
            self._finish(job, result)
        finally:
            if not requeued:
                scheduler.release(req)
                self.jobs.pop(req.id, None)
                self._wake.set()

    def _finish(self, job: Job, result: StreamResult) -> None:
        req = job.req
        job.finished = True
        if result.first_token_at is not None:
            self.health[req.backend].record_success(result.first_token_at - req.dispatched_at)
        if req.session_id:
            prompt_tokens = result.prompt_tokens or req.prompt_tokens
            self.sessions.update(
                req.tenant, req.session_id, req.backend, prompt_tokens + result.output_tokens
            )
        self.metrics.record_finished(req, result, self.backends[req.backend])
        job.out.put_nowait(Done(result))

    def _fail(self, job: Job, status: int, message: str) -> None:
        job.finished = True
        self.metrics.requests.labels(job.req.cls.name, job.req.backend, f"error_{status}").inc()
        job.out.put_nowait(Failed(status, message))

    async def _telemetry_loop(self) -> None:
        while True:
            for name, backend in self.backends.items():
                await poll_once(self.client, backend, self.health[name])
                self.schedulers[name].engine_kv_usage = self.health[name].stats.kv_usage
            await asyncio.sleep(self.cfg.telemetry_interval_s)

    def _log_decision(
        self, req: Request, decision: Decision | None, error: NoBackend | None
    ) -> None:
        if self.decision_log is None:
            return
        entry = {
            "t": self.clock(),
            "request": req.id,
            "tenant": req.tenant,
            "class": req.cls.name,
            "task": req.task_type,
            "prompt_tokens": req.prompt_tokens,
            "session": req.session_id,
        }
        if decision is not None:
            entry.update(
                backend=decision.backend,
                policy=decision.policy,
                reason=decision.reason,
                candidates=[asdict(c) for c in decision.candidates],
                rejected=decision.rejected,
            )
        else:
            entry.update(backend=None, error=str(error), rejected=error.rejected)
        self.decision_log.write(json.dumps(entry) + "\n")
        self.decision_log.flush()
