"""OpenAI-compatible HTTP surface. Tenant and class come from the API key, never from headers."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header
from fastapi import Request as HttpRequest
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from gateway.analyzer import BadRequest, analyze
from gateway.config import GatewayConfig, TenantConfig
from gateway.engine import Done, Engine, Failed, Job
from gateway.router import NoBackend
from gateway.scheduler import QueueFull

DISCONNECT_POLL_S = 0.5


def tenant_keys(cfg: GatewayConfig) -> dict[str, TenantConfig]:
    keys: dict[str, TenantConfig] = {}
    for tenant in cfg.tenants:
        key = os.environ.get(tenant.api_key_env)
        if not key:
            raise RuntimeError(f"tenant {tenant.name}: {tenant.api_key_env} is not set")
        keys[key] = tenant
    return keys


def _error(status: int, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "code": status}}, status_code=status, headers=headers
    )


def create_app(cfg: GatewayConfig, engine: Engine) -> FastAPI:
    keys = tenant_keys(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield
        await engine.stop()
        await engine.client.aclose()

    app = FastAPI(title="Intelligent AI Inference Gateway", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "backends": sorted(engine.backends)}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(engine.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/chat/completions")
    async def chat(http: HttpRequest, authorization: str | None = Header(default=None)):
        tenant = keys.get((authorization or "").removeprefix("Bearer ").strip())
        if tenant is None:
            return _error(401, "invalid API key")
        try:
            body = await http.json()
        except ValueError:
            return _error(400, "body must be JSON")
        if not isinstance(body, dict):
            return _error(400, "body must be a JSON object")
        try:
            req = analyze(body, tenant, cfg, engine.count_tokens, engine.clock(), uuid.uuid4().hex)
        except BadRequest as exc:
            return _error(400, str(exc))
        try:
            job = engine.submit(req)
        except NoBackend as exc:
            return _error(exc.status, f"{exc}: {exc.rejected}")
        except QueueFull as exc:
            return _error(429, str(exc), {"Retry-After": "1"})
        headers = {"X-Gateway-Backend": req.backend, "X-Gateway-Class": req.cls.name}
        if req.stream:
            return StreamingResponse(
                _sse(engine, job), media_type="text/event-stream", headers=headers
            )
        return await _collect(engine, job, http, headers)

    return app


async def _sse(engine: Engine, job: Job):
    try:
        while True:
            item = await job.out.get()
            if isinstance(item, Done):
                yield "data: [DONE]\n\n"
                return
            if isinstance(item, Failed):
                error = {"error": {"message": item.message, "code": item.status}}
                yield f"data: {json.dumps(error)}\n\n"
                return
            yield f"data: {json.dumps(item)}\n\n"
    finally:
        engine.cancel(job)  # no-op once finished; closes upstream if the client disconnected


async def _collect(engine: Engine, job: Job, http: HttpRequest, headers: dict) -> Response:
    content: list[str] = []
    finish, usage, model = None, None, None
    while True:
        try:
            item = await asyncio.wait_for(job.out.get(), DISCONNECT_POLL_S)
        except TimeoutError:
            if await http.is_disconnected():
                engine.cancel(job)
                return Response(status_code=499)
            continue
        if isinstance(item, Failed):
            return _error(item.status, item.message)
        if isinstance(item, Done):
            break
        model = item.get("model", model)
        usage = item.get("usage") or usage
        for choice in item.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            finish = choice.get("finish_reason") or finish
    req = job.req
    usage = usage or {
        "prompt_tokens": req.prompt_tokens,
        "completion_tokens": item.result.output_tokens,
        "total_tokens": req.prompt_tokens + item.result.output_tokens,
    }
    return JSONResponse(
        {
            "id": f"chatcmpl-{req.id}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(content)},
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
        },
        headers=headers,
    )
