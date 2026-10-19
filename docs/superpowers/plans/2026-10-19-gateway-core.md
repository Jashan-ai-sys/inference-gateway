# Gateway Core Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the gateway software: an OpenAI-compatible FastAPI gateway whose Model Router and Workload Scheduler work together, tested end to end against llm-d-inference-sim on the Mac.

**Architecture:** Most of the logic is plain, synchronous, unit-tested code:
- `analyzer`: request analysis
- `kv`: KV-cache arithmetic
- `vtc`: VTC fairness
- `ratelimit`: token-bucket rate limits
- `scheduler`: per-backend class queues and admission control
- `telemetry`: backend health and `/metrics` polling
- `sessions`: conversation tracking
- `router`: backend choice

One async `Engine` connects them to an httpx streaming `dispatcher` with a single dispatch loop. `api` exposes it over HTTP. Every policy is chosen from config, so each benchmark arm is the same code with different settings.

**Tech Stack:** Python 3.12, uv, FastAPI, uvicorn, httpx, pydantic v2, PyYAML, prometheus-client, tokenizers, pytest, pytest-asyncio, ruff, Docker (llm-d-inference-sim v1.0.0), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-19-inference-gateway-design.md`

**Later plans:**
- Plan 2 (benchmark harness, `HYPOTHESES.md`, Terraform) is written after this plan ships.
- Plan 3 (GPU sessions, `RESULTS.md`, README, interview prep) is written after Plan 2 ships.

## Global Constraints

- Python `>=3.12`; dependencies managed with uv; ruff line length 100.
- Tenant and service class come from the API key only, never from a client header (spec §4.1).
- vLLM `priority`: lower runs sooner. interactive = 0, standard = 1, background = 2. Sent only to backends with `send_priority: true` (spec §4.8).
- VTC weights: prompt tokens × 1, output tokens × 2 (spec §4.5).
- KV admission: reserved + request ≤ 0.9 × KV capacity. Reservations are rounded up to blocks of 16. Capacity is the value vLLM reports, never "GPU memory minus weights" (spec §4.6).
- Aging: promote one class, at most once, after waiting `aging_fraction` (0.5) × the class TTFT target. Top-class requests get one re-route at the same threshold. A request still queued after its deadline gets 503 (spec §4.5 and §4.7, corrected alongside this plan).
- The simulator is FIFO-only: never use simulator numbers as scheduling evidence.
- Config lives in one YAML file per environment (`config/sim.yaml` here). It holds the classes, registry and tenants that spec §2 described as separate files.
- Git:
  - Work on branch `feat/gateway-core`, never on `main`; the user's hook blocks commits there.
  - Identity is already set in the repo: "Jashanprit Singh" with the GitHub no-reply email.
  - No Claude attribution in commits.
  - Do not push anywhere. The user creates the GitHub repo and approves the first push.
- Format before each commit: `make fmt`.

## File structure

```
pyproject.toml, Makefile, README.md, .github/workflows/ci.yml, docker-compose.sim.yml
config/sim.yaml                 classes, backends (registry), tenants for local development
gateway/__init__.py
gateway/config.py               pydantic config models, YAML loader, env overrides
gateway/analyzer.py             Request dataclass, token counting, class/deadline assignment
gateway/kv.py                   KV-cache arithmetic (bytes per token, block rounding)
gateway/vtc.py                  Virtual Token Counter fairness
gateway/ratelimit.py            token buckets for hosted rate limits
gateway/scheduler.py            per-backend class queues, aging, expiry, admission
gateway/telemetry.py            /metrics parsing, EWMAs, circuit breaker, poller
gateway/sessions.py             session → backend memory for the switch-cost signal
gateway/router.py               filter, predicted TTFT, policies
gateway/dispatcher.py           upstream SSE streaming and failure classification
gateway/metrics.py              gateway Prometheus metrics
gateway/engine.py               dispatch loop connecting everything
gateway/api.py                  FastAPI app
gateway/main.py                 process entry point
scripts/smoke_hosted.py         one-off hosted-provider usage check
tests/conftest.py               shared fixtures and fakes
tests/test_*.py                 unit tests per module
tests/integration/test_sim.py   tests against llm-d-inference-sim
```

---

### Task 0: Account prerequisites (user, not code)

These are long-lead items from spec §7.1. Start them on day 1; nothing in this plan waits on them.

- [ ] **Step 1:** In the AWS console, confirm the account is on the **Paid plan** (Billing → Free plan / Upgrade plan).
- [ ] **Step 2:** Find the quota code, then request it. In us-east-1:

```bash
aws service-quotas list-service-quotas --service-code ec2 --region us-east-1 \
  --query "Quotas[?contains(QuotaName,'G and VT')].[QuotaName,QuotaCode,Value]" --output table
aws service-quotas request-service-quota-increase --service-code ec2 --region us-east-1 \
  --quota-code <code for 'Running On-Demand G and VT instances'> --desired-value 16
```

- [ ] **Step 3:** Create a Groq API key (console.groq.com) and a Gemini API key (aistudio.google.com). Keep them in a password manager, not in the repo.

---

### Task 1: Project scaffold, CI and simulator

**Files:**
- Create: `pyproject.toml`, `Makefile`, `README.md`, `gateway/__init__.py`, `tests/__init__.py` (empty, so tests can `from tests.conftest import ...`), `tests/test_smoke.py`, `docker-compose.sim.yml`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces: package `gateway` with `gateway.__version__ == "0.1.0"`; make targets `install`, `fmt`, `lint`, `test`, `sim-up`, `sim-down`, `integration`, `run`; simulators on `localhost:8001` (strong) and `localhost:8002` (small).

- [ ] **Step 1: Create the branch**

```bash
cd ~/personal/inference-gateway
git checkout -b feat/gateway-core
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "inference-gateway"
version = "0.1.0"
description = "Model router + workload scheduler in front of vLLM and hosted LLM APIs"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "httpx>=0.27",
  "pydantic>=2.8",
  "pyyaml>=6",
  "prometheus-client>=0.20",
  "tokenizers>=0.20",
]

[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["gateway"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]
pythonpath = ["."]
markers = ["integration: needs the llm-d-inference-sim containers (make sim-up)"]
addopts = "-m 'not integration'"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

- [ ] **Step 3: Write `Makefile`**

```make
.PHONY: install fmt lint test sim-up sim-down integration run

install: ; uv sync
fmt: ; uv run ruff format . && uv run ruff check --fix .
lint: ; uv run ruff check . && uv run ruff format --check .
test: ; uv run pytest
sim-up: ; docker compose -f docker-compose.sim.yml up -d
sim-down: ; docker compose -f docker-compose.sim.yml down
integration: ; uv run pytest -m integration
run: ; GW_CONFIG=config/sim.yaml uv run python -m gateway.main
```

- [ ] **Step 4: Write `gateway/__init__.py`, `README.md` and the smoke test**

`gateway/__init__.py`:

```python
"""Intelligent AI Inference Gateway."""

__version__ = "0.1.0"
```

`README.md`:

```markdown
# Intelligent AI Inference Gateway

A model router coupled to a workload scheduler, in front of vLLM and hosted LLM APIs,
with a benchmark of where scheduling should live. Design: `docs/superpowers/specs/`.

    make install && make test        # unit tests
    make sim-up && make integration  # tests against llm-d-inference-sim
```

`tests/test_smoke.py`:

```python
import gateway


def test_package_version():
    assert gateway.__version__ == "0.1.0"
```

- [ ] **Step 5: Install and run the smoke test**

Run: `make install && make test`
Expected: `1 passed`.

- [ ] **Step 6: Check the simulator's flags before writing the compose file**

Run: `docker run --rm ghcr.io/llm-d/llm-d-inference-sim:v1.0.0 --help`

Confirm these names and their units:
- `--model`, `--port`, `--max-num-seqs`, `--max-waiting-queue-length`
- `--mode` (`random`)
- `--time-to-first-token`, `--inter-token-latency`

If a flag differs, use the name and unit from `--help` in the next step.

- [ ] **Step 7: Write `docker-compose.sim.yml`**

```yaml
# Two simulated vLLM replicas. Functional testing only: the simulator is a FIFO worker pool,
# not a continuous-batching engine, so its latencies are not evidence for scheduling claims.
services:
  sim-strong:
    image: ghcr.io/llm-d/llm-d-inference-sim:v1.0.0
    command: ["--model", "Qwen/Qwen3-8B-AWQ", "--port", "8001", "--mode", "random",
              "--max-num-seqs", "8", "--max-waiting-queue-length", "64",
              "--time-to-first-token", "200ms", "--inter-token-latency", "25ms"]
    ports: ["8001:8001"]
  sim-small:
    image: ghcr.io/llm-d/llm-d-inference-sim:v1.0.0
    # Deliberately tiny queue so integration tests can exercise 429 backpressure.
    command: ["--model", "Qwen/Qwen3-1.7B", "--port", "8002", "--mode", "random",
              "--max-num-seqs", "1", "--max-waiting-queue-length", "1",
              "--time-to-first-token", "100ms", "--inter-token-latency", "10ms"]
    ports: ["8002:8002"]
```

- [ ] **Step 8: Start the simulators and check them**

Run: `make sim-up && sleep 5 && curl -s localhost:8001/metrics | grep -E "vllm:(num_requests_running|kv_cache_usage_perc)"`

Expected: both metric names are printed. If `kv_cache_usage_perc` is missing, note it in the README. Telemetry treats a missing metric as 0.

- [ ] **Step 9: Write `.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest

  integration:
    runs-on: ubuntu-latest
    needs: unit
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync
      - run: docker compose -f docker-compose.sim.yml up -d
      - name: Wait for simulators
        run: |
          for i in $(seq 60); do
            curl -sf localhost:8001/metrics >/dev/null && curl -sf localhost:8002/metrics >/dev/null && exit 0
            sleep 1
          done
          exit 1
      - run: uv run pytest -m integration
```

Before committing, check the action major versions are current (github.com/actions/checkout, github.com/astral-sh/setup-uv). Bump them if newer majors exist.

- [ ] **Step 10: Commit**

```bash
make fmt && make lint && make test
git add -A
git commit -m "chore: project scaffold, CI and simulator compose"
```

---

### Task 2: Configuration

**Files:**
- Create: `gateway/config.py`, `config/sim.yaml`, `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Weights(cost, latency, quality_risk, error_rate)`
  - `ClassConfig(name, rank, ttft_target_s, tpot_target_s, queue_limit, output_p90, timeout_s, vllm_priority, min_quality, weights)`
  - `BackendConfig(name, kind, base_url, model, metrics_url, api_key_env, context_limit, price_in_per_m, price_out_per_m, quality, allows_sensitive, max_in_flight, kv_capacity_tokens, block_size, send_priority, rpm, tpm, prefill_tokens_per_s, throughput_tokens_per_s, base_ttft_s, extra_body)`, with methods `.api_key() -> str | None` and `.cost(prompt_tokens: int, output_tokens: int) -> float`
  - `TenantConfig(name, api_key_env, allowed_classes, default_class, sensitive)`
  - `GatewayConfig(classes, backends, tenants, scheduling, routing, static_routes, vtc_prompt_weight, vtc_output_weight, kv_headroom, aging_fraction, telemetry_interval_s, decision_log)`, with methods `.classes_by_rank()`, `.class_by_name(name)`, `.backend_by_name(name)` and `.active_backends()`
  - `load_config(path) -> GatewayConfig`
  - `apply_env_overrides(cfg) -> GatewayConfig`
  - Test helpers in `tests/conftest.py`: `cfg` fixture, `with_backend`, `with_class`, `only_backends`, `make_request`, `chat_chunk`, `sse_bytes`, `sse_response`, `SlowStream`, `FakeClock`, `mock_transport`, `drain`

- [ ] **Step 1: Write `config/sim.yaml`**

```yaml
# Local development against llm-d-inference-sim (docker-compose.sim.yml).
# Performance priors (prefill/throughput/base TTFT) are placeholders; Plan 3 replaces them with
# M0 measurements. Self-hosted prices are 0 here; the GPU config sets amortised $/token.
scheduling: gateway
routing: adaptive
aging_fraction: 0.5
kv_headroom: 0.9
decision_log: null

classes:
  - name: interactive
    rank: 0
    ttft_target_s: 1.0
    tpot_target_s: 0.1
    queue_limit: 64
    output_p90: 300
    timeout_s: 60
    vllm_priority: 0
    weights: {cost: 0.1, latency: 0.6, quality_risk: 0.2, error_rate: 0.1}
  - name: standard
    rank: 1
    ttft_target_s: 5.0
    tpot_target_s: 0.15
    queue_limit: 256
    output_p90: 800
    timeout_s: 300
    vllm_priority: 1
    weights: {cost: 0.3, latency: 0.3, quality_risk: 0.3, error_rate: 0.1}
  - name: background
    rank: 2
    ttft_target_s: 60.0
    tpot_target_s: 0.5
    queue_limit: 1024
    output_p90: 2000
    timeout_s: 1200
    vllm_priority: 2
    weights: {cost: 0.6, latency: 0.1, quality_risk: 0.2, error_rate: 0.1}

backends:
  - name: sim-strong
    kind: vllm
    base_url: http://localhost:8001/v1
    metrics_url: http://localhost:8001/metrics
    model: Qwen/Qwen3-8B-AWQ
    context_limit: 8192
    quality: {chat: 0.8, code: 0.75, math: 0.7}
    kv_capacity_tokens: 37000
    max_in_flight: 8
    prefill_tokens_per_s: 4000
    throughput_tokens_per_s: 600
    base_ttft_s: 0.2
    extra_body: {chat_template_kwargs: {enable_thinking: false}}
  - name: sim-small
    kind: vllm
    base_url: http://localhost:8002/v1
    metrics_url: http://localhost:8002/metrics
    model: Qwen/Qwen3-1.7B
    context_limit: 8192
    quality: {chat: 0.6, code: 0.45, math: 0.4}
    kv_capacity_tokens: 24000
    max_in_flight: 16
    prefill_tokens_per_s: 12000
    throughput_tokens_per_s: 1500
    base_ttft_s: 0.1
    extra_body: {chat_template_kwargs: {enable_thinking: false}}
  - name: groq-gpt-oss-20b
    kind: hosted
    base_url: https://api.groq.com/openai/v1
    model: openai/gpt-oss-20b
    api_key_env: GROQ_API_KEY
    context_limit: 131072
    price_in_per_m: 0.075
    price_out_per_m: 0.30
    quality: {chat: 0.75, code: 0.7, math: 0.75}
    allows_sensitive: true
    max_in_flight: 8
    rpm: 30
    tpm: 8000
    prefill_tokens_per_s: 50000
    throughput_tokens_per_s: 1000
    base_ttft_s: 0.35

tenants:
  - {name: tenant-a, api_key_env: GW_KEY_A, allowed_classes: [interactive, standard, background], default_class: standard}
  - {name: tenant-b, api_key_env: GW_KEY_B, allowed_classes: [interactive, standard, background], default_class: interactive}
  - {name: tenant-c, api_key_env: GW_KEY_C, allowed_classes: [standard, background], default_class: background, sensitive: true}

static_routes: {chat: sim-small, code: sim-strong, math: sim-strong}
```

- [ ] **Step 2: Write `tests/conftest.py`**

This holds the shared fixtures used by every later task.

```python
import asyncio
import inspect
import itertools
import json
from pathlib import Path

import httpx
import pytest

from gateway.config import GatewayConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = ROOT / "config" / "sim.yaml"
_ids = itertools.count()


@pytest.fixture
def cfg(monkeypatch) -> GatewayConfig:
    for letter in "abc":
        monkeypatch.setenv(f"GW_KEY_{letter.upper()}", f"key-{letter}")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return load_config(SIM_CONFIG)


def with_backend(cfg: GatewayConfig, name: str, **changes) -> GatewayConfig:
    backends = [b.model_copy(update=changes) if b.name == name else b for b in cfg.backends]
    return cfg.model_copy(update={"backends": backends})


def with_class(cfg: GatewayConfig, name: str, **changes) -> GatewayConfig:
    classes = [c.model_copy(update=changes) if c.name == name else c for c in cfg.classes]
    return cfg.model_copy(update={"classes": classes})


def only_backends(cfg: GatewayConfig, *names: str) -> GatewayConfig:
    return cfg.model_copy(update={"backends": [b for b in cfg.backends if b.name in names]})


def make_request(
    cfg: GatewayConfig,
    *,
    tenant: str = "tenant-a",
    cls: str = "standard",
    prompt: int = 100,
    max_tokens: int = 50,
    now: float = 0.0,
    task: str = "chat",
    session: str | None = None,
    sensitive: bool = False,
    content: str = "hi",
):
    from gateway.analyzer import Request  # lazy: conftest is used from Task 2, analyzer from Task 3

    c = cfg.class_by_name(cls)
    return Request(
        id=f"r{next(_ids)}",
        tenant=tenant,
        sensitive=sensitive,
        cls=c,
        rank=c.rank,
        vllm_priority=c.vllm_priority,
        body={"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens},
        prompt_tokens=prompt,
        max_tokens=max_tokens,
        output_estimate=min(max_tokens, c.output_p90),
        task_type=task,
        session_id=session,
        stream=True,
        arrival=now,
        deadline=now + c.ttft_target_s,
    )


def chat_chunk(content: str | None = None, finish: str | None = None, usage: dict | None = None):
    chunk: dict = {"id": "c", "object": "chat.completion.chunk", "model": "m", "choices": []}
    if content is not None or finish is not None:
        delta = {"content": content} if content else {}
        chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
    if usage:
        chunk["usage"] = usage
    return chunk


def sse_bytes(chunks: list[dict], done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


def sse_response(tokens=("Hello", " world"), usage: dict | None = None) -> httpx.Response:
    chunks = [chat_chunk(t) for t in tokens] + [chat_chunk(finish="stop")]
    if usage:
        chunks.append(chat_chunk(usage=usage))
    return httpx.Response(
        200, content=sse_bytes(chunks), headers={"content-type": "text/event-stream"}
    )


class SlowStream(httpx.AsyncByteStream):
    """An upstream that keeps streaming tokens and records whether the gateway closed it."""

    def __init__(self, n_tokens: int = 1000, delay: float = 0.01) -> None:
        self.n_tokens = n_tokens
        self.delay = delay
        self.closed = False

    async def __aiter__(self):
        for i in range(self.n_tokens):
            await asyncio.sleep(self.delay)
            yield sse_bytes([chat_chunk(f"t{i}")], done=False)
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def mock_transport(chat) -> httpx.MockTransport:
    """Routes /metrics polls to an empty 200 and everything else to `chat` (sync or async)."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/metrics"):
            return httpx.Response(200, text="")
        result = chat(request)
        return await result if inspect.isawaitable(result) else result

    return httpx.MockTransport(handler)


async def drain(job, timeout: float = 5.0) -> list:
    """Collect a job's output until Done or Failed."""
    from gateway.engine import Done, Failed

    items = []
    while True:
        item = await asyncio.wait_for(job.out.get(), timeout)
        items.append(item)
        if isinstance(item, Done | Failed):
            return items
```

`make_request` imports `gateway.analyzer` lazily, so this conftest works before Task 3 exists. It is only called from Task 3 onwards.

- [ ] **Step 3: Write the failing tests `tests/test_config.py`**

```python
import pytest
from pydantic import ValidationError

from gateway.config import GatewayConfig, apply_env_overrides, load_config
from tests.conftest import SIM_CONFIG


def _raw() -> dict:
    import yaml

    return yaml.safe_load(SIM_CONFIG.read_text())


def test_sim_config_loads():
    cfg = load_config(SIM_CONFIG)
    assert [c.name for c in cfg.classes_by_rank()] == ["interactive", "standard", "background"]
    assert cfg.class_by_name("interactive").ttft_target_s == 1.0
    assert cfg.backend_by_name("sim-strong").kv_capacity_tokens == 37000


def test_vllm_backend_requires_kv_capacity():
    raw = _raw()
    raw["backends"][0]["kv_capacity_tokens"] = None
    with pytest.raises(ValidationError, match="kv_capacity_tokens"):
        GatewayConfig.model_validate(raw)


def test_tenant_default_class_must_be_allowed():
    raw = _raw()
    raw["tenants"][2]["default_class"] = "interactive"
    with pytest.raises(ValidationError, match="default_class"):
        GatewayConfig.model_validate(raw)


def test_tenant_unknown_class_rejected():
    raw = _raw()
    raw["tenants"][0]["allowed_classes"].append("vip")
    with pytest.raises(ValidationError, match="unknown classes"):
        GatewayConfig.model_validate(raw)


def test_static_route_must_name_a_backend():
    raw = _raw()
    raw["static_routes"]["chat"] = "nope"
    with pytest.raises(ValidationError, match="unknown backend"):
        GatewayConfig.model_validate(raw)


def test_hosted_backend_without_key_is_inactive(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    names = [b.name for b in load_config(SIM_CONFIG).active_backends()]
    assert names == ["sim-strong", "sim-small"]
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert "groq-gpt-oss-20b" in [b.name for b in load_config(SIM_CONFIG).active_backends()]


def test_backend_cost():
    groq = load_config(SIM_CONFIG).backend_by_name("groq-gpt-oss-20b")
    assert groq.cost(1_000_000, 1_000_000) == pytest.approx(0.375)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("GW_SCHEDULING", "passthrough")
    monkeypatch.setenv("GW_ROUTING", "cheapest")
    cfg = apply_env_overrides(load_config(SIM_CONFIG))
    assert (cfg.scheduling, cfg.routing) == ("passthrough", "cheapest")
    monkeypatch.setenv("GW_ROUTING", "random")
    with pytest.raises(ValidationError):
        apply_env_overrides(load_config(SIM_CONFIG))
```

- [ ] **Step 4: Run the tests to see them fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gateway.config'`.

- [ ] **Step 5: Implement `gateway/config.py`**

```python
"""Gateway configuration: service classes, backends and tenants, loaded from one YAML file."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)


class Weights(BaseModel):
    cost: float
    latency: float
    quality_risk: float
    error_rate: float


class ClassConfig(BaseModel):
    name: str
    rank: int  # 0 is the most urgent class
    ttft_target_s: float
    tpot_target_s: float
    queue_limit: int
    output_p90: int
    timeout_s: float
    vllm_priority: int  # lower runs sooner in vLLM
    min_quality: float = 0.0
    weights: Weights


class BackendConfig(BaseModel):
    name: str
    kind: Literal["vllm", "hosted"]
    base_url: str  # OpenAI-compatible root, e.g. http://localhost:8001/v1
    model: str
    metrics_url: str | None = None
    api_key_env: str | None = None
    context_limit: int
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0
    quality: dict[str, float] = Field(default_factory=dict)
    allows_sensitive: bool = True
    max_in_flight: int = 32
    kv_capacity_tokens: int | None = None
    block_size: int = 16
    send_priority: bool = False
    rpm: int | None = None
    tpm: int | None = None
    prefill_tokens_per_s: float
    throughput_tokens_per_s: float
    base_ttft_s: float
    extra_body: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _vllm_needs_capacity(self) -> BackendConfig:
        if self.kind == "vllm" and not self.kv_capacity_tokens:
            raise ValueError(f"backend {self.name}: vllm backends need kv_capacity_tokens")
        return self

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def cost(self, prompt_tokens: int, output_tokens: int) -> float:
        return (prompt_tokens * self.price_in_per_m + output_tokens * self.price_out_per_m) / 1e6


class TenantConfig(BaseModel):
    name: str
    api_key_env: str
    allowed_classes: list[str]
    default_class: str
    sensitive: bool = False


class GatewayConfig(BaseModel):
    classes: list[ClassConfig]
    backends: list[BackendConfig]
    tenants: list[TenantConfig]
    scheduling: Literal["passthrough", "gateway"] = "gateway"
    routing: Literal["adaptive", "strongest", "cheapest", "static"] = "adaptive"
    static_routes: dict[str, str] = Field(default_factory=dict)
    vtc_prompt_weight: float = 1.0
    vtc_output_weight: float = 2.0
    kv_headroom: float = 0.9
    aging_fraction: float = 0.5
    telemetry_interval_s: float = 1.0
    decision_log: str | None = None

    @model_validator(mode="after")
    def _check_references(self) -> GatewayConfig:
        class_names = {c.name for c in self.classes}
        for t in self.tenants:
            unknown = set(t.allowed_classes) - class_names
            if unknown:
                raise ValueError(f"tenant {t.name}: unknown classes {sorted(unknown)}")
            if t.default_class not in t.allowed_classes:
                raise ValueError(f"tenant {t.name}: default_class must be in allowed_classes")
        backend_names = {b.name for b in self.backends}
        for task, backend in self.static_routes.items():
            if backend not in backend_names:
                raise ValueError(f"static route {task}: unknown backend {backend}")
        return self

    def classes_by_rank(self) -> list[ClassConfig]:
        return sorted(self.classes, key=lambda c: c.rank)

    def class_by_name(self, name: str) -> ClassConfig:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(name)

    def backend_by_name(self, name: str) -> BackendConfig:
        for b in self.backends:
            if b.name == name:
                return b
        raise KeyError(name)

    def active_backends(self) -> list[BackendConfig]:
        """Backends that can be used now: hosted ones need their API key in the environment."""
        active = []
        for b in self.backends:
            if b.api_key_env and not b.api_key():
                log.warning("backend %s disabled: %s is not set", b.name, b.api_key_env)
                continue
            active.append(b)
        return active


def load_config(path: str | Path) -> GatewayConfig:
    return GatewayConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def apply_env_overrides(cfg: GatewayConfig) -> GatewayConfig:
    """Benchmark arms switch policy with GW_SCHEDULING / GW_ROUTING instead of extra files."""
    updates = {}
    if value := os.environ.get("GW_SCHEDULING"):
        updates["scheduling"] = value
    if value := os.environ.get("GW_ROUTING"):
        updates["routing"] = value
    if not updates:
        return cfg
    return GatewayConfig.model_validate({**cfg.model_dump(), **updates})
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 8 passed.

- [ ] **Step 7: Commit**

```bash
make fmt
git add -A
git commit -m "feat: gateway configuration models and sim config"
```

---

### Task 3: Request analyzer

**Study first (user, about 45 min):**
- prefill vs decode
- TTFT, TPOT and ITL definitions (spec §6.4)

Explain back: why does a request's deadline use its TTFT target rather than its total latency?

**Files:**
- Create: `gateway/analyzer.py`, `tests/test_analyzer.py`

**Interfaces:**
- Consumes: `GatewayConfig`, `ClassConfig`, `TenantConfig` from Task 2.
- Produces:
  - `Request`: a dataclass with `eq=False`, fields in the order shown below.
  - `BadRequest(Exception)`
  - `GATEWAY_FIELDS`
  - `approx_tokens(text) -> int`
  - `qwen_token_counter(model="Qwen/Qwen3-8B") -> Callable[[str], int]`
  - `prompt_text(messages) -> str`
  - `resolve_class(tenant, requested) -> str`
  - `analyze(body, tenant, cfg, count_tokens, now, request_id) -> Request`

- [ ] **Step 1: Write the failing tests `tests/test_analyzer.py`**

```python
import pytest

from gateway.analyzer import BadRequest, analyze, approx_tokens, prompt_text


def _tenant(cfg, name):
    return next(t for t in cfg.tenants if t.name == name)


def _body(**extra):
    return {"messages": [{"role": "user", "content": "x" * 400}], **extra}


def test_default_class_and_deadline(cfg):
    req = analyze(_body(), _tenant(cfg, "tenant-b"), cfg, approx_tokens, now=10.0, request_id="r1")
    assert req.cls.name == "interactive"
    assert req.deadline == pytest.approx(11.0)
    assert req.rank == 0 and req.vllm_priority == 0


def test_requested_class_must_be_allowed(cfg):
    tenant_c = _tenant(cfg, "tenant-c")
    assert (
        analyze(_body(service_class="standard"), tenant_c, cfg, approx_tokens, 0, "r").cls.name
        == "standard"
    )
    with pytest.raises(BadRequest, match="not allowed"):
        analyze(_body(service_class="interactive"), tenant_c, cfg, approx_tokens, 0, "r")


def test_gateway_fields_are_stripped_from_forwarded_body(cfg):
    body = _body(service_class="standard", task_type="code", session_id="s1", temperature=0)
    req = analyze(body, _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")
    assert req.task_type == "code" and req.session_id == "s1"
    assert set(req.body) == {"messages", "temperature"}


def test_token_counts_and_output_estimate(cfg):
    req = analyze(_body(max_tokens=5000), _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")
    assert req.prompt_tokens == 100 + 4  # 400 chars / 4, plus 4 template tokens per message
    assert req.max_tokens == 5000
    assert req.output_estimate == 800  # capped at the standard class p90


def test_max_tokens_defaults_to_class_p90(cfg):
    req = analyze(_body(), _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")
    assert req.max_tokens == 800


def test_list_content_parts_are_counted():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "ab"}, {"type": "image_url"}]}
    ]
    assert prompt_text(messages) == "ab"


@pytest.mark.parametrize("body", [{}, {"messages": []}, {"messages": "hi"}])
def test_bad_messages_rejected(cfg, body):
    with pytest.raises(BadRequest):
        analyze(body, _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")


def test_non_positive_max_tokens_rejected(cfg):
    with pytest.raises(BadRequest):
        analyze(_body(max_tokens=-1), _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_analyzer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gateway.analyzer'`.

- [ ] **Step 3: Implement `gateway/analyzer.py`**

```python
"""Turns an incoming chat request into a scheduling Request: tenant, class, deadline, tokens."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from gateway.config import ClassConfig, GatewayConfig, TenantConfig

GATEWAY_FIELDS = frozenset({"service_class", "task_type", "session_id"})
CHAT_TEMPLATE_TOKENS_PER_MESSAGE = 4


class BadRequest(Exception):
    pass


@dataclass(eq=False)
class Request:
    id: str
    tenant: str
    sensitive: bool
    cls: ClassConfig
    rank: int  # current queue rank; aging can lower it by one
    vllm_priority: int
    body: dict  # what is forwarded upstream (gateway-only fields removed)
    prompt_tokens: int
    max_tokens: int
    output_estimate: int  # min(max_tokens, class p90): used for admission and queue-wait estimates
    task_type: str
    session_id: str | None
    stream: bool
    arrival: float
    deadline: float  # arrival + class TTFT target
    backend: str | None = None
    promoted: bool = False
    rerouted: bool = False
    dispatched_at: float | None = None


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def qwen_token_counter(model: str = "Qwen/Qwen3-8B") -> Callable[[str], int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(model)
    return lambda text: len(tokenizer.encode(text).ids)


def prompt_text(messages: list) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    return "\n".join(parts)


def resolve_class(tenant: TenantConfig, requested: str | None) -> str:
    if requested is None:
        return tenant.default_class
    if requested not in tenant.allowed_classes:
        raise BadRequest(f"class {requested!r} is not allowed for this API key")
    return requested


def analyze(
    body: dict,
    tenant: TenantConfig,
    cfg: GatewayConfig,
    count_tokens: Callable[[str], int],
    now: float,
    request_id: str,
) -> Request:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a non-empty list")
    cls = cfg.class_by_name(resolve_class(tenant, body.get("service_class")))
    raw_max = body.get("max_tokens") or body.get("max_completion_tokens")
    max_tokens = int(raw_max) if raw_max is not None else cls.output_p90
    if max_tokens <= 0:
        raise BadRequest("max_tokens must be positive")
    prompt_tokens = count_tokens(prompt_text(messages)) + CHAT_TEMPLATE_TOKENS_PER_MESSAGE * len(
        messages
    )
    session = body.get("session_id")
    return Request(
        id=request_id,
        tenant=tenant.name,
        sensitive=tenant.sensitive,
        cls=cls,
        rank=cls.rank,
        vllm_priority=cls.vllm_priority,
        body={k: v for k, v in body.items() if k not in GATEWAY_FIELDS},
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        output_estimate=min(max_tokens, cls.output_p90),
        task_type=str(body.get("task_type") or "chat"),
        session_id=str(session) if session else None,
        stream=bool(body.get("stream", False)),
        arrival=now,
        deadline=now + cls.ttft_target_s,
    )
```

`max_tokens=-1` is falsy-safe here because `raw_max is not None` is the check, so `-1` reaches the `<= 0` guard.

- [ ] **Step 4: Run all tests to see them pass**

Run: `uv run pytest -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
make fmt
git add -A
git commit -m "feat: request analyzer with class, deadline and token estimates"
```

---

### Task 4: KV-cache arithmetic

**Study first (user):** KV-cache sizing by hand (spec §4.6).

Checkpoint A, part 1. Before writing the tests, compute these without help:
- Qwen3-8B BF16 KV bytes per token, from its `config.json`
- how many 4,096-token requests fit in 37,000 KV tokens with 0.9 headroom

**Files:**
- Create: `gateway/kv.py`, `tests/test_kv.py`

**Interfaces:**
- Produces:
  - `kv_bytes_per_token(num_hidden_layers, num_key_value_heads, head_dim, dtype_bytes) -> int`
  - `round_up_to_blocks(tokens, block_size) -> int`
  - `reservation_tokens(prompt_tokens, output_estimate, block_size) -> int`

- [ ] **Step 1: Write the failing tests `tests/test_kv.py`**

```python
from gateway.kv import kv_bytes_per_token, reservation_tokens, round_up_to_blocks


def test_qwen3_8b_bf16_is_144_kib_per_token():
    # config.json: num_hidden_layers=36, num_key_value_heads=8, head_dim=128; BF16 = 2 bytes
    assert kv_bytes_per_token(36, 8, 128, 2) == 144 * 1024


def test_gqa_vs_mha_uses_kv_heads_not_query_heads():
    mha = kv_bytes_per_token(32, 32, 128, 2)  # 32 KV heads
    gqa = kv_bytes_per_token(32, 8, 128, 2)  # 8 KV heads, same model otherwise
    assert mha == 4 * gqa


def test_round_up_to_blocks():
    assert round_up_to_blocks(0, 16) == 0
    assert round_up_to_blocks(16, 16) == 16
    assert round_up_to_blocks(17, 16) == 32


def test_reservation_covers_prompt_and_output_in_whole_blocks():
    assert reservation_tokens(4000, 300, 16) == 4304
    assert reservation_tokens(4001, 300, 16) == 4320
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_kv.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gateway/kv.py`**

```python
"""KV-cache arithmetic used for admission control and documentation."""


def kv_bytes_per_token(
    num_hidden_layers: int, num_key_value_heads: int, head_dim: int, dtype_bytes: int
) -> int:
    """Keys and values (x2) for every layer. Uses KV heads, not attention heads: under GQA the
    KV cache is smaller than the query-head count suggests."""
    return 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes


def round_up_to_blocks(tokens: int, block_size: int) -> int:
    """vLLM allocates the KV cache in fixed-size blocks, so a partial block costs a whole one."""
    return -(-tokens // block_size) * block_size


def reservation_tokens(prompt_tokens: int, output_estimate: int, block_size: int) -> int:
    return round_up_to_blocks(prompt_tokens + output_estimate, block_size)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_kv.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
make fmt
git add -A
git commit -m "feat: KV-cache arithmetic with block rounding"
```

---

### Task 5: VTC fairness

**Study first (user):** read §4 of Sheng et al., "Fairness in Serving Large Language Models" (OSDI '24, arXiv:2401.00588).

Explain back: what goes wrong without the counter lift?

**Files:**
- Create: `gateway/vtc.py`, `tests/test_vtc.py`

**Interfaces:**
- Produces: `VTC(prompt_weight=1.0, output_weight=2.0)` with methods `counter(tenant)`, `on_backlogged(tenant, backlogged)`, `charge_prompt(tenant, tokens)`, `charge_output(tenant, tokens)` and `pick(backlogged) -> str | None`.

- [ ] **Step 1: Write the failing tests `tests/test_vtc.py`**

```python
from gateway.vtc import VTC


def test_pick_lowest_counter_with_name_tiebreak():
    vtc = VTC()
    assert vtc.pick([]) is None
    assert vtc.pick(["b", "a"]) == "a"
    vtc.charge_prompt("a", 10)
    assert vtc.pick(["a", "b"]) == "b"


def test_charges_are_weighted():
    vtc = VTC(prompt_weight=1.0, output_weight=2.0)
    vtc.charge_prompt("a", 100)
    vtc.charge_output("a", 10)
    assert vtc.counter("a") == 120


def test_returning_idle_tenant_is_lifted_to_lowest_backlogged_counter():
    vtc = VTC()
    vtc.charge_prompt("a", 1000)  # a has been busy
    vtc.on_backlogged("b", backlogged=["a"])  # b returns from idle with counter 0
    assert vtc.counter("b") == 1000


def test_lift_never_lowers_a_counter():
    vtc = VTC()
    vtc.charge_prompt("a", 1000)
    vtc.charge_prompt("b", 2000)
    vtc.on_backlogged("b", backlogged=["a"])
    assert vtc.counter("b") == 2000


def test_no_lift_when_nobody_else_is_backlogged():
    vtc = VTC()
    vtc.charge_prompt("a", 1000)
    vtc.on_backlogged("b", backlogged=[])
    assert vtc.counter("b") == 0


def test_returning_tenant_alternates_instead_of_monopolising():
    vtc = VTC()
    for _ in range(50):  # a served alone for a while
        vtc.charge_prompt("a", 100)
    vtc.on_backlogged("b", backlogged=["a"])
    served = []
    for _ in range(10):
        t = vtc.pick(["a", "b"])
        served.append(t)
        vtc.charge_prompt(t, 100)
    assert served.count("a") == 5 and served.count("b") == 5
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_vtc.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gateway/vtc.py`**

```python
"""Virtual Token Counter fairness (Sheng et al., "Fairness in Serving LLMs", OSDI '24).

Each tenant has a counter of weighted tokens it has been served. The backlogged tenant with the
lowest counter goes next, so a tenant sending more requests does not get more service."""

from collections.abc import Iterable


class VTC:
    def __init__(self, prompt_weight: float = 1.0, output_weight: float = 2.0) -> None:
        self.prompt_weight = prompt_weight
        self.output_weight = output_weight
        self._counters: dict[str, float] = {}

    def counter(self, tenant: str) -> float:
        return self._counters.get(tenant, 0.0)

    def on_backlogged(self, tenant: str, backlogged: Iterable[str]) -> None:
        """Lift a tenant that just became backlogged to the lowest counter among the others, so
        credit saved while idle cannot be spent starving tenants that kept working."""
        others = [self.counter(t) for t in backlogged if t != tenant]
        if others:
            self._counters[tenant] = max(self.counter(tenant), min(others))

    def charge_prompt(self, tenant: str, tokens: int) -> None:
        self._counters[tenant] = self.counter(tenant) + self.prompt_weight * tokens

    def charge_output(self, tenant: str, tokens: int) -> None:
        self._counters[tenant] = self.counter(tenant) + self.output_weight * tokens

    def pick(self, backlogged: Iterable[str]) -> str | None:
        candidates = list(backlogged)
        if not candidates:
            return None
        return min(candidates, key=lambda t: (self.counter(t), t))
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_vtc.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
make fmt
git add -A
git commit -m "feat: VTC tenant fairness with counter lift"
```

---

### Task 6: Hosted rate limits

**Files:**
- Create: `gateway/ratelimit.py`, `tests/test_ratelimit.py`

**Interfaces:**
- Produces:
  - `TokenBucket(per_minute)` with `available(now) -> float` and `take(amount, now)`
  - `HostedLimiter(rpm, tpm)` with `has_room(tokens, now) -> bool` and `take(tokens, now)`

- [ ] **Step 1: Write the failing tests `tests/test_ratelimit.py`**

```python
import pytest

from gateway.ratelimit import HostedLimiter, TokenBucket


def test_bucket_starts_full_and_refills_continuously():
    bucket = TokenBucket(per_minute=60)
    assert bucket.available(0.0) == 60
    bucket.take(60, 0.0)
    assert bucket.available(0.0) == 0
    assert bucket.available(10.0) == pytest.approx(10)


def test_bucket_is_capped_at_one_minute():
    bucket = TokenBucket(per_minute=60)
    bucket.take(1, 0.0)
    assert bucket.available(1000.0) == 60


def test_limiter_checks_requests_and_tokens():
    limiter = HostedLimiter(rpm=2, tpm=1000)
    assert limiter.has_room(600, 0.0)
    limiter.take(600, 0.0)
    assert not limiter.has_room(600, 0.0)  # token budget
    assert limiter.has_room(300, 0.0)
    limiter.take(300, 0.0)
    assert not limiter.has_room(10, 0.0)  # request budget (2 per minute)


def test_limiter_without_limits_always_has_room():
    assert HostedLimiter(rpm=None, tpm=None).has_room(10**9, 0.0)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_ratelimit.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gateway/ratelimit.py`**

```python
"""Token buckets that mirror a hosted provider's per-minute request and token limits."""

from __future__ import annotations


class TokenBucket:
    """Refills continuously at `per_minute` units per minute and holds at most one minute's worth."""

    def __init__(self, per_minute: float) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.level = float(per_minute)
        self.updated: float | None = None

    def _refill(self, now: float) -> None:
        if self.updated is not None:
            self.level = min(self.capacity, self.level + (now - self.updated) * self.rate)
        self.updated = now

    def available(self, now: float) -> float:
        self._refill(now)
        return self.level

    def take(self, amount: float, now: float) -> None:
        self._refill(now)
        self.level -= amount


class HostedLimiter:
    def __init__(self, rpm: int | None, tpm: int | None) -> None:
        self.requests = TokenBucket(rpm) if rpm else None
        self.tokens = TokenBucket(tpm) if tpm else None

    def has_room(self, tokens: int, now: float) -> bool:
        if self.requests and self.requests.available(now) < 1:
            return False
        return not (self.tokens and self.tokens.available(now) < tokens)

    def take(self, tokens: int, now: float) -> None:
        if self.requests:
            self.requests.take(1, now)
        if self.tokens:
            self.tokens.take(tokens, now)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_ratelimit.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
make fmt
git add -A
git commit -m "feat: token buckets for hosted rate limits"
```

---

### Task 7: Per-backend scheduler

**Study first (user, about 60 min):**
- continuous batching
- PagedAttention and blocks
- read vLLM v0.32.0 `vllm/v1/core/sched/scheduler.py`: waiting and running queues, the token budget, preemption

Checkpoint B explanation: why must the gateway hold requests back (an in-flight cap plus KV admission) for its priorities to matter at all?

**Files:**
- Create: `gateway/scheduler.py`, `tests/test_scheduler.py`

**Interfaces:**
- Consumes:
  - `Request` (Task 3)
  - `reservation_tokens` (Task 4)
  - `VTC` (Task 5)
  - `HostedLimiter` (Task 6)
  - `BackendConfig`, `ClassConfig` (Task 2)
- Produces:
  - `QueueFull(Exception)` with `.class_name`.
  - `ClassQueue(cls, vtc)` with `__len__`, `__iter__`, `backlogged()`, `push(req, *, front=False, force=False)`, `head()` and `remove(req) -> bool`.
  - `BackendScheduler(backend, classes, vtc, kv_headroom=0.9)`. Attributes: `backend`, `queues`, `limiter`, `in_flight: dict[str, int]`, `engine_kv_usage: float`, `saturated_until: float`.
  - `BackendScheduler` methods: `enqueue(req)`, `requeue_front(req)`, `remove(req) -> bool`, `queued() -> list[Request]`, `reservation(req) -> int`, `kv_reserved() -> int`, `can_admit(req, now) -> bool`, `admit(req, now)`, `next_admissible(now) -> Request | None`, `release(req)`, `promote_aged(now, aging_fraction) -> list[Request]`, `at_risk(now, aging_fraction) -> list[Request]`, `expire(now) -> list[Request]` and `queue_wait(rank) -> float`.

- [ ] **Step 1: Write the failing tests `tests/test_scheduler.py`**

```python
import pytest

from gateway.scheduler import BackendScheduler, QueueFull
from gateway.vtc import VTC
from tests.conftest import make_request, with_class


def sched_for(cfg, name="sim-strong", vtc=None, **changes):
    backend = cfg.backend_by_name(name).model_copy(update=changes)
    return BackendScheduler(backend, cfg.classes, vtc or VTC(), cfg.kv_headroom)


def test_strict_class_priority(cfg):
    s = sched_for(cfg)
    bg = make_request(cfg, cls="background")
    it = make_request(cfg, cls="interactive")
    s.enqueue(bg)
    s.enqueue(it)
    assert s.next_admissible(0.0) is it
    assert s.next_admissible(0.0) is bg


def test_blocked_urgent_head_blocks_lower_classes(cfg):
    s = sched_for(cfg)  # capacity 37000 * 0.9 = 33300 tokens
    s.enqueue(make_request(cfg, cls="interactive", prompt=40000, max_tokens=10))
    s.enqueue(make_request(cfg, cls="background", prompt=10, max_tokens=10))
    assert s.next_admissible(0.0) is None


def test_in_flight_cap(cfg):
    s = sched_for(cfg, max_in_flight=1)
    a, b = make_request(cfg), make_request(cfg)
    s.enqueue(a)
    s.enqueue(b)
    assert s.next_admissible(0.0) is a
    assert s.next_admissible(0.0) is None
    s.release(a)
    assert s.next_admissible(0.0) is b


def test_kv_admission_with_block_rounding_and_release(cfg):
    s = sched_for(cfg)
    first = make_request(cfg, prompt=20000, max_tokens=300)
    second = make_request(cfg, prompt=20000, max_tokens=300)
    s.enqueue(first)
    s.enqueue(second)
    assert s.next_admissible(0.0) is first
    assert s.in_flight[first.id] == 20304  # 20300 rounded up to a multiple of 16
    assert s.next_admissible(0.0) is None  # 40608 > 33300
    s.release(first)
    assert s.next_admissible(0.0) is second


def test_engine_reported_kv_usage_tightens_admission(cfg):
    s = sched_for(cfg)
    s.engine_kv_usage = 0.95
    s.enqueue(make_request(cfg, prompt=10, max_tokens=10))
    assert s.next_admissible(0.0) is None


def test_queue_limit_and_front_requeue(cfg):
    cfg = with_class(cfg, "standard", queue_limit=1)
    s = sched_for(cfg)
    a, b, c = make_request(cfg), make_request(cfg), make_request(cfg)
    s.enqueue(a)
    with pytest.raises(QueueFull):
        s.enqueue(b)
    s.requeue_front(c)  # requeues bypass the limit and go first
    assert s.next_admissible(0.0) is c


def test_vtc_orders_tenants_within_a_class(cfg):
    vtc = VTC()
    vtc.charge_prompt("tenant-a", 1000)  # tenant-a was served a lot earlier
    s = sched_for(cfg, vtc=vtc)
    b = make_request(cfg, tenant="tenant-b")
    a = make_request(cfg, tenant="tenant-a")
    s.enqueue(b)  # b becomes backlogged alone: no lift, stays at 0
    s.enqueue(a)  # a keeps 1000 (a lift never lowers a counter)
    assert s.next_admissible(0.0) is b
    # Had a been queued first, b would have been lifted to 1000 on arrival: that is the
    # counter lift working as intended (b cannot spend credit it saved while idle).


def test_aging_promotes_once_at_half_the_class_target(cfg):
    s = sched_for(cfg)
    bg = make_request(cfg, cls="background", now=0.0)
    s.enqueue(bg)
    assert s.promote_aged(29.0, 0.5) == []
    assert s.promote_aged(31.0, 0.5) == [bg]
    assert bg.rank == 1 and bg.promoted
    assert s.promote_aged(100.0, 0.5) == []  # only one promotion


def test_at_risk_top_class(cfg):
    s = sched_for(cfg)
    it = make_request(cfg, cls="interactive", now=0.0)
    s.enqueue(it)
    assert s.at_risk(0.4, 0.5) == []
    assert s.at_risk(0.6, 0.5) == [it]
    it.rerouted = True
    assert s.at_risk(0.6, 0.5) == []


def test_expire_removes_requests_past_their_deadline(cfg):
    s = sched_for(cfg)
    it = make_request(cfg, cls="interactive", now=0.0)
    s.enqueue(it)
    assert s.expire(0.9) == []
    assert s.expire(1.1) == [it]
    assert s.queued() == []


def test_queue_wait_counts_work_at_or_above_a_rank(cfg):
    s = sched_for(cfg)  # throughput 600 tokens/s
    s.enqueue(make_request(cfg, cls="interactive", prompt=500, max_tokens=300))
    s.enqueue(make_request(cfg, cls="standard", prompt=1000, max_tokens=800))
    s.enqueue(make_request(cfg, cls="standard", prompt=1000, max_tokens=800))
    assert s.queue_wait(0) == pytest.approx(800 / 600)
    assert s.queue_wait(1) == pytest.approx(4400 / 600)


def test_hosted_limiter_blocks_admission(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    s = sched_for(cfg, name="groq-gpt-oss-20b")  # tpm 8000
    big = make_request(cfg, prompt=7000, max_tokens=500)
    s.enqueue(big)
    assert s.next_admissible(0.0) is big
    s.release(big)
    s.enqueue(make_request(cfg, prompt=7000, max_tokens=500))
    assert s.next_admissible(0.0) is None


def test_saturated_backend_admits_nothing(cfg):
    s = sched_for(cfg)
    s.saturated_until = 5.0
    s.enqueue(make_request(cfg))
    assert s.next_admissible(1.0) is None
    assert s.next_admissible(5.0) is not None
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gateway/scheduler.py`**

```python
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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
make fmt
git add -A
git commit -m "feat: per-backend class queues, aging, expiry and KV admission"
```

---

### Task 8: Telemetry and backend health

**Files:**
- Create: `gateway/telemetry.py`, `tests/test_telemetry.py`

**Interfaces:**
- Consumes: `BackendConfig` (Task 2).
- Produces:
  - `EngineStats(running, waiting, kv_usage)`
  - `parse_vllm_metrics(text) -> EngineStats`
  - `Ewma(alpha=0.2)` with `.value` and `update(x)`
  - `BackendHealth(backend, failure_threshold=3, open_s=10.0)` with `.stats`, `base_ttft() -> float`, `error_rate() -> float`, `available(now) -> bool`, `record_success(ttft_s)` and `record_failure(now)`
  - `async poll_once(client, backend, health) -> None`

- [ ] **Step 1: Write the failing tests `tests/test_telemetry.py`**

```python
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gateway/telemetry.py`**

```python
"""Backend health: vLLM /metrics polling, latency and error EWMAs, and a circuit breaker."""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from gateway.config import BackendConfig

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


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
        name, value = match.group(1), float(match.group(3))
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
        """Dispatch-to-first-token, measured; the config prior until the first success."""
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
    except httpx.HTTPError:
        return
    health.stats = parse_vllm_metrics(response.text)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
make fmt
git add -A
git commit -m "feat: vLLM metrics parsing, health EWMAs and circuit breaker"
```

---

### Task 9: Sessions and the router

**Study first (user, about 60 min):**
- prefix caching (block hashing)
- chunked prefill and its TTFT-vs-ITL trade-off
- why KV caches are model-specific: skim the abstracts of HeteroFold (arXiv:2609.32259) and RaReCache (arXiv:2610.11358)

Checkpoint C explanation: an 8,000-token prompt and a 100-token prompt arrive together. Why does the short one get a high TTFT, and which part of this project fixes it?

**Files:**
- Create: `gateway/sessions.py`, `gateway/router.py`, `tests/test_router.py`

**Interfaces:**
- Consumes:
  - `Request` (Task 3)
  - `BackendScheduler` (Task 7)
  - `BackendHealth` (Task 8)
  - `GatewayConfig`, `BackendConfig` (Task 2)
- Produces:
  - `SessionState(backend, tokens)`
  - `SessionStore(max_sessions=10000)` with `get(tenant, session_id) -> SessionState | None` and `update(tenant, session_id, backend, tokens)`
  - `Candidate(backend, cost, predicted_ttft, quality, error_rate, uncached_tokens, score=None)`
  - `Decision(backend, policy, candidates, rejected, reason)`
  - `NoBackend(Exception)` with `.status` and `.rejected`
  - `uncached_tokens(req, backend_name, sessions) -> int`
  - `normalise(values) -> list[float]`
  - `route(req, cfg, backends, schedulers, health, sessions, now, exclude=frozenset()) -> Decision`

- [ ] **Step 1: Write the failing tests `tests/test_router.py`**

```python
import pytest

from gateway.router import NoBackend, normalise, route, uncached_tokens
from gateway.scheduler import BackendScheduler
from gateway.sessions import SessionStore
from gateway.telemetry import BackendHealth
from gateway.vtc import VTC
from tests.conftest import make_request, only_backends, with_backend, with_class


def state(cfg):
    backends = {b.name: b for b in cfg.active_backends()}
    vtc = VTC()
    schedulers = {
        n: BackendScheduler(b, cfg.classes, vtc, cfg.kv_headroom) for n, b in backends.items()
    }
    health = {n: BackendHealth(b) for n, b in backends.items()}
    return backends, schedulers, health, SessionStore()


def decide(cfg, req, now=0.0, st=None, **kw):
    backends, schedulers, health, sessions = st or state(cfg)
    return route(req, cfg, backends, schedulers, health, sessions, now, **kw)


def test_normalise():
    assert normalise([1.0, 3.0, 2.0]) == [0.0, 1.0, 0.5]
    assert normalise([5.0, 5.0]) == [0.0, 0.0]


def test_idle_interactive_chat_prefers_fast_small_model(cfg):
    assert decide(cfg, make_request(cfg, cls="interactive")).backend == "sim-small"


def test_background_avoids_paid_hosted_model(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert decide(cfg, make_request(cfg, cls="background")).backend != "groq-gpt-oss-20b"


def test_interactive_overflows_to_hosted_when_local_queues_would_miss_deadline(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    st = state(cfg)
    _, schedulers, _, _ = st
    for _ in range(5):  # 5 x 700 tokens queued per local backend: > 1 s of work on both
        schedulers["sim-small"].enqueue(
            make_request(cfg, cls="interactive", prompt=400, max_tokens=300)
        )
        schedulers["sim-strong"].enqueue(
            make_request(cfg, cls="interactive", prompt=400, max_tokens=300)
        )
    decision = decide(cfg, make_request(cfg, cls="interactive"), st=st)
    assert decision.backend == "groq-gpt-oss-20b"
    assert decision.rejected == {"sim-strong": "deadline", "sim-small": "deadline"}


def test_session_affinity_counts_re_prefill_when_switching_models(cfg):
    cfg = only_backends(cfg, "sim-strong", "sim-small")
    same = {"prefill_tokens_per_s": 4000, "base_ttft_s": 0.1, "quality": {"chat": 0.7}}
    cfg = with_backend(with_backend(cfg, "sim-strong", **same), "sim-small", **same)
    st = state(cfg)
    st[3].update("tenant-a", "s1", "sim-strong", 3000)
    req = make_request(cfg, cls="standard", prompt=3200, session="s1")
    assert uncached_tokens(req, "sim-strong", st[3]) == 200
    assert uncached_tokens(req, "sim-small", st[3]) == 3200
    decision = decide(cfg, req, st=st)
    assert decision.backend == "sim-strong"


def test_sessions_are_scoped_per_tenant(cfg):
    sessions = SessionStore()
    sessions.update("tenant-a", "s1", "sim-strong", 3000)
    req = make_request(cfg, tenant="tenant-b", prompt=3200, session="s1")
    assert uncached_tokens(req, "sim-strong", sessions) == 3200


def test_context_privacy_quality_filters(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = with_backend(cfg, "groq-gpt-oss-20b", allows_sensitive=False)
    cfg = with_class(cfg, "background", min_quality=0.7)
    decision = decide(cfg, make_request(cfg, cls="background", sensitive=True))
    assert decision.rejected == {"sim-small": "quality", "groq-gpt-oss-20b": "privacy"}
    assert decision.backend == "sim-strong"


def test_only_permanent_rejections_give_400(cfg):
    with pytest.raises(NoBackend) as err:
        decide(cfg, make_request(cfg, prompt=10000))
    assert err.value.status == 400


def test_transient_rejections_give_429(cfg):
    st = state(cfg)
    for h in st[2].values():
        h.open_until = 100.0
    with pytest.raises(NoBackend) as err:
        decide(cfg, make_request(cfg), st=st)
    assert err.value.status == 429
    assert set(err.value.rejected.values()) == {"circuit_open"}


def test_rate_limited_hosted_backend_is_skipped(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    st = state(cfg)
    st[1]["groq-gpt-oss-20b"].limiter.take(8000, 0.0)
    decision = decide(cfg, make_request(cfg), st=st)
    assert decision.rejected["groq-gpt-oss-20b"] == "rate_limit"


def test_exclude(cfg):
    decision = decide(cfg, make_request(cfg, cls="interactive"), exclude=frozenset({"sim-small"}))
    assert decision.backend == "sim-strong"
    assert decision.rejected["sim-small"] == "excluded"


def test_policies(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    req = make_request(cfg, cls="standard", task="code")
    assert decide(cfg.model_copy(update={"routing": "strongest"}), req).backend == "sim-strong"
    assert decide(cfg.model_copy(update={"routing": "cheapest"}), req).backend in {
        "sim-strong",
        "sim-small",
    }
    assert decide(cfg.model_copy(update={"routing": "static"}), req).backend == "sim-strong"
```

In `test_context_privacy_quality_filters`, background requests get 60 s deadlines, so only the quality and privacy rules can reject. sim-small's chat quality is 0.6, below 0.7, so it fails on quality. groq is rejected for privacy because its quality filter passes first.

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gateway/sessions.py`**

```python
"""Remembers which backend served each conversation, for the model-switch cost (spec §4.3)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class SessionState:
    backend: str
    tokens: int  # prompt + output of the last turn: the prefix that backend may still have cached


class SessionStore:
    def __init__(self, max_sessions: int = 10_000) -> None:
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[tuple[str, str], SessionState] = OrderedDict()

    def get(self, tenant: str, session_id: str) -> SessionState | None:
        return self._sessions.get((tenant, session_id))

    def update(self, tenant: str, session_id: str, backend: str, tokens: int) -> None:
        key = (tenant, session_id)
        self._sessions[key] = SessionState(backend, tokens)
        self._sessions.move_to_end(key)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
```

- [ ] **Step 4: Implement `gateway/router.py`**

```python
"""Model Router: hard-constraint filter, predicted TTFT from the scheduler's queue state and the
uncached prompt tokens, then a routing policy."""

from __future__ import annotations

from dataclasses import dataclass

from gateway.analyzer import Request
from gateway.config import BackendConfig, GatewayConfig
from gateway.scheduler import BackendScheduler
from gateway.sessions import SessionStore
from gateway.telemetry import BackendHealth

PERMANENT_REJECTIONS = frozenset({"context", "quality", "privacy"})


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
        if not health[name].available(now):
            rejected[name] = "circuit_open"
            continue
        scheduler = schedulers[name]
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
        if now + ttft > req.deadline:
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
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests/test_router.py -v`
Expected: 13 passed.

If `test_idle_interactive_chat_prefers_fast_small_model` fails, recompute the scores by hand from `config/sim.yaml` before changing any code. For an idle interactive chat request (prompt 100) with Groq inactive:
- predicted TTFT: small 0.108 s, strong 0.225 s
- cost: 0 for both
- normalised latency: small 0, strong 1; normalised quality risk: small 1, strong 0
- score: small 0.2 × 1 = 0.20, strong 0.6 × 1 = 0.60, so small wins

- [ ] **Step 6: Commit**

```bash
make fmt
git add -A
git commit -m "feat: router with deadline-aware filter, switch cost and policies"
```

---

### Task 10: Upstream dispatcher

**Files:**
- Create: `gateway/dispatcher.py`, `tests/test_dispatcher.py`

**Interfaces:**
- Consumes: `Request` (Task 3), `BackendConfig` (Task 2).
- Produces:
  - `Backpressure(Exception)` with `.status` and `.retry_after`
  - `UpstreamError(Exception)` with `.status`
  - `StreamResult(output_tokens, prompt_tokens, first_token_at, finished_at, finish_reason)`
  - `build_body(req, backend) -> dict`
  - `count_chunk_tokens(chunk) -> int`
  - `async stream_chat(client, backend, body, on_chunk, clock) -> StreamResult`

- [ ] **Step 1: Write the failing tests `tests/test_dispatcher.py`**

```python
import asyncio
import json

import httpx
import pytest

from gateway.dispatcher import Backpressure, UpstreamError, build_body, stream_chat
from tests.conftest import FakeClock, SlowStream, make_request, sse_response, with_backend


async def _noop(chunk):
    return None


def test_build_body_sets_upstream_fields(cfg):
    backend = cfg.backend_by_name("sim-strong")
    req = make_request(cfg, cls="interactive", max_tokens=42)
    req.body["max_completion_tokens"] = 7
    body = build_body(req, backend)
    assert body["model"] == "Qwen/Qwen3-8B-AWQ"
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 42 and "max_completion_tokens" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "priority" not in body


def test_priority_only_for_backends_that_want_it(cfg):
    cfg = with_backend(cfg, "sim-strong", send_priority=True)
    body = build_body(make_request(cfg, cls="background"), cfg.backend_by_name("sim-strong"))
    assert body["priority"] == 2


async def test_stream_counts_tokens_and_prefers_reported_usage(cfg):
    clock = FakeClock(5.0)
    seen = []

    async def on_chunk(chunk):
        seen.append(chunk)

    usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    transport = httpx.MockTransport(lambda r: sse_response(("a", "b"), usage=usage))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await stream_chat(client, cfg.backend_by_name("sim-strong"), {}, on_chunk, clock)
    assert result.output_tokens == 3  # usage wins over the 2 counted deltas
    assert result.prompt_tokens == 12
    assert result.first_token_at == 5.0
    assert result.finish_reason == "stop"
    assert len(seen) == 4


async def test_stream_without_usage_uses_counted_tokens(cfg):
    transport = httpx.MockTransport(lambda r: sse_response(("a", "b", "c")))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await stream_chat(
            client, cfg.backend_by_name("sim-strong"), {}, _noop, FakeClock()
        )
    assert result.output_tokens == 3 and result.prompt_tokens is None


async def test_sends_bearer_key_for_hosted_backends(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    captured = {}

    def handler(request):
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return sse_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await stream_chat(
            client, cfg.backend_by_name("groq-gpt-oss-20b"), {"x": 1}, _noop, FakeClock()
        )
    assert captured == {"auth": "Bearer secret", "body": {"x": 1}}


@pytest.mark.parametrize("status", [429, 503])
async def test_busy_statuses_are_backpressure(cfg, status):
    transport = httpx.MockTransport(lambda r: httpx.Response(status, headers={"Retry-After": "2"}))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(Backpressure) as err:
            await stream_chat(client, cfg.backend_by_name("sim-strong"), {}, _noop, FakeClock())
    assert err.value.retry_after == 2.0


async def test_server_error_and_connection_error_are_upstream_errors(cfg):
    backend = cfg.backend_by_name("sim-strong")
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(UpstreamError) as err:
            await stream_chat(client, backend, {}, _noop, FakeClock())
    assert err.value.status == 500

    def refuse(request):
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        with pytest.raises(UpstreamError):
            await stream_chat(client, backend, {}, _noop, FakeClock())


async def test_cancelling_closes_the_upstream_stream(cfg):
    stream = SlowStream(n_tokens=1000, delay=0.01)
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        task = asyncio.create_task(
            stream_chat(client, cfg.backend_by_name("sim-strong"), {}, _noop, FakeClock())
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert stream.closed
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_dispatcher.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gateway/dispatcher.py`**

```python
"""Streams one request to one OpenAI-compatible backend and classifies failures for the engine.

The gateway always streams upstream, even for non-streaming clients, so TTFT is measured the
same way for every request. Cancelling the calling task closes the upstream connection, which
makes vLLM abort the request and free its KV blocks."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from gateway.analyzer import Request
from gateway.config import BackendConfig

BACKPRESSURE_STATUSES = frozenset({429, 503})


class Backpressure(Exception):
    def __init__(self, status: int, retry_after: float | None) -> None:
        super().__init__(f"backend busy ({status})")
        self.status = status
        self.retry_after = retry_after


class UpstreamError(Exception):
    def __init__(self, status: int | None, detail: str) -> None:
        super().__init__(f"{status}: {detail}" if status else detail)
        self.status = status


@dataclass
class StreamResult:
    output_tokens: int
    prompt_tokens: int | None
    first_token_at: float | None
    finished_at: float
    finish_reason: str | None


def build_body(req: Request, backend: BackendConfig) -> dict:
    body = {**req.body, **backend.extra_body}
    body.pop("max_completion_tokens", None)
    body.pop("priority", None)
    body.update(
        model=backend.model,
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=req.max_tokens,
    )
    if backend.send_priority:
        body["priority"] = req.vllm_priority
    return body


def count_chunk_tokens(chunk: dict) -> int:
    """vLLM and most providers stream about one token per content delta; usage corrects it."""
    count = 0
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content"):
            count += 1
    return count


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


async def stream_chat(
    client: httpx.AsyncClient,
    backend: BackendConfig,
    body: dict,
    on_chunk: Callable[[dict], Awaitable[None]],
    clock: Callable[[], float],
) -> StreamResult:
    key = backend.api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    counted, usage, first_token_at, finish = 0, None, None, None
    try:
        async with client.stream(
            "POST", f"{backend.base_url}/chat/completions", json=body, headers=headers
        ) as response:
            if response.status_code in BACKPRESSURE_STATUSES:
                raise Backpressure(response.status_code, _retry_after(response))
            if response.status_code >= 400:
                detail = (await response.aread()).decode(errors="replace")[:500]
                raise UpstreamError(response.status_code, detail)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise UpstreamError(None, json.dumps(chunk["error"])[:500])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                n = count_chunk_tokens(chunk)
                if n and first_token_at is None:
                    first_token_at = clock()
                counted += n
                for choice in chunk.get("choices") or []:
                    finish = choice.get("finish_reason") or finish
                await on_chunk(chunk)
    except httpx.TransportError as exc:
        raise UpstreamError(None, f"{type(exc).__name__}: {exc}") from exc
    output_tokens = usage.get("completion_tokens", counted) if usage else counted
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    return StreamResult(output_tokens, prompt_tokens, first_token_at, clock(), finish)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_dispatcher.py -v`
Expected: 9 passed (the parametrised backpressure test counts twice).

- [ ] **Step 5: Commit**

```bash
make fmt
git add -A
git commit -m "feat: streaming dispatcher with backpressure and cancellation"
```

---

### Task 11: Metrics and the engine

**Files:**
- Create: `gateway/metrics.py`, `gateway/engine.py`, `tests/test_engine.py`

**Interfaces:**
- Consumes: everything from Tasks 2–10.
- Produces:
  - `GatewayMetrics(registry=None)` with `.registry` and `record_finished(req, result, backend) -> bool`.
  - `Done(result)`, `Failed(status, message)`.
  - `Job(req)` with `.out: asyncio.Queue`, `.task`, `.finished`.
  - `Engine(cfg, client, *, clock=time.monotonic, count_tokens=approx_tokens, decision_log=None, metrics=None)`. Attributes: `cfg`, `client`, `clock`, `count_tokens`, `metrics`, `vtc`, `backends`, `schedulers`, `health`, `sessions`, `jobs`.
  - `Engine` methods: `async start()`, `async stop()`, `submit(req) -> Job` (raises `NoBackend` or `QueueFull`), `cancel(job)` and `_tick()` (one synchronous scheduling pass; tests call it directly).

- [ ] **Step 1: Write the failing tests `tests/test_engine.py`**

```python
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
        job = engine.submit(make_request(cfg, prompt=100, session="s1"))
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
        jobs = [engine.submit(make_request(cfg, tenant="tenant-a", content="a")) for _ in range(6)]
        jobs += [engine.submit(make_request(cfg, tenant="tenant-b", content="b")) for _ in range(2)]
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
        items = await drain(engine.submit(make_request(cfg)))
        await engine.stop()
    assert isinstance(items[-1], Done) and len(calls) == 2


async def test_upstream_error_fails_job_and_counts_failure(cfg):
    cfg = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    async with httpx.AsyncClient(transport=mock_transport(lambda r: httpx.Response(500))) as c:
        engine = Engine(cfg, c)
        items = await drain(engine.submit(make_request(cfg)))
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
        busy = engine.submit(make_request(cfg, cls="background"))
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
        queued = engine.submit(make_request(cfg))
        engine.cancel(queued)
        assert engine.schedulers["sim-strong"].queued() == [] and engine.jobs == {}

    passthrough = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    slow = SlowStream(n_tokens=1000, delay=0.01)
    transport = mock_transport(
        lambda r: httpx.Response(200, stream=slow, headers={"content-type": "text/event-stream"})
    )
    async with httpx.AsyncClient(transport=transport) as c:
        engine = Engine(passthrough, c)
        running = engine.submit(make_request(passthrough))
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
        job = engine.submit(make_request(cfg, tenant="tenant-b", cls="interactive"))
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
        engine.submit(make_request(cfg))
    assert engine.jobs == {}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gateway/metrics.py`**

```python
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
        tpot = 0.0
        if result.output_tokens > 1:
            tpot = (result.finished_at - result.first_token_at) / (result.output_tokens - 1)
        self.tpot.labels(cls, name).observe(tpot)
        met = ttft <= req.cls.ttft_target_s and tpot <= req.cls.tpot_target_s
        if met:
            self.goodput.labels(cls, name).inc()
        return met
```

- [ ] **Step 4: Implement `gateway/engine.py`**

```python
"""Connects router, schedulers, dispatcher and telemetry with one dispatch loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
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
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests/test_engine.py -v`
Expected: 8 passed.

If `test_quiet_tenant_is_not_starved_by_noisy_tenant` shows a different order, print `engine.vtc._counters` after each completion. The expected sequence: tenant-a is served first (name tie-break at 0); tenant-a is then charged 100 + 10, so tenant-b (still 0) goes next; then they alternate.

- [ ] **Step 6: Run the whole suite**

Run: `make test`
Expected: all unit tests pass.

- [ ] **Step 7: Commit**

```bash
make fmt
git add -A
git commit -m "feat: engine dispatch loop, re-routing, expiry and gateway metrics"
```

---

### Task 12: HTTP API and entry point

**Files:**
- Create: `gateway/api.py`, `gateway/main.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `Engine`, `Job`, `Done`, `Failed` (Task 11); `analyze`, `BadRequest` (Task 3); `NoBackend` (Task 9); `QueueFull` (Task 7); `load_config`, `apply_env_overrides` (Task 2).
- Produces:
  - `create_app(cfg, engine) -> FastAPI`, serving:
    - `POST /v1/chat/completions` (streaming and non-streaming)
    - `GET /metrics`
    - `GET /healthz`
    - response headers `X-Gateway-Backend` and `X-Gateway-Class`
  - `tenant_keys(cfg) -> dict[str, TenantConfig]`
  - `python -m gateway.main`

- [ ] **Step 1: Write the failing tests `tests/test_api.py`**

```python
from contextlib import asynccontextmanager

import httpx

from gateway.api import create_app
from gateway.engine import Engine
from tests.conftest import mock_transport, only_backends, sse_response, with_class

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


async def test_rejects_unknown_key(cfg):
    async with gateway(cfg) as (client, _):
        response = await client.post("/v1/chat/completions", json=BODY)
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


async def test_metrics_and_health(cfg):
    async with gateway(only_backends(cfg, "sim-strong")) as (client, _):
        await client.post("/v1/chat/completions", json=BODY, headers=AUTH_A)
        metrics = await client.get("/metrics")
        health = await client.get("/healthz")
    sample = 'gateway_requests_total{cls="standard",backend="sim-strong",outcome="ok"} 1.0'
    assert sample in metrics.text  # labels appear in declaration order: cls, backend, outcome
    assert health.json() == {"status": "ok", "backends": ["sim-strong"]}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gateway.api'`.

- [ ] **Step 3: Implement `gateway/api.py`**

```python
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
```

- [ ] **Step 4: Implement `gateway/main.py`**

```python
"""Process entry point: `GW_CONFIG=config/sim.yaml python -m gateway.main`."""

from __future__ import annotations

import logging
import os

import httpx
import uvicorn

from gateway.analyzer import approx_tokens, qwen_token_counter
from gateway.api import create_app
from gateway.config import apply_env_overrides, load_config
from gateway.engine import Engine


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = apply_env_overrides(load_config(os.environ.get("GW_CONFIG", "config/sim.yaml")))
    count_tokens = (
        qwen_token_counter() if os.environ.get("GW_TOKENIZER") == "qwen" else approx_tokens
    )
    decision_log = open(cfg.decision_log, "a") if cfg.decision_log else None
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=5.0),
        limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
    )
    engine = Engine(cfg, client, count_tokens=count_tokens, decision_log=decision_log)
    uvicorn.run(
        create_app(cfg, engine),
        host=os.environ.get("GW_HOST", "127.0.0.1"),
        port=int(os.environ.get("GW_PORT", "8080")),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: 8 passed.

If `test_streaming_response` hangs, check the httpx version in `uv.lock`. `ASGITransport` buffers streaming responses fully, which is fine here because the mock upstream ends.

- [ ] **Step 6: Commit**

```bash
make fmt
git add -A
git commit -m "feat: OpenAI-compatible API with key-scoped classes and metrics endpoint"
```

---

### Task 13: Integration against the simulator, and the hosted smoke check

**Files:**
- Create: `tests/integration/__init__.py` (empty), `tests/integration/test_sim.py`, `scripts/smoke_hosted.py`

**Interfaces:**
- Consumes: `create_app`, `Engine`, `load_config`.
- Produces: integration tests run with `make integration`; `scripts/smoke_hosted.py` checks that a provider returns usage in its stream (spec §9).

- [ ] **Step 1: Write `tests/integration/test_sim.py`**

```python
"""Needs `make sim-up`. Functional checks only: the simulator is FIFO, so no latency claims."""

import asyncio
import json

import httpx
import pytest

from gateway.api import create_app
from gateway.engine import Engine
from gateway.telemetry import parse_vllm_metrics
from tests.conftest import only_backends

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
        await client.aclose()
        await engine.stop()
    lines = [line for line in response.text.splitlines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    assert any(json.loads(line[5:]).get("choices") for line in lines[:-1])


async def test_cancel_frees_the_simulator_slot(cfg):
    cfg = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    engine = Engine(cfg, httpx.AsyncClient(timeout=httpx.Timeout(60.0)))
    from tests.conftest import make_request

    job = engine.submit(make_request(cfg, max_tokens=2000))
    await asyncio.wait_for(job.out.get(), 10)  # first chunk
    assert await _running(8001) >= 1
    engine.cancel(job)
    for _ in range(50):
        if await _running(8001) == 0:
            break
        await asyncio.sleep(0.1)
    await engine.stop()
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
    finally:
        await client.aclose()
        await engine.stop()
    assert [r.status_code for r in responses] == [200] * 6
```

- [ ] **Step 2: Run the integration tests**

Run: `make sim-up && sleep 5 && make integration`
Expected: 4 passed.

If `test_cancel_frees_the_simulator_slot` fails because the simulator does not abort on client disconnect, keep the test. Mark it:

```python
@pytest.mark.xfail(reason="llm-d-inference-sim does not abort on disconnect; checked on real vLLM in Plan 3")
```

Record this in the README. Do not delete the assertion.

If `test_gateway_absorbs_simulator_429s` fails with 503s, the gateway is exhausting its deadline. Raise the standard class's `ttft_target_s` in a local `with_class` copy for this test only. Do not change `config/sim.yaml`.

- [ ] **Step 3: Write `scripts/smoke_hosted.py`**

```python
"""One-off check (spec §9): does a hosted provider return token usage in its stream?

Usage: GROQ_API_KEY=... uv run python scripts/smoke_hosted.py \
         [base_url] [model] [api_key_env]"""

import json
import os
import sys

import httpx


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "https://api.groq.com/openai/v1"
    model = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-oss-20b"
    key = os.environ[sys.argv[3] if len(sys.argv) > 3 else "GROQ_API_KEY"]
    body = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Say hello."}],
    }
    usage = None
    headers = {"Authorization": f"Bearer {key}"}
    url = f"{base}/chat/completions"
    with httpx.stream("POST", url, json=body, headers=headers, timeout=30) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            data = line[5:].strip() if line.startswith("data:") else ""
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage
    print(f"{model}: usage in stream = {usage}")
    return 0 if usage else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the smoke check (user, needs a Groq key)**

Run: `GROQ_API_KEY=<key> uv run python scripts/smoke_hosted.py`
Expected: prints a usage dict and exits 0.

If it exits 1, the dispatcher's counted-token fallback is what accounting uses for Groq. Note that in the README.

For Gemini, run `GEMINI_API_KEY=<key> uv run python scripts/smoke_hosted.py https://generativelanguage.googleapis.com/v1beta/openai <model id> GEMINI_API_KEY`.

- [ ] **Step 5: Manual end-to-end run (user)**

```bash
export GW_KEY_A=key-a GW_KEY_B=key-b GW_KEY_C=key-c
make run   # in one terminal
curl -s localhost:8080/v1/chat/completions -H "Authorization: Bearer key-b" \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":20}' -D - | head -20
curl -s localhost:8080/metrics | grep gateway_requests_total
```

Expected: HTTP 200 with `x-gateway-class: interactive` and an `x-gateway-backend` header, and a non-zero `gateway_requests_total`.

**Checkpoint C (user explains back):**
1. Run the same curl with `GW_SCHEDULING=passthrough make run` and with `GW_ROUTING=cheapest make run`. Explain what changed and why, using the decision log. Set `decision_log: decisions.jsonl` in a local copy of the config to see it.
2. Explain why the simulator cannot show any difference between passthrough and gateway scheduling for latency.

- [ ] **Step 6: Commit**

```bash
make fmt
git add -A
git commit -m "test: integration against llm-d-inference-sim and hosted usage smoke check"
```

---

## After this plan

- Run `make lint && make test && make integration`, then ask the user to review the `feat/gateway-core` branch.
- The user decides whether to create the public GitHub repo and push. Pushing is outward-facing and is not part of this plan.
- Next: Plan 2 (benchmark harness, `HYPOTHESES.md`, `infra/account`, `infra/gpu`), written against the interfaces above.
