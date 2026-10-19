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
