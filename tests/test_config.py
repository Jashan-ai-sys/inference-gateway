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
