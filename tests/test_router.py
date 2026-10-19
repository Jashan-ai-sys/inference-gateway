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


def test_request_too_large_for_kv_is_permanent(cfg):
    cfg = only_backends(with_backend(cfg, "sim-strong", kv_capacity_tokens=1000), "sim-strong")
    with pytest.raises(NoBackend) as err:
        decide(cfg, make_request(cfg, cls="standard", prompt=2000, max_tokens=50))
    assert err.value.status == 400
    assert err.value.rejected == {"sim-strong": "kv_capacity"}


def test_request_too_large_for_hosted_tpm_is_permanent(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    with pytest.raises(NoBackend) as err:
        decide(cfg, make_request(cfg, cls="background", prompt=9000, max_tokens=50))
    assert err.value.status == 400
    assert err.value.rejected["groq-gpt-oss-20b"] == "rate_limit_size"


def test_passthrough_does_not_shed_on_predicted_deadline(cfg):
    cfg = only_backends(cfg.model_copy(update={"scheduling": "passthrough"}), "sim-strong")
    cfg = with_backend(cfg, "sim-strong", base_ttft_s=100.0)
    decision = decide(cfg, make_request(cfg, cls="interactive"))
    assert decision.backend == "sim-strong"
    assert decision.candidates[0].predicted_ttft > 100.0
