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
