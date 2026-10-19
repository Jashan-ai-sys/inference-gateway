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
