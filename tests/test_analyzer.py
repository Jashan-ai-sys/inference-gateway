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
    body = _body(
        service_class="standard",
        task_type="code",
        session_id="s1",
        temperature=0,
        model="evil-model",
        priority=-100,
        n=1,
        tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice="auto",
        parallel_tool_calls=True,
        logprobs=True,
        top_logprobs=2,
    )
    req = analyze(body, _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")
    assert req.task_type == "code" and req.session_id == "s1"
    assert req.body == {"messages": body["messages"], "temperature": 0}


def test_token_counts_and_output_estimate(cfg):
    req = analyze(_body(max_tokens=5000), _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")
    assert req.prompt_tokens == 100 + 4  # 400 chars / 4, plus 4 template tokens per message
    assert req.max_tokens == 5000
    assert req.output_estimate == 800  # capped at the standard class p90


def test_max_tokens_defaults_to_class_p90(cfg):
    req = analyze(_body(), _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")
    assert req.max_tokens == 800


def test_list_text_parts_are_counted():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "c"}]}
    ]
    assert prompt_text(messages) == "ab\nc"


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
        {"role": "user", "content": [{"type": "text", "text": "ab", "extra": "x" * 10000}]},
        {"role": "user", "content": "hi", "name": "x" * 10000},
        {"role": "tool", "content": "hi"},
        {"role": "user", "content": {"text": "hi"}},
        {"role": "user"},
    ],
)
def test_uncountable_message_content_is_rejected(message):
    # Everything forwarded upstream must be counted, or admission and fairness can be bypassed.
    with pytest.raises(BadRequest):
        prompt_text([message])


@pytest.mark.parametrize("body", [{}, {"messages": []}, {"messages": "hi"}])
def test_bad_messages_rejected(cfg, body):
    with pytest.raises(BadRequest):
        analyze(body, _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")


def test_non_positive_max_tokens_rejected(cfg):
    with pytest.raises(BadRequest):
        analyze(_body(max_tokens=-1), _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")


@pytest.mark.parametrize("bad", ["abc", "5", 1.5, True, [], 0, -1])
def test_bad_max_tokens_rejected(cfg, bad):
    with pytest.raises(BadRequest, match="max_tokens"):
        analyze(_body(max_tokens=bad), _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")


def test_max_completion_tokens_preferred_over_max_tokens(cfg):
    req = analyze(
        _body(max_tokens=100, max_completion_tokens=300),
        _tenant(cfg, "tenant-a"),
        cfg,
        approx_tokens,
        0,
        "r",
    )
    assert req.max_tokens == 300


@pytest.mark.parametrize("text", [None, 5])
def test_text_part_with_non_string_text_is_rejected(text):
    with pytest.raises(BadRequest):
        prompt_text([{"role": "user", "content": [{"type": "text", "text": text}]}])


@pytest.mark.parametrize("messages", [["hi"], [None], [5]])
def test_non_dict_message_rejected(cfg, messages):
    with pytest.raises(BadRequest):
        analyze({"messages": messages}, _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")


@pytest.mark.parametrize("n", [2, 0, True, "1"])
def test_n_other_than_one_rejected(cfg, n):
    with pytest.raises(BadRequest, match="n must be 1"):
        analyze(_body(n=n), _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")


def test_non_bool_stream_rejected(cfg):
    with pytest.raises(BadRequest, match="stream"):
        analyze(_body(stream="false"), _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")


def test_non_string_service_class_rejected(cfg):
    with pytest.raises(BadRequest, match="service_class"):
        analyze(_body(service_class=5), _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")


@pytest.mark.parametrize("body", [None, ["messages"], "messages"])
def test_non_object_body_rejected(cfg, body):
    with pytest.raises(BadRequest):
        analyze(body, _tenant(cfg, "tenant-a"), cfg, approx_tokens, 0, "r")
