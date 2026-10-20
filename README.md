# Intelligent AI Inference Gateway

A model router coupled to a workload scheduler, in front of vLLM and hosted LLM APIs,
with a benchmark of where scheduling should live. Design: `docs/superpowers/specs/`.

    make install && make test        # unit tests
    make sim-up && make integration  # tests against llm-d-inference-sim

## Running locally

    export GW_KEY_A=key-a GW_KEY_B=key-b GW_KEY_C=key-c   # tenant API keys (Bearer scheme)
    export GW_METRICS_TOKEN=ops-token                      # GET /metrics needs it; unset -> 404
    make sim-up && make run                                # gateway on :8080

    curl -s localhost:8080/v1/chat/completions -H "Authorization: Bearer key-b" \
      -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":20}' -D -
    curl -s localhost:8080/metrics -H "Authorization: Bearer ops-token" | grep gateway_requests_total

`GW_SCHEDULING=passthrough` and `GW_ROUTING=cheapest` switch policy without editing the config.
`scripts/smoke_hosted.py` checks that a hosted provider returns token usage in its stream (needs
an API key; not yet run).

Known simulator gap: `test_cancel_frees_the_simulator_slot` is marked xfail. llm-d-inference-sim
keeps a request in `num_requests_running` until its generation finishes, even after the client
disconnects (it passes only when the random output happens to be short). Slot release on cancel
is checked on real vLLM in Plan 3.
