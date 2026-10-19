# Intelligent AI Inference Gateway

A model router coupled to a workload scheduler, in front of vLLM and hosted LLM APIs,
with a benchmark of where scheduling should live. Design: `docs/superpowers/specs/`.

    make install && make test        # unit tests
    make sim-up && make integration  # tests against llm-d-inference-sim
