# Intelligent AI Inference Gateway — design spec

- **Date:** 2026-10-19
- **Status:** approved section by section in conversation; awaiting review of this written spec
- **Owner:** Jashanprit Singh
- **Diagram:** [../../architecture.svg](../../architecture.svg)
- **Research inputs:** [../../research/](../../research/) (inference-technique catalogue and mapping)

## 1. Goal

Build a portfolio project that proves inference-engineering and cloud skills for Associate AI Engineer, Inference AI Engineer and Cloud AI Engineer roles. It is an OpenAI-compatible gateway with two cooperating engines:

- a **Model Router** that decides *which* backend serves a request, from model capability, quality, cost and predicted latency;
- a **Workload Scheduler** that decides *when* a request gets compute, from service class, deadline, per-tenant fairness and replica capacity;

plus a **benchmark harness** that measures, on a real GPU, where scheduling should live: in the gateway, inside vLLM, both, or in separate replica pools.

### What makes it different

Both halves exist elsewhere. Routers (RouteLLM, LiteLLM, OpenRouter auto, vLLM Semantic Router) pick a model but ignore backend queue state, deadlines and tenants. Schedulers (llm-d Flow Control, NVIDIA Dynamo router queueing) already ship priority bands, tenant fairness and deadline ordering, but only pick an endpoint inside one model pool and never route to hosted APIs. This project:

1. **Couples the two.** The router ranks backends using the scheduler's predicted queue wait against the request's deadline, so an interactive request that would miss its deadline on the local GPU can overflow to a hosted API.
2. **Publishes a measured comparison** of six scheduling placements on the same workload.

The README must name llm-d, Dynamo, LiteLLM and RouteLLM, reuse their vocabulary, and state that the scheduler features re-implement known ideas in order to measure them. No novelty claims beyond the two points above.

### Success criteria

- All six scheduling arms and four routing arms run from one command each, against both the simulator and real vLLM.
- A results table with real GPU measurements, confidence intervals and full provenance is committed.
- Every scheduling claim in the README comes from real vLLM runs, not the simulator.
- Total AWS spend at or under the $15 budget alert.
- Public repo with measured results within about 2 weeks.
- Every question in `docs/interview-prep.md` is answered with this project's own measured numbers.

## 2. Scope

### In v1

| Component | Path | Responsibility |
|---|---|---|
| API | `gateway/api` | FastAPI, OpenAI-compatible `POST /v1/chat/completions` (streaming and non-streaming), `GET /metrics`, `GET /healthz`. API key maps to tenant and allowed classes. Responses carry `X-Gateway-Backend` and `X-Gateway-Class`. Per-tenant fairness comes from VTC and queue limits, not separate per-tenant rate limits. |
| Analyzer | `gateway/analyzer` | Local input-token count (Qwen tokenizer; character-based estimate for hosted models, corrected from returned usage), class and deadline assignment, output-token estimate. |
| Config and registry | `gateway/config` + one YAML per environment (`config/sim.yaml`, later one per GPU setup) | Classes; static backend facts (model, base URL, quality score per task type, prices, context limit, KV capacity, privacy flags, rate limits, performance priors); tenants. `GW_SCHEDULING` and `GW_ROUTING` environment variables switch benchmark arms. |
| Telemetry | `gateway/telemetry` | Polls each replica's `/metrics` every 1 s; parses hosted-API rate-limit headers; keeps TTFT/TPOT/error EWMAs per backend. |
| Router | `gateway/router` | Hard-constraint filter, predicted TTFT, weighted score, overflow, decision log. |
| Scheduler | `gateway/scheduler` | Per-backend class queues, VTC tenant fairness, aging, admission control, shedding. |
| Dispatcher | `gateway/dispatcher` | One httpx client per backend; streaming; timeouts; cancellation; 429/503 as backpressure; per-backend circuit breaker. |
| Metrics | `gateway/metrics` | Gateway Prometheus metrics, including per-class goodput. |
| Benchmark | `bench/` | Workload YAML, arm configs, GuideLLM orchestration, analysis, charts, `HYPOTHESES.md`, `RESULTS.md`. |
| Infra | `infra/account`, `infra/gpu` | Terraform: budget alert; one-off GPU stack. |
| Tests | `tests/` | Unit and integration tests (§9). |

**Stack:** Python 3.12, FastAPI, httpx, pydantic, prometheus_client, pytest, uv, Terraform. No database: decisions go to a JSONL log, metrics to Prometheus exposition files captured per run.

### Not in v1

- Our own continuous-batching simulator (first item after v1; see §11).
- Prefill/decode disaggregation; cross-model KV-cache reuse (HeteroFold arXiv:2609.32259, linear mapping arXiv:2608.03893, RaReCache arXiv:2610.11358).
- Learned router (contextual bandit or RouteLLM-style classifier). The scoring function is the plug-in point.
- Deploying the gateway itself to ECS/ALB; autoscaling; UI.

## 3. Backends

| Backend | Where | Role | Notes |
|---|---|---|---|
| `llm-d-inference-sim` (v1.0.0, Apache-2.0) | Docker on the Mac, CI, and the GPU box | Functional development and tests | Speaks the OpenAI API and vLLM metric names. FIFO worker pool only: no continuous batching, KV limit, preemption or priority, and returns 429 when its queue is full. Never used for scheduling claims. |
| vLLM `v0.32.0` (pinned image `vllm/vllm-openai:v0.32.0`) | AWS g6.4xlarge, 1× L4 24 GB | Real measurements | Two servers on one GPU, started sequentially. |
| Groq `openai/gpt-oss-20b` | Hosted | Fast, cheap overflow target | ~1,000 tok/s, $0.075/$0.30 per 1M tokens; free plan 30 RPM, 8K TPM. |
| Gemini Flash (exact model from config) | Hosted | Stronger, slower option | Free tier trains on data, so excluded for tenants marked `sensitive`. Free-tier limits read from AI Studio into config, not hard-coded. |

### Models on the GPU

- **Headline pair (Pair A):** `Qwen/Qwen3-1.7B` (BF16, small and fast) and `Qwen/Qwen3-8B-AWQ` (strong). Both Apache-2.0, ungated.
  - Strong at `--gpu-memory-utilization 0.55`, about 37k KV tokens (estimate).
  - Small at `0.35`, about 24k KV tokens (estimate).
  - Thinking mode off for interactive traffic: `chat_template_kwargs: {enable_thinking: false}`.
  - `--max-model-len 8192` set explicitly.
- **Scheduler-stress pair (Pair B):** `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct-AWQ`, which have a roughly 3× larger KV pool, so queueing effects come from the scheduler rather than KV exhaustion.
- Real KV capacity is read from each server's startup log ("GPU KV cache size: N tokens") into the registry, then pinned with `--kv-cache-memory-bytes` so every arm gets identical capacity.

## 4. Request lifecycle

### 4.1 Analyse

| Class | TTFT target (deadline = arrival + target) | Queue limit | Output estimate (p90) |
|---|---|---|---|
| `interactive` | 1 s | 64 | 300 tokens |
| `standard` | 5 s | 256 | 800 tokens |
| `background` | 60 s | 1024 | 2,000 tokens |

Values live in `config/classes.yaml` and are tuned by the benchmark. The class comes from the API key's configuration, never from a client header; a key may allow several classes and pick one per request via a body field, validated against that allow-list.

### 4.2 Filter (hard constraints)

A backend is eligible only if every rule holds:

1. prompt tokens + `max_tokens` fit its context limit;
2. its quality score for the request's task type is at least the class minimum (`min_quality` per class in `config/classes.yaml`; scores come from M6 and default to 0 until measured, so the rule is a no-op before then);
3. its privacy flags allow the tenant;
4. its rate-limit budget (RPM/TPM token bucket) has room;
5. its predicted TTFT (§4.3) is within the remaining deadline.

If nothing is eligible, the request is rejected with 429 (capacity) or 400 (no backend can ever serve it, e.g. context too long), with the reason in the body.

### 4.3 Predict

For each eligible backend `b` and class `c`:

- `queue_wait(b, c)` = queued tokens ahead at class `c` or higher on `b` ÷ measured token throughput of `b`;
- `predicted_ttft(b)` = `queue_wait(b, c)` + prefill estimate for the **uncached** tokens of this prompt on `b` + base TTFT EWMA of `b`.

The scheduler exposes `queue_wait`; the router consumes it. This is the coupling point.

**Model-switch cost (session affinity).**
- A multi-turn conversation is identified by an optional `session_id` body field, scoped to the tenant. The gateway remembers which backend served the session's last turn and how many tokens that turn's prompt plus output contained.
- On that backend, the cached prefix is assumed to be those tokens (vLLM prefix caching; it may have been evicted, so this is an optimistic estimate, corrected by the measured TTFT EWMA). On any other backend, including a different model, nothing is cached: KV caches are model-specific, so the whole conversation is prefilled again.
- So the uncached-token term makes the router keep a conversation on its current model unless switching clearly wins on the other terms. The decision log records the switch cost whenever the router switches anyway.
- This penalty is exactly what cross-model KV-cache reuse (HeteroFold, RaReCache) would shrink. The README reports the measured switch cost (M1) as the upper bound on what those methods could save for this workload.

### 4.4 Rank

`J(b) = α·cost + β·latency + γ·quality_risk + δ·error_rate`, each term min-max normalised to [0, 1] across the eligible set. Weights per class in `config/classes.yaml` (interactive weights latency; background weights cost). Lowest `J` wins. The decision log records the request id, tenant, class, every candidate's terms and score, the winner and the reason.

### 4.5 Queue

Each backend has one bounded queue per class.

- **Across classes:** strict priority with aging. A request that has waited `aging_fraction` (default 0.5) of its class TTFT target is promoted one class, once. Aging must fire before the deadline, which is the full target; otherwise a request would expire at the moment it became eligible for promotion.
- **Within a class:** VTC (Sheng et al., OSDI '24). Each tenant has a counter. Next request = earliest-queued request of the backlogged tenant with the lowest counter.
  - Charge `1 × prompt_tokens` at dispatch and `2 × output_tokens` as tokens stream (from SSE chunks, else final usage).
  - **Counter lift:** when a tenant becomes backlogged again, set its counter to `max(own, min counter of currently backlogged tenants)`. This prevents saved-up credit from starving others; it gets a dedicated test.

### 4.6 Admit and dispatch

A self-hosted replica accepts the next request only if both hold:

1. in-flight requests < replica cap (Little's-law start: measured throughput × e2e latency near the M0 saturation knee; it is a benchmark knob);
2. reserved KV tokens + (prompt tokens + p90 output for the class) ≤ 0.9 × KV capacity.

**KV capacity:**
- Capacity is vLLM's reported `num_gpu_blocks × block_size` (from the startup log or `vllm:cache_config_info`). It is never estimated as "GPU memory minus weights", which ignores activations, CUDA graphs, the runtime and fragmentation.
- A request's reservation is rounded up to whole blocks (block size 16), because vLLM allocates in blocks.
- For documentation and sanity checks, KV bytes per token = `2 × num_hidden_layers × num_key_value_heads × head_dim × bytes`, read from the model's `config.json`. It uses `num_key_value_heads`, not `num_attention_heads`; under GQA the difference is 4× or more. Example, Qwen3-8B in BF16: 2 × 36 × 8 × 128 × 2 = 144 KiB per token.

The reservation ledger is released on completion and corrected every second from `vllm:kv_cache_usage_perc` (a 0–1 fraction) and `vllm:num_requests_waiting`. Hosted backends use a token bucket from their rate limits instead.

### 4.7 Failure handling

| Event | Behaviour |
|---|---|
| Class queue full | 429 with `Retry-After` |
| Top-class request has waited `aging_fraction` of its target | Re-route once to another backend if that backend's predicted TTFT still meets the deadline; otherwise it stays queued |
| Deadline passes while queued | 503, no GPU time spent |
| Replica returns 429 or 503 | Backpressure: requeue at the front of its class, mark replica saturated briefly; not an error |
| Replica connection error or 5xx | Circuit breaker counts it; after 3 consecutive failures, backend ineligible for 10 s |
| Client disconnects | Close the upstream stream; vLLM aborts the request and frees its KV |
| Upstream timeout | Per-class timeout; cancel upstream, return 504 |

### 4.8 vLLM-priority arms

Only in arms 3 and 4, the gateway sends `priority` in the request body: interactive = 0, standard = 1, background = 2 (lower runs sooner in vLLM; the opposite of llm-d/GKE). Verified on v0.32.0:

- `--scheduling-policy priority` orders the **waiting** queue by (priority, arrival).
- It does **not** preempt a running request to admit a higher-priority one. It only picks the lowest-priority running request as the victim when KV blocks run out.
- Under `fcfs`, a non-zero `priority` is **silently ignored**.

So an integration test asserts actual serving order on a priority-policy server.

## 5. Gateway metrics

Prometheus metrics, all labelled by class and backend (tenant where it is cheap):

- queue wait; queue depth;
- admits, rejects (by reason), sheds;
- re-routes and overflows;
- TTFT and TPOT measured at the gateway;
- tokens in and out;
- cost;
- VTC counters;
- **goodput:** requests meeting both the TTFT and TPOT targets of their class (DistServe definition).

## 6. Benchmark

### 6.1 Workload (`bench/workloads/*.yaml`, fixed seeds)

- **Tenants:** A sends 60% of load (noisy); B and C are quiet.
- **Classes:**

| Class | Share of requests | Prompt tokens | Output tokens |
|---|---|---|---|
| interactive | 30% | 200–800 | 50–300 |
| standard | 50% | 1–2k | 300–800 |
| background | 20% | 6–8k | 1–2k |

- **Sessions:** half of interactive requests are follow-up turns in conversations of 3–6 turns, sent with a `session_id`, so the switch-cost signal is exercised.
- **Arrivals:** Poisson at 0.7×, 1.0× and 1.3× of the saturation rate measured in M0.
- **Prompts:** synthetic length-controlled prompts for scheduling runs; GSM8K, an MBPP subset and simple chat for router quality.
- **Pinned decoding:** scheduling and calibration runs use `temperature 0`, a fixed `max_tokens` drawn from the class's output range by seed, and `ignore_eos: true`. Output lengths are then identical across arms, so differences come from scheduling, not from the model stopping at different points. Router-quality runs use greedy decoding without `ignore_eos`.
- **Load generator:** GuideLLM, one process per class and tenant, with the API key setting tenant and class. `vllm bench serve` directly against vLLM as a sanity baseline.

### 6.2 Arms

**Scheduling arms** (same workload and seeds):

| # | Arm | vLLM server config |
|---|---|---|
| 1 | FIFO passthrough | `--scheduling-policy fcfs` |
| 2 | Gateway queues + admission | `fcfs` |
| 3 | vLLM-native priority | `priority` |
| 4 | Gateway + vLLM priority | `priority` |
| 5 | Separate replica pools (interactive vs the rest) | two servers on one GPU |
| 6 | vLLM native admission | `--max-num-queued-reqs` / `--max-num-queued-tokens` (HTTP 503) |

Arm 5 isolates memory but shares compute, because both engines use the same SMs. The README must say this understates what two real GPUs would give.

**Routing arms:** always the strongest model; cheapest eligible; static task rules; adaptive (live load, cost and quality).

### 6.3 Experiments on the GPU (about 6 h including setup; M7 stretch adds about 20 min)

| Id | Experiment | Approx. GPU minutes |
|---|---|---|
| M0 | Calibration: TTFT vs prompt length; TPOT and throughput vs concurrency (where does throughput flatten, and why); KV usage vs context length; GPU utilisation and memory-bandwidth utilisation per phase via `nvidia-smi`/DCGM; saturation rate | 45 |
| M1 | Prefix caching on vs off, plus per-tenant `cache_salt`; **model-switch cost**: continue a 2k- and an 8k-token conversation on the same model (prefix hit) vs on the other model (full prefill), TTFT difference | 35 |
| M2 | Scheduling arms 1–6 at 0.7×, 1.0× and 1.3× load (the main result) | 110 |
| M3 | Chunked-prefill budget sweep: `--max-num-batched-tokens` 512/2048/8192, plus `--long-prefill-token-threshold` | 35 |
| M4 | BF16 vs FP8 vs FP8 + FP8 KV cache (`--quantization fp8`, `--kv-cache-dtype fp8`) | 40 |
| M5 | Speculative decoding (n-gram, then EAGLE-3 if available) vs concurrency | 40 |
| M6 | Router eval set: record every eval prompt's answer from each model once; replay the routing arms offline | 15 |
| M7 (stretch) | One Nsight Systems trace of a decode step at batch 1 and batch 32: kernel timeline, launch gaps, and the effect of CUDA graphs (`--enforce-eager` vs default) | 20 |

M4 reports weight FP8 and KV-cache FP8 as separate rows, because they solve different problems: weight FP8 cuts the bytes read per decode step; KV FP8 doubles KV capacity. M4 and M5 use an 8B model in BF16 that fits alone on the GPU. Exact checkpoint names and flags are confirmed with `vllm serve --help` on the pinned version before the run.

### 6.4 Metrics reported

Per class:

- TTFT p50/p95/p99, TPOT p50/p95, deadline-miss rate, goodput;
- background throughput (tokens/s);
- Jain's fairness index across tenants and max service gap;
- 429/503 rate;
- `vllm:num_preemptions_total`;
- cost per successful request;
- quality vs the eval set.

TTFT and TPOT are measured client-side and cross-checked against `vllm:time_to_first_token_seconds` and `vllm:request_time_per_output_token_seconds`.

TTFT is always reported split in two: **gateway queue wait** (from the gateway's own timestamps) and **engine TTFT** (dispatch to first token, which includes vLLM's own queue time, `vllm:request_queue_time_seconds`, and prefill). This makes it visible where waiting moved between arms. Definitions used everywhere: TTFT = arrival to first token; TPOT = (e2e − TTFT) ÷ (output tokens − 1) per request; ITL = gap between consecutive tokens.

### 6.5 Rigour

- `bench/HYPOTHESES.md` is committed before the first GPU run; results are judged against it.
- Discard 60 s warm-up; at least 3 min steady state per data point.
- Headline points repeated 3× with different seeds.
- Report the median with a 95% bootstrap confidence interval. Claim a difference only when intervals do not overlap.
- Every result row records the vLLM version, resolved AMI ID, model, server flags, gateway git commit and seed.
- Prometheus metrics are captured to files per run.
- No LLM judge; quality uses exact match (GSM8K) and unit tests (MBPP).

### 6.6 Outputs

- Raw data in `bench/results/<run-id>/`.
- `bench/RESULTS.md` with tables, matplotlib charts, and a "what we found / what we could not test" section.
- `docs/interview-prep.md`: each experiment mapped to the interview questions it answers, with this project's own measured numbers. Seed questions:

| Question | Answered by |
|---|---|
| Throughput rises 100 → 180 → 290 → 300 tok/s as batch goes 8 → 64; why does it flatten? | M0 |
| GPU utilisation is 92% in prefill but 45% in decode; why? | M0 |
| An 8,000-token prompt and a 100-token prompt arrive together and the short one gets a high TTFT; why, and what fixes it? | M2, M3 |
| TTFT rose from 200 ms to 2 s after enabling continuous batching; why? | M3 |
| How many 4k-token requests fit on a 24 GB GPU with an 8B model? | §4.6 arithmetic, M0 |
| When does speculative decoding hurt? | M5 |
| Does FP8 help prefill, decode, or capacity? | M4 |

### 6.7 README accuracy rules

- Use the L4's real specifications from NVIDIA's datasheet: 24 GB GDDR6 (not HBM), about 300 GB/s memory bandwidth, native FP8 tensor cores (Ada, compute capability 8.9). Verify against the datasheet before publishing.
- Model dimensions come from each model's `config.json`, never from tutorial examples.
- Common errors in tutorial material that must not be repeated:
  - Under GQA, W_K and W_V are `hidden × (num_kv_heads × head_dim)`, e.g. 4096 × 1024 for 8 KV heads, not 4096 × 4096.
  - SwiGLU feed-forward layers have three weight matrices (gate, up, down), not two; Llama-3-8B's intermediate size is 14336 (11008 is Llama-2-7B).
  - Do not claim attention stores an O(N²) score matrix in memory; vLLM's FlashAttention-style kernels do not materialise it.
  - Softmax worked examples must be recomputed (a common temperature example gets the arithmetic wrong).
- Roofline numbers are derived, not asserted: decode at batch 1 reads all weights once per token, so the floor is weight bytes ÷ memory bandwidth (≈ 16 GB ÷ 300 GB/s ≈ 53 ms per token for an 8B BF16 model).

## 7. AWS infrastructure

### 7.1 Prerequisites (runbook step 0)

1. Account on the Paid plan.
2. Quota "Running On-Demand G and VT instances" = 16 vCPU in us-east-1, filed on day 1. Confirm the quota code with `aws service-quotas list-service-quotas`; do not hard-code it.
3. Groq and Gemini API keys.

### 7.2 `infra/account/` (applied once, kept)

AWS Budgets alert at $15 with email notifications at 50%, 80% and 100%.

### 7.3 `infra/gpu/` (per run, destroyed after)

**Network:**
- VPC with one public subnet, internet gateway and route table. No NAT gateway, which costs more for no security gain here.
- Security group with zero inbound rules.

**Instance:**
- g6.4xlarge: 16 vCPU, 64 GiB, 1× L4. vLLM needs about 3 physical cores per server; g6.xlarge cannot host even one server.
- AMI from SSM parameter `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id`; the resolved id is recorded in results.
- IMDSv2 required; encrypted 100 GB gp3 root volume holding weights and the image.

**IAM:** role with `AmazonSSMManagedInstanceCore`, plus `ssm:GetParameter` on `/inference-gateway/*` only.

**User-data:** verify Docker and the NVIDIA container toolkit (install if missing); pull `vllm/vllm-openai:v0.32.0` and the `llm-d-inference-sim` image.

**Housekeeping:**
- Tag `Project=inference-gateway` on everything.
- Pin versions: Terraform `~> 1.16`, provider `hashicorp/aws ~> 6.69`.

### 7.4 Operation

- **Access:** `aws ssm start-session`, port forwarding for demos. No SSH.
- **Secrets:** Groq and Gemini keys are SecureString parameters under `/inference-gateway/`. Never in tfvars, git or user-data.
- **Make targets:**
  - `make gpu-up` creates the stack.
  - `bench/gpu/serve.sh <config>` starts the servers for one arm, sequentially when there are two, waiting on the first server's `/health`.
  - `make bench ARM=<arm>` runs one arm.
  - `make gpu-stop` stops the instance between sessions.
  - `make gpu-down` runs `terraform destroy` and then a check script that fails if any tagged instance, volume or IP remains.
- **Load generation:** the gateway and load generator run on the GPU box, never from India.

### 7.5 Cost estimate (us-east-1 on-demand, before verification on the bill)

| Item | Estimate |
|---|---|
| ~6 GPU h × $1.3232 | $7.94 |
| EBS for about 4 days | ~$1 |
| Public IPv4 while running | ~$0.03 |
| **Total before GST** | **~$9** |
| **With 18% GST** | **~$10.6 (~₹900)** |

GST is not covered by AWS credits, and credits may not cover GPU instances.

## 8. CI

GitHub Actions on every push, with no AWS credentials:

- ruff;
- unit tests;
- integration tests against the `llm-d-inference-sim` container;
- `terraform fmt -check` and `terraform validate`.

## 9. Tests

| Kind | What it checks |
|---|---|
| Unit | VTC ordering, counter lift on a returning idle tenant, aging promotion, admission arithmetic, ledger correction, router filter rules, score normalisation, predicted-wait formula, uncached-token prefill estimate (same backend vs other model), session tracking scoped per tenant, class from API key (header spoofing rejected) |
| Integration (simulator) | Streaming passthrough, cancellation frees upstream, 429/503 treated as backpressure, queue-full 429, deadline-expiry 503, circuit breaker, overflow to a stub hosted backend |
| Integration (real vLLM, on the GPU box) | Priority-policy server serves requests in the expected order; `priority` ignored under fcfs; usage present in the final stream chunk |
| Hosted (manual, one-off) | Groq and Gemini return usage in streams; fall back to a local estimate if not |

## 10. Build plan (about 2 weeks, about 3 h on weekdays and 5 h at weekends)

Each build stage has a study item first, and its checkpoint is explained back before the next stage starts, so every part of the system can be explained from first principles. Study items are about 45–60 minutes and come out of the daily time budget.

| Days | Study first | Build | Checkpoint (explained back) |
|---|---|---|---|
| 1 | — | AWS step 0, repo, uv, CI, simulator on the Mac, `HYPOTHESES.md` draft | CI green; simulator answers |
| 2–3 | Prefill vs decode; KV-cache sizing by hand for Qwen3-8B; TTFT/TPOT/ITL definitions | API, dispatcher, telemetry; integration tests vs simulator and Groq | **A:** passthrough streams; cancel frees upstream. Compute KV bytes per token and max concurrent 4k requests unaided. |
| 4–5 | Continuous batching; PagedAttention and blocks; read vLLM v0.32.0 `vllm/v1/core/sched/scheduler.py` (waiting/running queues, token budget, preemption) | Scheduler: queues, VTC, aging, admission, caps, shedding | **B:** quiet tenants keep their fair share under a noisy tenant (simulator). Explain why the gateway must hold requests back for its priorities to matter. |
| 6–7 | Prefix caching (block hashing); chunked prefill and its TTFT-vs-ITL trade-off; why KV caches are model-specific (skim HeteroFold/RaReCache abstracts) | Router: registry, filter, predicted wait, model-switch cost, score, overflow, decision log; gateway metrics (about +1 day, taken from the day-14 buffer) | **C:** all arms run end-to-end on the simulator. Answer the 8,000-vs-100-token question. |
| 8 | Weight vs KV quantisation (FP8 on Ada); speculative decoding and acceptance rate; roofline arithmetic | Benchmark harness and analysis; full runbook dry-run on the simulator; `infra/gpu` validated; hypotheses frozen | **D:** one command produces a results table; Terraform plan clean. Write the M0–M5 hypotheses themselves. |
| 9–11 | Review own hypotheses before each session | GPU sessions: M0+M1, M2, M3–M7; `make gpu-down` + leftover check | **E:** real results committed with provenance |
| 12–13 | — | `RESULTS.md`, `docs/interview-prep.md`, README (positioning, trade-offs, explain-tier techniques, §6.7 accuracy rules), final diagram | **F:** public repo ready; every question in `interview-prep.md` answered aloud with the project's own numbers |
| 14 | — | Buffer | |

**Cut order if late:** M7, then M5, then M3, then arm 6. Study items and checkpoint explanations are not cut. The scheduler and the router–scheduler coupling are never cut.

### Risks and fallbacks

- **GPU quota not approved by day 9:** run the same Docker commands on a rented L4 elsewhere; apply the Terraform once the quota arrives.
- **Weak or null results:** report them with confidence intervals and the reason.
- **Two-server startup flakiness:** start the servers sequentially; pin KV size with `--kv-cache-memory-bytes`.

## 11. After v1

1. Our own continuous-batching simulator, about 300 lines:
   - per-step token budget and chunked prefill;
   - paged KV blocks with prefix hashing;
   - recompute preemption;
   - fcfs and priority policies.

   Validate it against the M0, M2 and M3 GPU numbers and publish the error table. Prior art: Vidur (MLSys '24), BLIS.
2. Cross-model KV-cache reuse research extension: reproduce RaReCache (arXiv:2610.11358, which itself tested Qwen3-0.6B → 14B) on this project's Qwen3-1.7B → Qwen3-8B pair, outside vLLM (Hugging Face transformers). Measure answer quality and TTFT against the fraction of tokens recomputed, and compare the saving with the switch cost measured in M1. First check whether the authors released code; if not, implement the rank-disagreement selection from the paper. Compare with HeteroFold (arXiv:2609.32259) if time allows.
3. Then, by interest: KV-event-precise prefix routing; prefill/decode disaggregation; ECS deployment of the gateway.

## 12. Explain-only topics (README trade-offs section and interview prep)

- Tensor, pipeline, data, expert and context parallelism.
- Prefill/decode disaggregation (DistServe, Splitwise) and Llumnix-style live migration.
- FlashAttention 2/3/4, FlashInfer, CUDA graphs, fused kernels.
- RadixAttention vs vLLM automatic prefix caching.
- GQA, MLA and sliding-window effects on KV size.
- INT8 / GPTQ / NVFP4 hardware fit.
- Medusa and MTP.
- Multi-LoRA serving.
- KV offload (LMCache).
- Autoscaling on queue and KV metrics.

Details and sources are in `docs/research/inference-techniques-mapping.md`.

## 13. Open items (non-blocking)

- Confirm on the box: L4 usable memory (`nvidia-smi`), per-instance overhead from vLLM logs, Docker presence on the AMI.
- Confirm hosted-provider stream usage and Gemini free-tier limits.

## 14. References

**Papers:**
- VTC, Sheng et al., OSDI '24, arXiv:2401.00588
- Sarathi-Serve, OSDI '24, arXiv:2403.02310
- DistServe, OSDI '24, arXiv:2401.09670
- Splitwise, ISCA '24, arXiv:2311.18677
- Llumnix, OSDI '24, arXiv:2406.03243
- QLM, SoCC '24, arXiv:2407.00047
- RouteLLM, arXiv:2406.18665
- FrugalGPT, arXiv:2305.05176

**Systems:** llm-d (Flow Control, inference-sim), NVIDIA Dynamo, Kubernetes Gateway API Inference Extension, LiteLLM, vLLM Semantic Router, GuideLLM.

**vLLM v0.32.0 source:** `vllm/v1/core/sched/scheduler.py`, `vllm/config/scheduler.py`, `docs/configuration/optimization.md`.
