> **Status (2026-10-19):** research input, not the design. Produced by a research workflow; estimates are not measurements.
>
> **Correction from the vLLM v0.32.0 source check:** `--scheduling-policy priority` does NOT preempt a running request to admit a waiting higher-priority one. It (a) orders the waiting queue by (priority, arrival) and (b) picks the lowest-priority running request as the victim when KV blocks run out. Adjust the M2 hypothesis accordingly. Metric names below were verified on v0.32.0: `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc` (0–1), `vllm:time_to_first_token_seconds`, `vllm:request_time_per_output_token_seconds`, `vllm:num_preemptions_total`. v0.32.0 also adds native admission control: `--max-num-queued-reqs`, `--max-num-queued-tokens` (HTTP 503).

# Inference Gateway: technique-to-project mapping

## Assumptions

These are my defaults. Each one changes some numbers further down.

**GPU: L4 on g6.2xlarge, not A10G.**
- **Why L4:**
  - L4 is Ada (SM89), so it has native FP8 tensor cores. A10G is Ampere and runs FP8 as Marlin weight-only.
  - L4 has about 300 GB/s of memory bandwidth, against about 600 GB/s on A10G. The lower bandwidth makes memory-bound decode effects large and easy to see.
  - It is a little cheaper.
- **Cost of choosing L4:** decode is about 2x slower, so each run takes longer.
- **Why the 2xlarge size:** it has 8 vCPU and 32 GB RAM. The xlarge size has 4 vCPU, and the gateway, the load generator and vLLM's API server would compete for those cores.
- **Prices:** about $0.80/h (g6.xlarge) to $0.98/h (g6.2xlarge) on demand in us-east-1. These are from memory, so check them.
- **Quota:** new AWS accounts usually have a quota of 0 for "Running On-Demand G and VT instances". Request the increase several days before you need it.

**Models: the Llama family.**
- Strong model: Llama-3.1-8B-Instruct.
- Small model: Llama-3.2-3B-Instruct.
- Draft model: Llama-3.2-1B-Instruct.
- Why Llama:
  - All three share a tokenizer, so the 1B works as a draft model.
  - A published EAGLE-3 head exists (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`; check the exact name).
  - FP8 runs online with `--quantization fp8`, so no extra checkpoint download is needed.
  - Llama is gated on Hugging Face. Accept the license before renting the GPU.
- Fallback without the gate: Qwen2.5-7B / 1.5B / 0.5B, which also share a tokenizer.

**KV arithmetic for this setup.** Use these numbers in the README and in interviews.
- **Llama-3.1-8B KV size:** 32 layers × 8 KV heads × 128 head dim × 2 (K and V) × 2 bytes = 128 KiB per token.
- **8B memory budget on L4:**
  - About 22.5 GiB is usable. At 0.92 utilisation that is about 20.7 GiB.
  - Subtract about 15 GiB of BF16 weights and about 1.5 GiB for activations and CUDA graphs.
  - That leaves about 4 GiB of KV, or roughly 30k tokens. That is only about 7 concurrent 4k-token requests.
- **With FP8 weights:** KV grows to about 11 GiB, or about 90k tokens.
- **With FP8 KV as well:** about 180k tokens.
- **Llama-3.2-3B:** 112 KiB per token and about 6 GiB of weights.
- **Decode floor for 8B BF16 on L4:**
  - Each step reads about 16 GB of weights. At 300 GB/s that takes at least 53 ms, so at most about 19 tokens/s per stream.
  - Achieved bandwidth is usually 75–85% of peak, so expect about 15–16 tokens/s.
  - All of this is approximate. Phase M0 will measure the real numbers.

**vLLM version.**
- Pin one vLLM version.
- The catalogue says v0.32.0. I cannot confirm several of its flags (for example `--kv-offloading-size`, `--performance-mode`, `--max-num-queued-reqs`, turboquant).
- Before renting the GPU, run `vllm serve --help` on the pinned version (a CPU box or Docker is fine) and check every flag below.

---

## 1. BUILD: what the gateway implements

**Principle.** The gateway decides admission, order and placement. vLLM decides batching. The gateway should never batch requests itself, because continuous batching inside vLLM already does that better.

**B1. In-flight concurrency cap per replica. This is the thing that makes gateway scheduling matter.**
- If the gateway forwards every request at once, vLLM's own FCFS waiting queue decides the order, and the gateway's priority logic does nothing.
- So the gateway holds requests back. It keeps in-flight requests at or below a cap, set from Little's law (cap ≈ target throughput × e2e latency, started near the measured knee of `--max-num-seqs`).
- Each replica's vLLM waiting queue should stay near zero. The cap is a benchmark knob: too low wastes throughput, too high hands control back to vLLM.

**B2. Admission control on token headroom.**
- Keep a ledger per replica of reserved KV tokens.
- Reservation per request = prompt tokens (counted with the model's HF tokenizer) + min(max_tokens, p90 output length for its class).
- Capacity = `num_gpu_blocks × block_size`, read from the `vllm:cache_config_info` metric or the startup log.
- Correct the ledger every 1–2 s from `vllm:kv_cache_usage_perc` and `vllm:num_requests_waiting`. The first metric was called `gpu_cache_usage_perc` in older versions, so check the name.
- If a request does not fit, queue it. If its queue is full, return 429 with Retry-After, or spill over to a hosted API.
- Tradeoff: reserving the full max_tokens is safe but leaves KV idle. Reserving the p90 runs hotter but risks preemption. Recommendation: use p90, and measure the preemption count to show the choice works (M2).

**B3. Workload classes with deadlines.**
- Three bounded queues: interactive, standard and background.
- Each request gets a deadline = arrival time + its class's TTFT SLO.
- Dispatch is strict priority across classes, with aging so background cannot starve. Within a class, dispatch is earliest-deadline-first.
- Requests whose deadline has passed are dropped before dispatch, so no GPU time is spent on them.
- In the "vLLM priority" and "both" arms, also forward vLLM's `priority` field (0/1/2).
- Uncertain: I believe vLLM rejects a non-zero `priority` unless the server runs with `--scheduling-policy priority`. Leave the field out in the fcfs arms.

**B4. Per-tenant fairness on tokens, not requests.**
- Inside each class, use a virtual token counter per tenant: weighted fair queueing where cost = prompt tokens + w × generated tokens.
- This follows VTC, from Sheng et al., "Fairness in Serving Large Language Models", OSDI 2024.
- Benchmark it with a noisy tenant sending about 60% of the load. Report Jain's fairness index and the p95 TTFT of the quiet tenants.

**B5. Bounded queues and load shedding.**
- Set a maximum depth per class, and shed background requests first.
- Shedding is reported as a metric, not hidden.
- Without bounds, an overload run at 1.3x capacity grows queues without limit, and p99 then only reflects how long the run lasted.

**B6. Cancellation propagation.**
- When a client disconnects or a deadline is cancelled, close the upstream stream so vLLM aborts the request and frees its KV.
- This is small to build and easy to discuss in an interview. Many demo gateways leak this.

**B7. Model router.**
- Registry fields: context length, tool/JSON support, price per token, latency EWMA, error rate, and region or residency.
- A hard-constraint filter runs first. Then a weighted score ranks the remaining models: α·cost + β·latency + γ·quality-risk.
- Adaptive arm: update the latency and error EWMAs from live traffic. Estimate quality-risk from a cheap prompt-difficulty feature (length, task type, presence of code or maths) trained on the eval set captured in M6.
- Prior work: RouteLLM (arXiv 2406.18665) and FrugalGPT (arXiv 2305.05176).
- **Honesty flag:** without a labelled eval set, the four routing arms cannot be compared on quality. Then only cost and latency get measured, and "cheapest-eligible" trivially wins.

**B8. Prefix-affinity routing (approximate).**
- Hash the tokenized prompt's first N full 16-token blocks as a chain, the same way vLLM does.
- Send each request to a replica with rendezvous or consistent hashing, with bounded load (Mirrokni et al. 2018): if the preferred replica is above (1+ε) × average load, fall back to the next one.
- It is only meaningful with 2 or more replicas: the simulator, or 2 small-model instances on the GPU (M2).
- Flag: this is approximate. A precise version would subscribe to `--kv-events-config` ZMQ events. List that as future work, not v1.

**B9. Per-tenant cache_salt.**
- Pass vLLM's `cache_salt` request field per tenant, so tenants cannot use cross-tenant prefix-cache timing side channels.
- This is one line of code, a security point for interviews, and something M1 measures.

**B10. Observability and goodput.**
- Prometheus metrics from the gateway:
  - queue wait per class
  - admits and rejects
  - shed counts
  - TTFT and TPOT measured at the gateway (they include gateway overhead)
  - goodput = requests meeting both the TTFT and TPOT SLO, per class (the DistServe definition)
- Goodput is the headline metric, not raw throughput.

**B11. Static-batching and FIFO baselines in the simulator.**
- These are a few lines in the simulator, not the gateway.
- They give the "FIFO → static batching → continuous batching" ladder from a continuous-batching teaching design, and show why continuous batching exists.

---

## 2. MEASURE: vLLM knobs on one GPU, ranked by interview value per GPU-hour

**How to run every experiment:**
- Point every runbook at the simulator first. The simulator must expose the same OpenAI API and the same `/metrics` names as vLLM, so no GPU time is spent debugging scripts.
- Keep the models and the compile cache on a persistent 100 GB gp3 EBS volume. Stop the instance between sessions; plan 3 sessions of about 2 hours each.
- Each config change means a vLLM restart. Budget about 1.5–3 min with a warm compile cache and 4–5 min cold (my estimate). Group sweeps so each server config is started once.
- Each data point gets at least 2–3 min of steady-state load after warm-up.
- p99 from about 500 requests is noisy. Report p95 as the main number, give p99 with a bootstrap CI, and repeat headline points twice.
- Load generation: the project's own load generator through the gateway, plus `vllm bench serve` directly against vLLM as a sanity baseline. Check its dataset names (`random`, `prefix_repetition`) on your version.

| Rank | Experiment | GPU min | Tier |
|---|---|---|---|
| 0 | M0 Calibration and saturation curve | ~35 | Prerequisite |
| 1 | M1 Prefix caching on/off, plus cache_salt | ~25 | Core |
| 2 | M5 Speculative decoding vs concurrency | ~40 | Core |
| 3 | M4 Quantisation: BF16 vs FP8 vs FP8+FP8-KV | ~40 | Core |
| 4 | M2 Scheduling arms + admission vs preemption + pools + prefix routing | ~110 | Core (the project's thesis) |
| 5 | M3 Chunked-prefill token budget sweep | ~35 | Core |
| 6 | M6 Router eval-set capture | ~15 | Core |
| 7 | M7 CUDA graphs vs eager | ~15 | Stretch |
| 8 | M4b AWQ INT4 | ~15 | Stretch |

**Budget:**
- Core: about 300 min of experiments plus about 40 min of setup and debugging ≈ 5.7 h.
- If you need to cut, drop M3 to 2 configs (-15) and M5's EAGLE-3 arm (-15). That gives about 5.2 h.
- Stretch items fit only if earlier sessions run under budget.
- Cost at about $1/h is roughly $6–8. Set an AWS budget alarm anyway.

**M0. Calibration: saturation curve and cost model (8B BF16 and 3B). About 35 min.**
- **Hypothesis:**
  - Prefill time grows roughly linearly with prompt tokens (compute-bound). Rough check: 8B at about 50% MFU on 121 BF16 TFLOPS gives about 3.5–4k prompt tokens/s.
  - Decode step time stays almost flat as batch size rises up to some knee, because the weight read dominates (memory-bound). After the knee it rises with batch size and total context.
- **Metrics that move:** TTFT vs input length (128 to 8k, concurrency 1); TPOT and aggregate output tokens/s vs concurrency (1 to 128); `num_gpu_blocks`.
- **Flags:** defaults only. Note the `max_num_batched_tokens` and `max_num_seqs` values the startup log reports.
- **Why it ranks first:** it fits the simulator's cost model, gives a roofline-style plot, and puts real numbers behind "decode is memory-bound". Every other result depends on it.

**M1. Automatic prefix caching on/off, plus cache_salt. About 25 min, 2 restarts.**
- **Hypothesis:**
  - With a shared 2k-token system prompt on about 80% of requests, APC cuts TTFT p50 several-fold and raises throughput.
  - A different salt per tenant drops the hit rate to the share of traffic within each tenant.
- **Metrics:** TTFT p50/p95; `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`.
- **Flags:** default is on. Turn it off with `--no-enable-prefix-caching`; the catalogue lists `--disable-prefix-caching`, so check which spelling your version accepts. Salt is set per request with `"cache_salt"`.
- **Why this rank:** cheap, a dramatic result, and it is the reason B8 exists.

**M5. Speculative decoding vs concurrency (8B). About 40 min, 3 configs.**
- **Hypothesis:**
  - At concurrency 1–4, n-gram speculation cuts TPOT about 1.5–2.5x on copy-heavy tasks (summarise or edit a provided text) and gives little on open chat.
  - EAGLE-3 helps on both.
  - At concurrency 64 or more, the gain disappears or turns negative, because decode stops being bandwidth-bound and verification compute is wasted.
- **Metrics:** TPOT p50 at concurrency {1, 4, 16, 64} × 2 datasets; acceptance rate from `vllm:spec_decode_num_accepted_tokens` / `vllm:spec_decode_num_draft_tokens`.
- **Flags:**
  - `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":4}'`
  - `--speculative-config '{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":3}'`
- The draft-model arm (1B) is optional.
- **Why this rank:** "when does spec decoding hurt?" is a classic interview question, and this answers it with a measurement.

**M4. Quantisation: BF16 vs online FP8 vs FP8 plus FP8 KV (8B). About 40 min, 2 extra restarts.**
- **Hypothesis:**
  - On L4, FP8 W8A8 roughly halves the weight read, so TPOT at concurrency 1 improves about 1.6–1.9x.
  - It also frees about 7 GiB, which roughly triples KV token capacity, and prefill gets somewhat faster.
  - FP8 KV doubles capacity again.
  - Quality loss is small.
- **Metrics:** TPOT at concurrency {1, 8, 32}; prefill tokens/s; `num_gpu_blocks`; the concurrency at which preemptions start; GSM8K accuracy on 200 questions.
- **Flags:** `--quantization fp8`; `--kv-cache-dtype fp8`.
- **Honesty flags:**
  - 200 GSM8K questions give about ±3.5 points of CI, so only large quality drops will show.
  - On A10G the result would differ: weight-only Marlin, so no prefill gain.
- **M4b, stretch (about 15 min):** AWQ INT4 with `--quantization awq_marlin` (checkpoint such as `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`; check the name). Expect the best TPOT at concurrency 1 and no prefill gain.

**M2. Scheduling arms: the project's thesis. About 110 min.**
- **Model and memory:** Llama-3.2-3B BF16. Cap KV with `--kv-cache-memory-bytes` so the single instance and the two-pool setup get equal total KV; this creates real KV pressure.
- **Load:** a mix of about 30% interactive, 50% standard and 20% background; 3 tenants, one of them noisy; Poisson arrivals at 0.7x, 1.0x and 1.3x of the saturation rate measured in M0.
- **Arms:**

| Server config | Arms run on it | Approx. time |
|---|---|---|
| `--scheduling-policy fcfs` | FIFO passthrough; gateway queues (B1–B5); admission on vs off at 1.3x | ~45 min |
| `--scheduling-policy priority` | vLLM-native priority (passthrough with `priority`); both | ~30 min |
| Two instances, each at `--gpu-memory-utilization 0.45`, started one after the other | Separate pools (interactive vs the rest); round-robin vs prefix-affinity (B8) | ~25 min |

- **Hypothesis:**
  - Gateway queues cut interactive p95 TTFT vs FIFO under overload.
  - vLLM-native priority adds preemption of running background requests. That further helps interactive requests but raises `vllm:num_preemptions` (shown as `_total` in Prometheus) and background e2e latency.
  - "Both" wins for interactive goodput.
  - Admission control keeps preemptions near zero at 1.3x.
- **Metrics:** goodput per class; interactive p95 TTFT; background throughput and starvation; Jain's fairness index; preemptions; shed rate.
- **Honesty flag for the pools arm:** the two engines share the same SMs, so this arm measures queue and KV isolation, not compute isolation. It will understate what two real GPUs would give. Say so in the README.

**M3. Chunked-prefill token budget sweep (8B BF16). About 35 min, 3 restarts.**
- **Hypothesis:** when long prompts (4–8k) arrive into a stream of short chats:
  - a small `--max-num-batched-tokens` (512) keeps decode ITL p99 low but slows TTFT for the long prompts
  - a large one (8192) does the reverse
  - `--long-prefill-token-threshold` limits how much one long prompt can take per step
- **Metrics:** ITL p99 of short requests; TTFT p95 of long requests; output tokens/s.
- **Flags:** `--max-num-batched-tokens {512, 2048, 8192}`; one run with `--long-prefill-token-threshold 1024`.
- **Flag:** chunked prefill may not truly turn off in the V1 engine, so the budget sweep is the real experiment. Do not build on `--disable-chunked-prefill`.

**M6. Router eval-set capture. About 15 min.**
- Run about 300 prompts once through both 3B and 8B: GSM8K, an MBPP subset and simple chat.
- Record each answer, its latency and its token counts. Score offline: exact match for GSM8K, unit tests for MBPP. Avoid an LLM judge, or say plainly that it is biased if you use one.
- Then all four routing arms are replayed with zero further GPU time. This is a "record once, replay offline" design; say so in the README.

**M7. CUDA graphs vs `--enforce-eager`. Stretch, about 15 min.**
- **Hypothesis:** eager mode is much slower in TPOT at batch 1–8 (CPU launch overhead) and the gap narrows at large batch.
- It turns kernel-launch overhead into a measured number.

**Optional, if time remains:** KV offload to CPU RAM with `--kv-offloading-size 16`. This flag name is not confirmed. Expected result: higher prefix hit rate when a multi-turn working set does not fit in HBM.

---

## 3. EXPLAIN: too heavy for one GPU, so README design trade-offs and interview prep

- **Tensor parallelism:** each layer's matrices are split across GPUs with an all-reduce per layer. It cuts per-token latency and fits bigger models, but needs NVLink-class links and duplicates KV once TP exceeds the KV-head count (8 for Llama-8B).
- **Pipeline parallelism:** layers are split across GPUs or nodes with cheap point-to-point transfers. It is how a model larger than one node is served, at the cost of pipeline bubbles and higher per-token latency.
- **Data parallelism and replica scaling:** this is the cluster version of what the gateway already does. More replicas give throughput, not latency, and placement (load vs prefix locality, B8) becomes the hard part.
- **Expert parallelism:** MoE experts are spread across GPUs with all-to-all dispatch. Expert load imbalance and network latency dominate, so EPLB is used to rebalance.
- **Context parallelism (prefill and decode):** work is split along the sequence for very long contexts. It reduces TTFT and KV duplication, and only pays off at long context.
- **PD disaggregation:** prefill and decode run on separate pools that are tuned for TTFT and ITL independently. M3 shows the interference problem it removes, and the costs are KV transfer bandwidth and tuning the P:D ratio (DistServe, Splitwise).
- **KV-aware routing with live cache events:** the precise version of B8 (llm-d, Dynamo, vllm-router `cache_aware`). Plumbing replaces drift, and hotspots still need load balancing.
- **Autoscaling on queue and KV metrics:** the gateway already exports the right signals (waiting queue, KV usage, goodput). Scaling lags because cold starts take minutes, and scaling down throws away warm prefix caches.
- **Multi-LoRA serving:** one base model serves many tenant adapters in the same batch. It fits the multi-tenant story well, but is left out of v1.
- **KV offload / LMCache:** the KV cache spills to CPU RAM, disk or remote stores. It only wins when reload is faster than recompute, which means long reused prefixes.
- **FlashAttention 2/3/4, FlashInfer, fused kernels, torch.compile:** kernel-level wins that the simulator folds into calibrated constants. Use them as the bridge from CUDA fundamentals: tiling into shared memory, avoiding HBM round-trips, and why FA3 needs Hopper.
- **RadixAttention vs vLLM APC:** a token-granular radix tree vs a block-granular hash chain. Same goal, different trade-offs between eviction and scheduling.
- **GQA, MLA and sliding-window attention:** KV size is set at training time. Show the arithmetic: Llama-8B's GQA uses 8 KV heads against 32 query heads, which makes KV 4x smaller.
- **INT8 (SmoothQuant), GPTQ, NVFP4/MXFP4:** the right choice depends on hardware. NVFP4 needs Blackwell for native speed, and INT8 is the choice on Ampere.
- **Distillation, Medusa, MTP:** distillation is training-time. Medusa is a legacy method superseded by EAGLE. MTP only works on models trained with MTP heads.

---

## 4. What the simulator must model, and what it can ignore

**Must model, or the results are meaningless:**
- **Iteration-level continuous batching.** Each step is limited by `max_num_seqs` and a token budget (`max_num_batched_tokens`). Requests join and leave at every step.
- **V1 step order.** Running decodes are scheduled first, then prefill chunks fill the rest of the budget (chunked prefill).
- **Step-time cost model, fitted from M0:**
  - `t_step = a + b·prefill_tokens + c·num_decode_seqs + d·Σ(context_len of decodes)`
  - `a` covers fixed overhead (CPU, launch, sampling).
  - The weight-read floor shows up as `a` when batch size is small.
  - Fit separate parameters for each (model, quantisation) profile.
- **Paged KV blocks.** Block size 16, capacity = the calibrated `num_gpu_blocks`, and blocks allocated as tokens grow.
- **Preemption by recompute when blocks run out.** Under fcfs the most recently admitted running request is evicted; under priority the lowest-priority one. The evicted request goes back to the waiting queue and pays its prefill again.
- **Prefix caching:** block-chain hash hits, LRU over free cached blocks, and cache_salt. Needed for M1 parity and for B8.
- **Both `fcfs` and `priority` policies, and request aborts.**
- **Unknown output lengths.** Sample them from a distribution per class. Neither the engine nor the gateway knows a request's output length in advance, and that is what makes B2 hard.
- **Streaming timestamps:** TTFT, per-step ITL and e2e latency, in the same `/metrics` names as vLLM.
- **Hosted-API backends:** a TTFT distribution, a tokens/s rate, 429 rate limits and price per token.
- **Multiple replicas, for the routing and pools arms.**

**Can ignore, because the effect is folded into calibrated constants or is out of scope:**
- attention and GEMM kernel choice
- fused kernels and torch.compile
- CUDA graphs (fold into `a`; padding to capture sizes can be ignored)
- quantisation numerics (a separate calibrated profile, not simulated)
- speculative decoding (left out, or at most an acceptance-rate multiplier on decode; label it as such)
- async scheduling
- detokenization CPU cost
- TP, PP, EP and CP internals
- PD disaggregation
- KV offload
- LoRA
- gateway-to-replica network latency (a small constant)

**Validation gate. This is the honesty anchor.**
- Hold out 2–3 GPU configs from M2 and M3 that were not used for fitting.
- The simulator should predict TTFT p50/p95 and throughput within about ±15–20%.
- Publish the error table in the README. If the simulator misses, say where (preemption-heavy regimes are the likely place).
- Prior art to cite: Vidur (Agrawal et al., MLSys 2024).

---


## Uncertain items: check before relying on them

1. vLLM flag names and defaults on the pinned version: prefix-caching negation, `priority` rejected under fcfs, metric names and `_total` suffixes, the default `max_num_batched_tokens`, `--kv-offloading-size`.
2. Hugging Face repo names for the EAGLE-3 head and the AWQ checkpoint.
3. All GPU-minute figures, restart times, speedup ranges and AWS prices are estimates, not measurements.
4. Two vLLM engines on one GPU can be fiddly at startup (memory profiling). Start them one after the other and leave headroom.
5. The quality-risk router signal is the weakest part of the design. Without the M6 eval set, report the routing arms on cost and latency only and say so.