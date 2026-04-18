# Cacheon (SN14) — Architecture & Engineering Notes

This document is the single place for architecture decisions, incentive mechanism design, build sequence, and lessons learned for **Cacheon** (Bittensor subnet 14). Update it when something changes or when a decision gets revisited.

## Current state

| Item     | Value                                               |
| -------- | --------------------------------------------------- |
| Phase    | 5 — Validator + API (next up)                       |
| Model    | Qwen2.5-7B-Instruct                                 |
| Hardware | H100 80GB SXM (production), RTX 4090 (dev)          |
| Runtime  | HuggingFace Transformers (monkey-patched attention) |

Phases 1 (Harness), 2 (Scoring), 3 (Sandbox), and 4 (Prompts) are complete. See [Build sequence](#build-sequence) for status of each phase.

## System overview

One validator (the subnet owner). One eval pipeline. One person setting weights.

```
BITTENSOR CHAIN
  │
  │  miner registers: {"model": "hf/repo", "revision": "sha"}
  │  one commitment per hotkey — permanent, no re-do
  ▼
CPU SERVER  ← remote_validator.py (not yet built — Phase 5)
  │
  │  [Main loop, ~every 360s]
  │  1. fetch metagraph + all revealed commitments
  │  2. sandbox-precheck new submissions (AST + subprocess — Phase 3)
  │  3. identify king (lowest score in state) and new challengers
  │  4. if no new challengers → set_weights(king), sleep, loop
  │  5. if challengers exist → SSH to GPU pod with eval job
  │
  │  ── SSH/SFTP ──────────────────────────────────────────────
  ▼
GPU POD  ← pod_eval.py (not yet built — Phase 5)
  │
  │  no wallet keys, no chain access
  │  load Qwen2.5-7B-Instruct
  │  generate prompts (block-hash seeded PG19 — Phase 4)
  │  for each challenger:
  │    baseline = harness.run(PassthroughPolicy, prompts)
  │    miner    = harness.run(MinerPolicy, prompts)
  │    result   = scoring.score(baseline, miner)
  │  write results.json
  │
  └──────── results.json back over SSH ───────────────────────▶ CPU SERVER
  │
  │  6. did any challenger beat king? → new king
  │  7. set_weights(winner_uid = 1.0, all others = 0.0)
```

**Why the CPU/GPU split?** Security boundary within our own infrastructure. If the GPU pod is compromised, the attacker gets model weights and eval results — not wallet keys, not the ability to set weights. The two sides communicate over SSH: prompts and policy pointers in, a results JSON out.

**Why ~360s polling interval?** GPU eval takes minutes to hours — there's no benefit to reacting faster. One full scan per epoch (fetch metagraph + all revealed commitments, compare against known state) is sufficient. The loop runs immediately after a successful eval; 360s sleep only when idle (no challengers) or on chain/eval failures.

**Other validators on Cacheon (SN14):** Other validator hotkeys can exist on the subnet and set their own weights independently. In practice, nobody else is running independent eval — it would require the same GPU setup and matching eval methodology. The Bittensor protocol allows other validators to copy or follow the subnet owner's weights via chain consensus. The public API (leaderboard, scores) is for miners and community monitoring, not for weight-setting by other validators.

## Incentive mechanism

### King-of-the-hill / winner-take-all

One reigning champion. All emission goes to the king. Challengers must beat the king's score to dethrone them.

GPU eval only runs when new challengers exist. If no new miner has registered since the last round, the king keeps weight and no inference runs. This is the right cost model: GPU inference is expensive; running it every block for every registered miner would make the subnet uneconomical.

**Tradeoffs considered:**

- Top-N split (e.g. 60/25/10/5%) distributes incentive more broadly but weakens the signal — miners optimize for "good enough to be top-4" rather than "best." Winner-takes-all creates a sharper optimization target.
- Running eval on all miners every round would be more statistically robust but costs 10–100× more GPU time. King-of-the-hill amortizes that cost by only evaluating new entrants.

### One commitment per hotkey

Miners register once per hotkey. To submit a new policy, register a new hotkey. This creates commitment pressure: you can't iterate cheaply on the same registration. Miners are incentivized to test locally before committing.

**Why not allow re-submission on the same hotkey?** Unlimited resubmission turns the subnet into a free hyperparameter search service. One-shot creates real skin in the game.

### Scoring formula

```python
if kl_divergence > QUALITY_THRESHOLD:
    score = 0.0
else:
    score = (0.6 * memory_reduction) + (0.4 * latency_improvement)

QUALITY_THRESHOLD = 0.1   # nats — tunable
```

- **Hard KL gate, not soft penalty.** A soft penalty creates incentives to trade output quality for efficiency in ways that are hard to detect. Hard reject above threshold is cleaner and easier to reason about.
- **60/40 memory/latency split.** Memory is weighted higher because it's the primary bottleneck at scale — KV cache memory determines max context length and batch size. Latency matters but is secondary. Both weights are tunable once we see real submissions.
- **Memory measured by harness, not self-reported.** `torch.cuda.max_memory_allocated()` is used, not `policy.memory_bytes()`. Self-reported memory can be faked; GPU allocator measurement cannot. `memory_bytes()` exists as a cross-check and debugging aid — a large discrepancy flags the submission.
- **Long context required for signal.** Qwen2.5-7B uses GQA (`num_kv_heads=4` vs `num_attention_heads=28`), so its KV cache is already 7× smaller than full MHA. At 2K tokens the cache is ~113 MB (~0.7% of peak memory) — memory reduction is invisible. At 32K tokens the cache is ~1.8 GB (~12% of peak), which is large enough for compression gains to register. Prompts are truncated to ~131K chars (~32K tokens) to target this range. Decode runs 256 tokens to give eviction/compression policies enough steps to differentiate.

## Harness architecture

### HuggingFace, not vLLM

vLLM manages its own KV cache internally (PagedAttention). Replacing it would mean fighting undocumented internals that change between versions. HuggingFace models are plain Python — every attention layer is a `forward()` method you can swap at runtime.

Monkey-patching: replace each layer's `self_attn.forward` with a function that calls `policy.write()` and `policy.attend()` after Q/K/V projections and RoPE. Baseline and miner policy run on the same HF stack, so relative comparisons are valid.

**Qwen-specific gotchas (applies to Qwen2.5-7B-Instruct):**

- **GQA**: `num_key_value_heads != num_attention_heads`. Use the right count when reshaping after projection.
- **RoPE**: Qwen applies rotary position embeddings to Q and K after projection, before attention. Apply RoPE before calling `policy.write()` — otherwise the miner's cache stores unencoded keys and attention scores will be wrong.
- **Prefill shape**: `seq_len = full prompt length` during prefill, `seq_len = 1` during decode. `write()` handles both without special-casing.

### The `write` + `attend` interface shape

Three separate hooks (write, score, aggregate) would force score tensor materialization and create a circular dependency for eviction strategies:

- H2O-style eviction needs attention weights to track token importance. Attention weights are computed during scoring. If scoring is a separate hook that runs after `write`, the eviction decision in `write` has no access to the current step's attention weights.
- TurboQuant computes `Q @ K_compressed.T + QJL_correction → softmax → aggregate` in one fused call. Splitting scoring and aggregation forces materialization of the full `[batch, heads, 1, seq_len]` score tensor — which defeats the point at long context.

`write` + `attend` lets the miner own the full pass. The miner's object holds state across calls, so importance accumulators, codebooks, and rotation matrices are all accessible inside `attend` when needed.

### Single-operator eval in V1

One validator, one GPU pod, one person deciding weights. This is not a decentralization tradeoff — it's just the operational reality of V1. The subnet owner runs both the CPU server and the GPU pod.

The CPU/GPU split is a **security boundary**, not an architectural requirement for decentralization. It means a compromised GPU pod can't steal wallet keys or forge weight-setting.

**Why not let other validators run their own eval?** Different hardware produces different KL values. An H100 and an A100 running the same policy produce slightly different floating-point results. If multiple independent validators ran eval and set weights, they'd disagree on who's king and produce conflicting weight signals. Centralized eval on one reference machine eliminates that variance.

**Known tradeoff:** Trust is concentrated in the subnet operator. The mitigation path is publishing eval rollouts — prompts, baseline logits, KL values — to verifiable storage (e.g. Cloudflare R2) so anyone can audit results after the fact. Verify-after-the-fact rather than run-it-yourself. This is the V2 plan.

## Build sequence

Five phases. Each produces something runnable and validates assumptions before committing to the next.

| Phase         | Status  | What to build                                                                                                                                                                                                                          | Done when                                                                                                                                                                       |
| ------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 — Harness   | ✅ Done | Load Qwen2.5-7B, monkey-patch attention, passthrough policy, prefill + decode, return logits + latency + peak memory                                                                                                                   | Passthrough produces identical token-for-token output to unpatched HF on 5 prompts                                                                                              |
| 2 — Scoring   | ✅ Done | `scoring.py`: KL divergence, memory delta, latency delta → score dict                                                                                                                                                                  | KL between identical logits = 0; KL between baseline and degraded policy > 0; score formula matches hand-computed cases                                                         |
| 3 — Sandbox   | ✅ Done | AST analysis + subprocess isolation (`runner.py`; firejail on CPU validator host in prod — not installed by GPU `setup.sh`), import allowlist, output validation                                                                       | `import os` rejected before execution; subprocess jailed with no network / isolated FS / memory cap when firejail present; legitimate TurboQuant-style policy runs successfully |
| 4 — Prompts   | ✅ Done | Block-hash seeded PG19 sampling via `prompts.py`                                                                                                                                                                                       | Same block hash always produces same passage set; different hashes produce different sets                                                                                       |
| 5 — Validator | 🔲 Todo | `remote_validator.py` (CPU): chain scan, challenger selection, SSH to GPU pod, `set_weights()`. `pod_eval.py` (GPU): load policy, run harness + scoring, write results JSON. `GET /leaderboard`, `GET /scores/{hotkey}`, `GET /health` | `remote_validator.py` detects new challenger, runs eval, king is dethroned when beaten; leaderboard reflects current king                                                       |

**Don't skip phases.** Each one validates assumptions the next phase depends on. Sandbox bolted on after the fact has gaps.

### Phase 1 stop condition

Passthrough policy must produce **identical token-for-token output** to unpatched HuggingFace on 5 prompts. `harness.verify()` checks this. If it fails, the monkey-patch has a bug — check RoPE ordering and GQA reshape before continuing.

### Phase 3 — Sandbox layers

**Layer 1 — Static AST analysis (`sandbox.py`, before execution):**

- Parse submitted file with Python's `ast` module — no execution
- Import allowlist: `torch`, `numpy`, `math`, `einops`, `inference_engine` (only `.policy` submodule)
- Blocked bare calls: `eval`, `exec`, `compile`, `open`, `input`, `breakpoint`, `__import__`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`
- Blocked attribute targets: `os`, `sys`, `subprocess`, `socket`, `importlib`, `ctypes`, `cffi`, `builtins`, `__builtins__`, `pickle`, `shelve`
- Relative imports rejected
- Structural check: exactly one `KVCachePolicy` subclass with `setup`, `write`, `attend`, `memory_bytes`

Layer 1 is a **fast feedback tool** — miners get a clear error in milliseconds. It is not the security boundary.

**Layer 2 — OS-level subprocess isolation (`runner.py`, at execution):**

- When `firejail` is installed on the host running `runner` (intended: **CPU validator** in Phase 5; GPU `setup.sh` does not install it): `--net=none` (no network), `--private=<workdir>` (isolated filesystem), `--rlimit-as=8g` (memory cap), `--rlimit-nproc=64` (process limit)
- Falls back to bare subprocess on dev machines / CI / GPU-only hosts where firejail is absent (warning logged)
- Hard timeout: 300 seconds, process group kill
- Validates `AttentionOutput.output`: correct shape, dtype, no NaN/Inf, values in `[-100, 100]`, attention weights sum to 1

Layer 2 is the **security boundary**. Even if a miner finds a creative AST bypass, firejail prevents network access, filesystem reads, and resource exhaustion at the OS level.

**Design decisions:**

- `super()`, `type()`, `dir()` are intentionally _not_ blocked — they are standard Python patterns miners need, and firejail makes AST-level paranoia unnecessary.
- Triton is not on the V1 allowlist. Triton kernels can read arbitrary GPU memory and are harder to sandbox. PyTorch ops are sufficient to implement TurboQuant's algorithm correctly. Triton unlocks in V2 with additional isolation.
- **`--private` path constraint:** `--private=<workdir>` hides everything under `$HOME` inside the jail. GPU pod `setup.sh` installs the repo and venv under `/workspace/` (Targon persistent volume). Firejail is intended for the **CPU server** (Phase 5), not the GPU pod—if it ever wrapped a GPU-side worker, Python and the repo would need to avoid paths hidden by `--private`. When provisioning the CPU server, install Python system-wide (e.g. `/usr/bin/python3`) and the repo outside home (e.g. `/opt/cacheon`) so neither is shadowed by `--private`.

### Phase 4 — Prompts

`prompts.py` provides `sample_prompts(block_hash, n)` — deterministic PG19 sampling seeded by the on-chain block hash.

**Key parameters:**

- `max_chars = 131,072` (~32K tokens) — targets the range where the GQA KV cache is ~12% of peak memory, making compression gains visible in scoring.
- `min_chars = 1,000` — skips PG19 frontmatter, headers, and OCR fragments that would produce degenerate short-context evals.
- `revision` pinned to a specific PG19 commit hash, so row ordering is stable regardless of upstream dataset changes.

Passages are truncated at sentence or word boundaries to avoid mid-word splits that waste tokens.

**Why PG19 over FineWeb/MMLU/etc.?** KV cache benchmarking needs long, continuous sequences that force the cache to grow. PG19 is full novels — median passage is 100K+ tokens before truncation. Web-crawl datasets (FineWeb) are mostly short documents under 1K tokens, which produce trivially small caches where memory reduction is noise.

**V2 path:** if the memory axis is still noisy after launch, upgrade the model to Qwen2.5-14B (KV cache ~5× larger per token, fits on one H100) before adding datasets. The harness is already parameterized by `model_name`.

### Phase 5 — Validator + API

Phase 5 has two parts:

**Part A — Validator loop (`remote_validator.py` on CPU server):**

The main loop. Runs forever. No FastAPI — direct Python, no external API call in the weight-setting path.

1. Fetch metagraph + all revealed commitments (~every 360s)
2. Sandbox-precheck new submissions
3. Identify king (lowest score from state) and new challengers (hotkey:block not yet evaluated)
4. If no challengers → `set_weights(king, 1.0)`, sleep, loop
5. If challengers → SSH to GPU pod: send `prompts.json` + policy pointers
6. GPU pod runs harness + scoring, writes `results.json`, SSH back
7. Update state: did any challenger beat the king?
8. `set_weights(winner_uid = 1.0, all others = 0.0)` via `subtensor.set_weights()`

**Part B — Monitoring API (FastAPI, optional):**

Read-only. Miners and community check status. Does not participate in weight-setting.

| Endpoint               | What it does                                        |
| ---------------------- | --------------------------------------------------- |
| `GET /health`          | Service status, last eval block, current king.      |
| `GET /leaderboard`     | Current king and recent challenger history. Public. |
| `GET /scores/{hotkey}` | Score history for a specific miner. Public.         |

## Key decisions log

| Decision                            | Reasoning                                                                                                                                                                                                                                                                                    | Alternatives considered                                                                            |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| HuggingFace over vLLM               | vLLM's internal KV cache can't be replaced without fighting undocumented internals. HF gives exact hook points. Relative comparisons valid on same stack.                                                                                                                                    | vLLM: rejected — PagedAttention is not hookable cleanly                                            |
| Single-operator eval in V1          | One validator, one GPU pod, one person setting weights. Different hardware produces different KL values — one reference machine eliminates variance. CPU/GPU split keeps wallet keys off the inference pod. Other validators can exist on-chain but nobody else is running independent eval. | Decentralized: deferred to V2 via verifiable rollouts                                              |
| King-of-the-hill / WTA              | Sharper optimization target than top-N split. GPU eval only runs when new challengers exist — amortizes cost.                                                                                                                                                                                | Top-N split: considered, rejected — weakens incentive signal                                       |
| One commitment per hotkey           | Commitment pressure: miners test locally before committing. Clean on-chain record.                                                                                                                                                                                                           | Unlimited resubmission: turns subnet into free hyperparameter search                               |
| On-chain commit pointer             | Policy code lives in public repo; on-chain record is `{model, revision}` JSON. Harness fetches and pins exact commit.                                                                                                                                                                        | Direct `.py` file upload: considered, replaced — pointer is cleaner and more auditable             |
| PG19 for V1 prompts                 | Long-form text pushes KV cache to ~12% of peak memory at 32K tokens, where compression gains are measurable. Block-hash seed makes selection deterministic and verifiable. Passages truncated at ~131K chars (~32K tokens) with sentence/word boundary cleanup.                              | FineWeb: rejected — median docs <1K tokens, cache too small for signal. Synthetic: deferred to V2+ |
| Hard KL gate                        | Soft penalty creates incentives to trade output quality for efficiency in unacceptable ways. Hard reject above threshold is cleaner.                                                                                                                                                         | Soft KL penalty in score: rejected                                                                 |
| `write` + `attend` over three hooks | Three hooks force score tensor materialization and create circular dependency for eviction. `write + attend` lets the miner own the full pass.                                                                                                                                               | Separate score/aggregate hooks: rejected — kills TurboQuant-class fused approaches                 |
| Triton blocked in V1                | Triton kernels can read arbitrary GPU memory. PyTorch implements TurboQuant's algorithm correctly. Kernel optimization is V2.                                                                                                                                                                | Allow Triton in V1: rejected — sandboxing is harder, algorithmic correctness is sufficient for V1  |
| ~20% attention weight checks        | Catches degenerate policies that collapse attention to one token without triggering the KL gate.                                                                                                                                                                                             | Always check: overhead not worth it every round                                                    |

## Out of scope for V1

| Item                                                    | When                                  |
| ------------------------------------------------------- | ------------------------------------- |
| Decentralized validator execution / verifiable rollouts | V2 — after harness is stable          |
| Triton / custom kernel support                          | V2 — requires kernel-level sandboxing |
| Multi-model evaluation                                  | V2 — harness is architected for it    |
| Parallel submission evaluation                          | V2 — serial queue is fine for V1      |
| Frontend / leaderboard UI                               | After API is stable                   |
| Additional prompt datasets (V2+)                        | After V1 is running                   |
| Scheduling, batching, offloading optimizations          | V2+                                   |
