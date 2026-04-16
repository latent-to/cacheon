<div align="center">

# **Cacheon (SN14) — Inference Optimization**

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Discord](https://discord.gg/bittensor) • [TAO.app](https://tao.app/subnets/14)

---

</div>

- [What This Subnet Is](#what-this-subnet-is)
- [Why It Matters](#why-it-matters)
- [What is the KV cache? How does it work?](#what-is-the-kv-cache-how-does-it-work)
- [Competition Model](#competition-model)
- [Miner Submission](#miner-submission)
- [Evaluation](#evaluation)
- [The Interface](#the-interface)
- [Sandbox and Security](#sandbox-and-security)
- [Roadmap](#roadmap)
- [License](#license)

## What This Subnet Is

When you use an LLM, it has to remember everything it has already seen in the conversation to pick the next word. That memory lives in a structure called the **KV cache**. Longer context means more of that memory — and on a GPU, memory is often the limiting factor for speed and cost. **Cacheon** (Bittensor Subnet 14) is where participants compete to run a fixed model **cheaper and faster** while keeping its answers the same (within measured quality bounds).

**What the subnet does:** miners submit code that controls how keys and values are stored and how attention is computed over them; validators score those policies on memory, latency, and output fidelity. The best policy wins emission (see [Competition Model](#competition-model)).

Cacheon is about **inference** (running a fixed, already-trained model), not training. We treat inference optimization as a search problem the subnet measures in the open.

The starting point is **KV cache** — the single largest memory bottleneck in transformer inference. V1 focuses exclusively on KV cache optimization. Later versions expand the optimization surface to cover scheduling, batching, and cross-hardware policy generation.

**TL;DR:** Cacheon (SN14) optimizes **inference-time** KV-cache behavior (V1: KV path only) on a **canonical model** (fixed weights). Miners implement **`KVCachePolicy`** (`write` / `attend`) to customize KV storage and attention; the harness scores **memory + latency** vs. passthrough with a **KL** quality floor.

## Why It Matters

Training is no longer the only bottleneck. In production, the real constraints are:

- **Latency** — tokens/sec, time-to-first-token
- **Memory** — KV cache scaling with context length
- **Cost** — GPU utilization and serving efficiency

Every transformer model stores a key-value cache that grows linearly with sequence length. At 100k tokens across 80 layers with `head_dim=128`, the KV cache alone consumes ~80GB in FP16. This is the wall.

Current work on KV cache optimization is fragmented — H2O, SnapKV, KIVI, TurboQuant, StreamingLLM are all tested differently, benchmarks aren't consistent, and results are hard to compare. Cacheon is trying to fix that.

## What is the KV cache? How does it work?

Models generate text one token at a time. For each new token, they still need access to everything that came before in that sequence. Recomputing the full past from scratch every step would be impossibly slow, so the model **caches** part of its work: for each layer, and for each position already processed, it stores two vectors per attention head — the **key** and **value** — that it will reuse when scoring attention for new tokens. That growing store is the **KV cache**. More tokens in context means more cached rows, which is why memory use shoots up on long prompts and long chats.

The following block is the same idea in the notation the implementation uses.

When a transformer generates token N, every layer computes:

```
Q = input @ W_q       # query for this token
K = input @ W_k       # key for this token
V = input @ W_v       # value for this token

scores = Q @ K_cache.T        # dot product with every cached key
weights = softmax(scores)
output = weights @ V_cache     # weighted sum of cached values
```

The KV cache is `K_cache` and `V_cache` — one row per previous token, per layer, per attention head. It grows by one row every decode step. There are two places to intervene:

1. **Storage** — compress K/V when writing to cache. Decompress when scoring. This covers quantization (INT8, FP8, 3-bit Lloyd-Max) and eviction (drop low-importance tokens).
2. **Scoring** — compute `Q @ K_compressed.T` directly against the compressed representation, with a correction term. Never decompress. This is what approaches like TurboQuant do — fused scoring against quantized keys with QJL bias correction inline.

Path 2 is strictly more powerful but requires the miner to control the attention computation, not just the storage format. Cacheon's interface supports both.

## Competition Model

Cacheon uses a **king-of-the-hill** structure. There is one reigning champion — the miner with the best score on record. All emission goes to the king. Challengers compete to dethrone the king.

A **challenger** is any hotkey with a commitment that hasn't been evaluated yet. The validator scans on-chain commitments roughly every 360 seconds. If no new commitments exist since the last round, the king keeps their weight and no inference runs. GPU eval is only triggered by new entries.

A challenger must beat the king's score to take the crown. The king is dethroned only when a strictly better policy is found. There's no partial credit — winner takes all emission.

## Miner Submission

Miners host their `KVCachePolicy` implementation as a public repository on HuggingFace or GitHub, then register it on-chain with a commit pointer:

```python
import json
commit_data = json.dumps({"model": "hf-username/my-kv-policy", "revision": "abc123sha"})
subtensor.set_reveal_commitment(wallet=wallet, netuid=netuid, data=commit_data, blocks_until_reveal=1)
```

**One submission per hotkey.** To submit a new policy, register a new hotkey. This creates commitment pressure — test locally before committing. Each hotkey maps to exactly one policy version, keeping the on-chain record clean.

The harness fetches the policy at the pinned revision and runs it inside a sandbox. The revision pin ensures the harness always evaluates the exact code that was committed.

## Evaluation

### Prompts

V1 uses **PG19** (Project Gutenberg books) as the prompt source. The block hash seeds a random selection of passages from the dataset — deterministic, reproducible, and tied to on-chain state. Anyone can verify which passages were used in a given round by replaying the seed derivation.

### Scoring

For each evaluation:

1. Run baseline inference (passthrough policy) — record logits, latency, peak memory
2. Run the miner's policy — same prompts, same setup, policy applied at the KV layer
3. Compare across three axes:
   - **Quality**: KL divergence between baseline and policy logits. Submissions exceeding the threshold are rejected.
   - **Memory**: peak GPU memory reduction relative to baseline, measured by the harness
   - **Latency**: time-to-first-token and tokens/sec relative to baseline

```python
if kl_divergence > 0.1:   # nats — tunable
    score = 0.0
else:
    score = (0.6 * memory_reduction) + (0.4 * latency_improvement)
```

### Harness

The harness is centralized in V1. One GPU pod runs all evaluations. A separate CPU server holds wallet keys, reads the chain, selects challengers, and sets weights. The GPU pod has no chain access and no wallet keys — it receives prompts and policy pointers over SSH, runs inference, and returns a results JSON.

This split is a security boundary: a compromised GPU pod cannot steal wallet keys or manipulate weight-setting. Centralized eval also ensures reproducibility — different hardware produces different KL values and would produce conflicting rankings.

The validator loop runs roughly every 360 seconds. It compares revealed on-chain commitments against known state to find new challengers. If none exist, the current king keeps its weight and no inference runs — GPU time is only spent when there's something new to evaluate.

## The Interface

Miners submit a Python class implementing the `KVCachePolicy` interface. The miner owns the cache — how K/V entries are stored and how attention is computed against them. See `inference_engine/policy.py` for the full definition and `inference_engine/passthrough.py` for a reference implementation.

```python
class KVCachePolicy:
    def setup(self, config: CacheConfig) -> None:
        """Called once per sequence. Initialize internal state."""

    def write(self, keys, values, layer_idx, positions) -> None:
        """Store K/V entries. Compress, quantize, evict — miner's choice."""

    def attend(self, query, layer_idx) -> AttentionOutput:
        """Full attention in one call: scoring + softmax + aggregation."""

    def memory_bytes(self) -> int:
        """Report current memory usage of all stored cache state."""
```

**Why `write` + `attend` and not three separate hooks:**

- `attend` needs to own scoring because eviction strategies (H2O) need attention weights to track token importance — those weights are computed during scoring, not during write. Splitting them creates a circular dependency.
- Fused scoring (TurboQuant) computes `Q @ K_compressed.T + correction` in one call without ever decompressing. Separate hooks force score tensor materialization and kill that approach at long context.
- Prefill and decode are handled identically: `write` receives `seq_len = prompt_length` during prefill and `seq_len = 1` during decode. No special-casing needed.

### Example policies

**TurboQuant (3-bit quantization with fused scoring)**

- `setup`: allocate random rotation matrix, initialize QJL sign storage, set up Lloyd-Max centroid tables
- `write`: random-rotate incoming K/V, Lloyd-Max quantize to 3-bit, store centroids + QJL signs
- `attend`: `Q @ K_mse.T + QJL_correction → softmax → aggregate from centroids` — no decompression

**H2O (heavy-hitter eviction)**

- `setup`: allocate importance score accumulator, set eviction budget
- `write`: store K/V in FP16; evict tokens with lowest accumulated importance when over budget
- `attend`: standard attention; update importance accumulator with this step's attention weights

**Basic INT8 quantization**

- `write`: INT8 quantize K/V, store scale factors
- `attend`: dequantize, standard attention

### Allowed / Not Allowed

**Allowed:**

- KV quantization (asymmetric bit allocation, Lloyd-Max, TurboQuant-style rotation + QJL correction)
- KV eviction and pruning strategies (H2O, StreamingLLM, scissorhands, attention-score-driven eviction)
- KV layout and storage strategies
- Fused scoring against compressed representations
- Per-layer and per-head adaptive policies

**Not allowed:**

- Model weight changes or finetuning
- Attention mechanism redesign
- Custom CUDA kernels or Triton (V1 restriction — PyTorch ops only)
- Batching-level or scheduling-level changes

## Sandbox and Security

Policy code is checked in two layers before full harness eval:

1. **Static analysis** (`inference_engine/sandbox.py`) — parse with `ast`, no execution. Import allowlist includes `torch`, `numpy`, `math`, `einops`, and `inference_engine` (only the package root and `inference_engine.policy`). Other `inference_engine` submodules are blocked so internal modules cannot re-export stdlib objects.

2. **Subprocess isolation** (`inference_engine/runner.py`) — the policy runs in a child process with a hard timeout and output validation. On hosts where **firejail** is installed (the intended production setup is the **CPU validator** in Phase 5), the runner uses it: no network (`--net=none`), isolated filesystem under the job temp dir, memory and process limits. The GPU pod image does not install firejail via `inference_engine/setup.sh` — it runs harness + scoring only. If `firejail` is not on `PATH` (macOS, CI, or GPU-only workflows), the runner falls back to a plain subprocess and logs a warning — fine for local dev, not the security posture for untrusted miner code in production.

The harness validates at each boundary: output shape/dtype/value checks, `memory_bytes()` cross-checked against `torch.cuda.max_memory_allocated()` (which the miner cannot fake), and wall-clock latency measurement of `write` + `attend`.

## Roadmap

### V1 — KV Cache Optimization

Single well-defined problem: optimize KV cache under fixed model, hardware, and runtime. King-of-the-hill competition. Winner-take-all emission. Centralized harness.

### V2 — Expanded Optimization Surface

- Triton kernel support with additional sandboxing
- Hybrid KV strategies (quantization + eviction combined)
- Cache offloading to CPU/NVMe
- Adaptive memory policies responding to context length at runtime
- Multi-turn, long-context reasoning, and RAG-style workloads
- Multiple hardware targets
- Verifiable eval rollouts for decentralized auditing

### V3 — Policy Generation

Given a model, hardware, workload distribution, and quality constraints — output the best inference policy for that configuration.

### V4 — Deployment

One-click config generation with direct integration into vLLM, Hugging Face, and TensorRT-LLM. Cacheon stops evaluating ideas and starts shipping them.

## End State

Cacheon becomes a system that continuously learns how to run models optimally across environments and hardware targets, and produces the inference configurations that actually get deployed.
