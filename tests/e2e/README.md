# E2E Tests

Run these manually on a GPU pod. They are **not collected by pytest** and **not run in CI**.

## Scripts

| Script       | Runs where | What it exercises                                                                             |
| ------------ | ---------- | --------------------------------------------------------------------------------------------- |
| `e2e_pod.py` | GPU pod    | Calls `pod_eval.run_job()` in-process — tests baseline + scoring + per-challenger DQ handling |

## Quick start (GPU pod)

```bash
export HF_TOKEN=hf_...

# GPU-side, in-process (fastest, tests the scoring path)
python tests/e2e/e2e_pod.py --device cuda --n-prompts 3
```

## What they test

- **Policy fetch** — downloads `policy.py` from HuggingFace via `validator.policy_fetch`
- **Precheck / sandbox** — AST allowlist validation via `validator.precheck`
- **Baseline** — PassthroughPolicy with SDPA (Flash Attention, no O(N²) matrix)
- **Challenger eval** — per-challenger try/except → DQ on OOM or crash (matches production `pod_eval.py`)
- **Scoring** — KL divergence, KV-cache memory reduction (harness-measured allocator delta), latency improvement

## Fixture policies

| Policy        | Attention     | Long-context safe | Expected outcome                                 |
| ------------- | ------------- | ----------------- | ------------------------------------------------ |
| `int8_sdpa`   | SDPA          | Yes               | Scores > 0 (passes KL gate, ~2x cache reduction) |
| `int8`        | Manual matmul | No (OOMs at 32K)  | DQ'd with "policy run failed: OOM"               |
| `naive_evict` | Manual matmul | No                | DQ'd (OOM or KL gate)                            |
| `passthrough` | SDPA          | Yes               | Scores ~0 (identity baseline, no compression)    |

## Prerequisites

- NVIDIA GPU with enough VRAM for Qwen 7B FP16 + KV cache (80 GB recommended)
- Run `tests/e2e/e2e_seed_hf.py` first to upload fixture policies to your HF namespace and generate `fixtures/example_policies.json`:
  ```bash
  export HF_TOKEN=hf_...
  python tests/e2e/e2e_seed_hf.py
  ```
