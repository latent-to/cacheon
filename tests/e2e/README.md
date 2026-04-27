# E2E Tests

Run these manually. They are **not collected by pytest** and **not run in CI**.

## Scripts

| Script       | Runs where | What it exercises                                                                                       |
| ------------ | ---------- | ------------------------------------------------------------------------------------------------------- |
| `e2e_pod.py` | GPU pod    | Calls `pod_eval.run_job()` in-process — tests baseline + scoring + per-challenger DQ handling           |
| `e2e_cpu.py` | CPU server | Full SSH/SFTP pipeline: fetches policies, uploads to GPU pod, runs pod_eval over SSH, downloads results |

## GPU Pod

```bash
export HF_TOKEN=hf_...

# Fast smoke-check (2 prompts — catches regressions quickly)
python tests/e2e/e2e_pod.py --device cuda --n-prompts 2

# Full run (3 prompts — matches default production sample size)
python tests/e2e/e2e_pod.py --device cuda --n-prompts 3

# JSON output (pipe into jq for structured inspection)
python tests/e2e/e2e_pod.py --device cuda --n-prompts 2 --json | jq .

# Verbose logging (shows fetch, precheck, and per-prompt timing)
python tests/e2e/e2e_pod.py --device cuda --n-prompts 2 -v
```

### CLI flags

| Flag               | Default | Description                                                           |
| ------------------ | ------- | --------------------------------------------------------------------- |
| `--device`         | `cuda`  | PyTorch device (`cuda` on pod, `cpu` for quick offline sanity checks) |
| `--n-prompts`      | `3`     | Number of PG19-seeded prompts to run per challenger                   |
| `--max-new-tokens` | `256`   | Max generation tokens per prompt (lower = faster run)                 |
| `--policies`       | auto    | Path to `fixtures/example_policies.json`                              |
| `--json`           | off     | Emit NDJSON result instead of human-readable table                    |
| `-v / --verbose`   | off     | Debug-level logging for fetch, precheck, and harness                  |

## CPU-side

```bash
export HF_TOKEN=hf_...

# Smoke-check the full CPU→GPU pipeline
python tests/e2e/e2e_cpu.py \
    --gpu-pod-ssh-host ssh.deployments.targon.com \
    --gpu-pod-ssh-user wrk-b6ptrqbmfkoj \
    --n-prompts 2

# Full run with verbose logging
python tests/e2e/e2e_cpu.py \
    --gpu-pod-ssh-host ssh.deployments.targon.com \
    --gpu-pod-ssh-user wrk-b6ptrqbmfkoj \
    --n-prompts 3 -v

# JSON output
python tests/e2e/e2e_cpu.py \
    --gpu-pod-ssh-host ssh.deployments.targon.com \
    --gpu-pod-ssh-user wrk-b6ptrqbmfkoj \
    --n-prompts 2 --json | jq .
```

### `e2e_cpu.py` CLI flags

| Flag                 | Default              | Description                              |
| -------------------- | -------------------- | ---------------------------------------- |
| `--gpu-pod-ssh-host` | _(required)_         | SSH hostname of the GPU pod              |
| `--gpu-pod-ssh-user` | _(required)_         | SSH username on the GPU pod              |
| `--gpu-pod-ssh-port` | `22`                 | SSH port                                 |
| `--gpu-pod-work-dir` | `/workspace/cacheon` | Repo checkout path on the pod            |
| `--device`           | `cuda`               | PyTorch device on the pod                |
| `--dtype`            | `float16`            | Model dtype on the pod                   |
| `--n-prompts`        | `3`                  | Number of prompts per challenger         |
| `--max-new-tokens`   | `256`                | Max tokens per prompt                    |
| `--timeout`          | `1200`               | SSH exec timeout (seconds)               |
| `--policies`         | auto                 | Path to `fixtures/example_policies.json` |
| `--json`             | off                  | Emit NDJSON result instead of table      |
| `-v / --verbose`     | off                  | Debug-level logging                      |

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
