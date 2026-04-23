# E2E Tests

Run these manually on a GPU pod. They are **not collected by pytest** and **not run in CI**.

## `e2e_eval.py`

Run it on GPU:

```bash
export HF_TOKEN=hf_...
python tests/e2e/e2e_eval.py --device cuda --n-prompts 3
```

### What it tests

- **Policy fetch** — downloads `policy.py` from HuggingFace via the real `validator.policy_fetch` path
- **Precheck / sandbox** — runs AST allowlist validation via the real `validator.precheck` path
- **Harness** — loads Qwen 7B, runs the policy against the PassthroughPolicy baseline
- **Scoring** — computes KL divergence, KV-cache memory reduction (`policy.memory_bytes()`), latency improvement

Prerequisites:

- NVIDIA GPU with enough VRAM for Qwen 7B FP16 + KV cache (~24 GB minimum, 80 GB recommended)
- Run `tests/e2e/e2e_seed_hf.py` first to generate `fixtures/example_policies.json`
