# Cacheon — Inference Engine

The GPU-side evaluation harness for **Cacheon** (SN14). Loads Qwen2.5-7B-Instruct, monkey-patches attention layers to route K/V through a `KVCachePolicy`, runs prefill + decode, and returns output text, logits, latency, and peak GPU memory.

This code lives on the GPU pod.

## Files

| File               | What it does                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| `policy.py`        | `KVCachePolicy` interface — the contract every miner submission implements                      |
| `passthrough.py`   | Baseline policy: uncompressed FP16 cache, standard attention. The control variable.             |
| `harness.py`       | Loads the model, monkey-patches attention, runs the generate loop, collects metrics             |
| `setup.sh`         | Provisions a fresh GPU instance: system deps, repo clone/pull, venv, model download, smoke test |
| `requirements.txt` | Python deps with version constraints explained                                                  |

## Setup (new GPU instance)

```bash
export GITHUB_PAT=your_github_pat_here
export HF_TOKEN=your_hf_token_here

bash -c "$(curl -fsSL -H "Authorization: token $GITHUB_PAT" \
  https://raw.githubusercontent.com/latent-to/cacheon/main/inference_engine/setup.sh)"
```

This clones the repo to `/root/cacheon`, creates a venv at `/root/venv`, installs deps, downloads model weights to `/root/.cache/huggingface` (~16GB), and runs the smoke test. Everything under `/root` persists across pod restarts on Lium.

On subsequent SSHs:

```bash
source /root/venv/bin/activate
cd /root/cacheon
```

## Running

Run these in order. Each layer catches different bugs.

```bash
# 1. Unit tests — no GPU, no model download (~4s)
#    Tests PassthroughPolicy math with hand-crafted tensors: write/attend shapes,
#    GQA repeat, causal mask, memory counting, cache reset.
pytest tests/test_inference_engine.py tests/test_harness_arch.py -v

# 2. Integration tests — 0.5B model on CUDA (~13s)
#    Tests the full end-to-end monkey-patch path with a real model.
#    Confirms write()/attend() are called for every layer, passthrough matches
#    unpatched HF output, RunResult has correct shapes.
pytest tests/test_harness_integration.py -v -m integration

# 3. Smoke test — 0.5B model, also works on MPS/CPU (~varies)
#    Human-readable output: monkey-patch sanity, verify, baseline metrics.
python scripts/smoke_test.py

# 4. Phase 1 gate — 7B model on CUDA (~25 min for 5 prompts × 128 tokens)
#    THE stop condition. Runs harness.verify() on 5 prompts with the real 7B.
#    Prints baseline latency and peak GPU memory — record these for Phase 2.
python -m inference_engine
```

**Confirmed baseline (GeForce RTX 3090, Qwen2.5-7B-Instruct, FP16, 5 prompts × 128 tokens):**

- Peak GPU: 15.49 GB
- Latency: 25.18s (includes slow reference generation — expected)
