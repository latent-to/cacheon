# Cacheon — Inference Engine

The GPU-side evaluation harness for **Cacheon** (SN14). Loads Qwen2.5-7B-Instruct, monkey-patches attention layers to route K/V through a `KVCachePolicy`, runs prefill + decode, and returns output text, logits, latency, and peak GPU memory.

This code lives on the GPU pod.

## Files

| File               | What it does                                                                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `policy.py`        | `KVCachePolicy` interface — the contract every miner submission implements                                                                                                                                   |
| `passthrough.py`   | Baseline policy: uncompressed FP16 cache, standard attention. The control variable.                                                                                                                          |
| `harness.py`       | Loads the model, monkey-patches attention, runs the generate loop, collects metrics                                                                                                                          |
| `scoring.py`       | Takes two `RunResult`s (baseline + miner) → `ScoreResult` (KL gate + weighted score)                                                                                                                         |
| `prompts.py`       | Deterministic PG19 prompt sampling seeded by block hash → `list[str]`. Passages truncated to ~32K tokens, filtered for min length, dataset revision pinned.                                                  |
| `sandbox.py`       | Static AST checks on submitted policy source (imports, blocked calls, structure)                                                                                                                             |
| `runner.py`        | Subprocess sandbox precheck (`run_check`): timeout + output validation; wraps with **firejail** when `firejail` is on `PATH` (intended on the **CPU validator** in Phase 5; not installed by GPU `setup.sh`) |
| `__main__.py`      | `python -m inference_engine` — smoke test and baseline metrics                                                                                                                                               |
| `setup.sh`         | Provisions a fresh **GPU** instance: git, curl, tmux, `python3-venv`, repo, venv, model weights, smoke test (no `firejail` — CPU server installs that for `runner`)                                          |
| `requirements.txt` | Python deps with version constraints explained                                                                                                                                                               |

## Setup (new GPU instance)

`setup.sh` installs **PyTorch from the CUDA 12.8 wheel index** (`download.pytorch.org/whl/cu128`) before `requirements.txt`, so the build matches typical cloud GPU drivers. If you install dependencies by hand, install a CUDA-12-compatible torch the same way or you may see a misleading “driver too old” error at import.

```bash
export GITHUB_PAT=your_github_pat_here
export HF_TOKEN=your_hf_token_here

bash -c "$(curl -fsSL -H "Authorization: token $GITHUB_PAT" \
  https://raw.githubusercontent.com/latent-to/cacheon/main/inference_engine/setup.sh)"
```

**Storage layout (Targon):** mount your persistent volume at **`/workspace`**. Repo, venv, and Hugging Face cache all live under that path.

| Path                            | Role                                |
| ------------------------------- | ----------------------------------- |
| `/workspace/cacheon`            | Git checkout                        |
| `/workspace/venv`               | Python venv                         |
| `/workspace/.cache/huggingface` | Model weights (Qwen2.5-7B-Instruct) |

On first run the script downloads weights into `HF_HOME` if missing; later runs skip when snapshots already exist.

On subsequent SSHs:

```bash
source /workspace/venv/bin/activate
cd /workspace/cacheon
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

# 4. Full harness run — 7B model on CUDA (~25 min for 5 prompts × 128 tokens)
#    Runs harness.verify() on 5 prompts with the real 7B (Phase 1 stop condition ✅).
#    Then runs PassthroughPolicy baseline and prints latency + peak GPU memory.
python -m inference_engine
```

**Confirmed baseline (GeForce RTX 3090, Qwen2.5-7B-Instruct, FP16, 5 prompts × 128 tokens):**

- Peak GPU: 15.49 GB
- Latency: 25.18s (includes slow reference generation — expected)
