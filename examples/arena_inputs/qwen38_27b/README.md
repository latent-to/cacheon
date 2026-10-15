# Qwen3.8-27B FP8 development inputs

Use `Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` with SGLang 0.5.20 on one
96 GB NVIDIA RTX PRO 6000 Blackwell GPU (TP1). The context limit is 262,144
tokens. Qualification compares two independent TP1 engines on separate GPUs
and swaps their physical lanes. A single GPU is sufficient for development.

The arena selector is `qwen38-27b-rtxpro6000-fp8-tp1`. The upload fee
is 0.2 TAO (200,000,000 RAO), with a 10% allocation of arena rewards.
Check the [live arena dashboard](https://dash.cacheon.ai/qwen3.8-27b)
for availability before paying for a submission.

Build the development image from the commissioned runtime source:

```bash
build_dir=$(mktemp -d)
mkdir "$build_dir/src"
git archive 8730fb42a2ff145e377ea98c3a6093a82e4c2824 | tar -x -C "$build_dir/src"
docker build -t cacheon-qwen38-27b \
  -f examples/arena_inputs/qwen38_27b/Dockerfile "$build_dir"
```

The recipe pins the dependencies and runtime source; a local rebuild is not
the operator's registered image digest. Inside the image, run
`python -m cacheon.cli compat --sglang-version 0.5.20`.

Keep this directory outside the submitted bundle. Mount it read-only at
`/arena`, the model at `/model`, the candidate at `/bundles/my_bundle`, and a
writable cache/result directory at `/work`. Set the bundle's competition:

```toml
[competition]
target = "forward_pass"
mode = "slot"
arena = "qwen38-27b-rtxpro6000-fp8-tp1"
```

The faithful control uses the `model` node and calls `module.forward` with
the original arguments. Other module addresses need their own execution and
audit coverage. This arena currently registers only `forward_pass`.

```bash
python -m cacheon.cli check /bundles/my_bundle \
  --model /model --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json --output /work/check-001
```

The four sequential batches each contain two synthetic prompts, generating
32 tokens with zero temperature and EOS stopping disabled. Batches two and
four repeat their preceding prompts and cache salts to exercise prefix reuse.
The config keeps radix caching enabled and captures decode CUDA graphs at
batch sizes 1, 2, 4 and 8. It caps running requests at eight; production
qualification owns its sealed concurrency and replay workload.

Native MTP uses EAGLE with three speculative steps, top-k one and four draft
tokens. The checkpoint supplies the draft head; no separate draft model is
needed. Attention KV uses `fp8_e4m3`; recurrent state remains FP32. Submitted
forward nodes must support target-verification calls as well as ordinary
prefill/decode. The draft model retains stock execution.

Qwen's hybrid cache includes recurrent state as well as attention KV. Node
audits compare that state against stock. Candidate and baseline engines have
separate caches, and evaluation boundaries reset them; reuse within a replay
does not carry candidate state into the next evaluation. The development
check exercises eager audits and graph execution but does not establish a
speed score or full 256k workload coverage.

For a public submission, the existing `chain-submit` fee option is
`--eval-cost-tao-rao 200000000`. Use it only after this arena is published as
open; an arena selector does not itself authorize admission.
