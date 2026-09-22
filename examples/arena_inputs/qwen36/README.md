# Qwen3.6-35B-A3B development inputs

Arena: `qwen36-35b-h100-bf16-tp1`. Model: Qwen3.6-35B-A3B in BF16. Develop
against the official
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) weights,
SGLang 0.5.20 and one H100 80 GB per engine (TP1). The engine configuration
retains FP8 KV, FP32 recurrent state, the Triton MoE runner and CUDA graphs.
The Gated DeltaNet decode backend stays at the engine default; `cutedsl`
generated incorrect answers in the retained commissioning tests.

Native MTP is enabled with `EAGLE`, three speculative steps, top-k one and
four draft tokens, using the MTP weights in the target checkpoint. No separate
draft download is required. `enable_linear_replayssm_spec` uses the upstream
ReplaySSM verifier to avoid per-draft state snapshots, which exceed H100 memory
at 48 requests with this BF16 model. FP32 recurrent state is retained.
Target-layer bundles retain the same node interface
and must handle `TARGET_VERIFY` as well as prefill and ordinary decode. The
draft worker is outside that contribution boundary. This runtime change needs
fresh qualification on the production workloads; the older timings below are
non-MTP evidence, not a performance claim for this configuration.

The prompts are excerpts of agentic coding sessions from
[`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
(CC-BY-4.0), formatted with the model's chat template and cut to exact token
counts.

These are development inputs, not the sealed qualification workload. Keep
this directory outside the submitted bundle. Mount it read-only at `/arena`,
the model read-only at `/model`, the bundle at `/bundles/my_bundle`, and a
writable result/cache directory at `/work` inside the arena image.

```bash
python -m cacheon.cli scan /bundles/my_bundle
python -m cacheon.cli check /bundles/my_bundle \
  --model /model --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json \
  --output /work/check-short-001 --timeout-seconds 1800
```

The short set takes eight unchanged 8,192-token prompts from the retained
public development set and generates eight tokens per request with EOS
stopping disabled. After it passes, check the separate long-context set:

```bash
python -m cacheon.cli check /bundles/my_bundle \
  --model /model --engine-config /arena/engine-config.json \
  --requests /arena/long-context-requests.json \
  --output /work/check-long-001 --timeout-seconds 1800
```

The long set contains one unchanged 65,536-token public prompt and eight
output tokens. The two sets retain the original prompt text and sampling
settings except for the explicit output-token limit. A smaller batch alone
did not resolve the old whole-layer timeout: the original sets generated
1,024 and 4,096 tokens, with the eager checker auditing each decode step.
There is no hidden truncation in `cacheon check`.

Keep the default minimum of four audited windows. The checker audits the
entire supplied set, then starts a fresh engine to check graph capture only
after the audit passes. A validated gross mismatch stops early. Insufficient
coverage remains `NO_DECISION`; a timeout is incomplete evidence. Use a new
output directory for each check. The limit is per engine, so audit and graph
execution together can take up to an hour per command, plus bounded teardown.

On the commissioned H100 image, an honest whole-layer claim (`model.layers.*`,
40 bound layers, 320 audited calls) completed the short set in about 250
seconds and the long set in about 240 seconds, audit and graph capture
included. A layer that returns wrong values stopped with exit 2 in about 120
seconds. The default 1,800-second limit per engine is the arena's published
development limit.

These sets do not establish full-batch coverage, speed or qualification.
Run full-width and longer generation checks when the optimization depends on
those regimes. Whole-layer claims (`model.layers.*`) had retained honest
audit evidence; a whole-model claim (`model`) failed honest comparison at
64k context and is not certified by this recipe. See the
[measured limitations](../../../docs/results/qwen-h100-node-slots.md).

For arena routing, add this to the bundle's existing competition table:

```toml
[competition]
target = "forward_pass"
mode = "slot"
arena = "qwen36-35b-h100-bf16-tp1"
```

The evaluation fee is 0.2 TAO: pass `--eval-cost-tao-rao 200000000` to
`chain-eval-cost` and `chain-submit --pay` (the CLI default transfers 0.5 TAO).

## Building the image

The Dockerfile and H100 MoE table reproduce the retained image recipe. Build
from a committed source checkout; the base image is pinned by digest. From
the repository root, prepare a fresh context and build:

```bash
context=$(mktemp -d)
mkdir "$context/src"
git archive HEAD | tar -x -C "$context/src"
cp examples/arena_inputs/qwen36/Dockerfile "$context/Dockerfile"
cp -r examples/arena_inputs/qwen36/moecfg "$context/moecfg"
docker build --network=none \
  --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" \
  -t cacheon-qwen-development "$context"
```

Pull the pinned base image before the offline build. The recipe preserves the
system-Python package layout and `cacheon.pth` required by the production
launcher, and sets `SGLANG_LOGPROB_CHUNK_SIZE=256` to avoid the recorded
teacher-logits allocation failure. No checkpoint, private prompts, arena seed,
credentials or validator registration belong in the image context.
