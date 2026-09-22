# GLM-5.3 development check inputs

Use these inputs with the commissioned GLM-5.3 NVFP4 model and SGLang 0.5.20
image on four B300 GPUs (TP4 and attention DP4). Keep this folder outside
the submitted bundle. Mount it read-only at `/arena`, the model at `/model`,
and a writable result/cache directory at `/work`.

Native MTP uses the checkpoint's nextn layer with `EAGLE`, one speculative
step, top-k one and two draft tokens. No separate draft model is required.
The full arena retains concurrency 128 at 8,192 input/1,024 output tokens and
concurrency 24 at 65,536 input/4,096 output tokens. Qualification of this
configuration is pending; historical non-MTP results do not establish its
throughput or the champion's benefit over stock.

Build the pinned image from the repository root with the selected revision:

```bash
build_dir=$(mktemp -d)
mkdir "$build_dir/src"
git archive HEAD | tar -x -C "$build_dir/src"
docker build -t cacheon-glm53-mtp \
  -f examples/arena_inputs/glm53/Dockerfile "$build_dir"
```

Inside that image, run `python -m cacheon.cli compat --sglang-version 0.5.20`
before the development check.

Route the bundle to this arena in its existing competition table:

```toml
[competition]
target = "forward_pass"
mode = "slot"
arena = "glm53-b300-node-v1"
```

```bash
python -m cacheon.cli check /bundles/my_bundle \
  --model /model --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json --output /work/check-001
```

The set contains eight public synthetic prompts, each tokenized to exactly
8,192 tokens with this GLM tokenizer, and eight deterministic output tokens
with EOS stopping disabled. It exercises a substantial prefill and decode on
every rank while keeping the untimed, per-call development audit bounded.
The text describes summing a list of integers and dividing by its length; each
prompt has a distinct numbered prefix. No hidden quality questions are included.

Use the default four-window minimum and 1,800-second limit per engine. The
checker runs every supplied request, then starts a fresh graph engine only
after the audit passes. A development PASS covers these inputs; the arena
still qualifies candidates on its complete served workload and quality set.
Larger batches, long context, and uneven request lengths remain necessary
when those regimes are part of the candidate's intended optimization.
