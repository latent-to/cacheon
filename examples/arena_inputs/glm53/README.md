# GLM-5.3 development check inputs

Use these inputs with the commissioned GLM-5.3 NVFP4 model and SGLang 0.5.20
image on four B300 GPUs (TP4 and attention DP4). Keep this folder outside
the submitted bundle. Mount it read-only at `/arena`, the model at `/model`,
and a writable result/cache directory at `/work`.

Native MTP uses the checkpoint's nextn layer with `EAGLE`, three speculative
steps, top-k one and four draft tokens. No separate draft model is required.
Prefix caching and the native hierarchical cache are enabled, matching the
agent-replay runtime. The host tier allocates 120 GB per DP rank, or 480 GB per
TP4 lane, in addition to model-loading memory. The development config caps
running requests at 24 per lane to bound graph setup. Qualification owns its
sealed workload and configuration; this check does not produce a speed score.

The image applies a bounds fix to SGLang 0.5.20's `memcpy_triton`: EAGLE draft
gather/scatter counts can exceed the local tensor rows. The copy is clamped to
both tensors, matching `memcpy_cpu`. Remove the patch when the pinned upstream
image includes this fix.

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

A `tree_cache` bundle uses `target = "prefix_cache"` with the same `mode` and
arena. The runtime factory is described in
[the cache contract](../../../docs/architecture/slot-contract.md#the-prefix-cache).
Development support does not advertise that target on a live arena; use the
arena's published target availability before submitting.

```bash
python -m cacheon.cli check /bundles/my_bundle \
  --model /model --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json --output /work/check-001
```

The first batch contains eight public synthetic prompts, each tokenized to
exactly 8,192 tokens with this GLM tokenizer. A second batch reuses 512-token
prefixes of those prompts, twice each. Both generate eight deterministic output
tokens with EOS stopping disabled. The 24 requests provide six cache completions
per DP rank under the default round-robin routing, above the four-call audit
minimum. A single eight-request batch only supplies two cache completions per
rank and cannot qualify the cache audit. These inputs exercise prefill, decode
and prefix reuse while keeping the untimed, per-call audit bounded.
The text describes summing a list of integers and dividing by its length; each
prompt has a distinct numbered prefix. No hidden quality questions are included.

Use the default four-window minimum and 1,800-second limit per engine. The
checker runs every supplied request, then starts a fresh graph engine only
after the audit passes. A development PASS covers these inputs; the arena
still qualifies candidates on its complete served workload and quality set.
Larger batches, long context, and uneven request lengths remain necessary
when those regimes are part of the candidate's intended optimization.
