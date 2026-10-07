# Your first component bundle

Start with a node control, then replace its implementation. The CPU command
checks packaging and callable interfaces. The engine command uses the real
model, binder and audit in the published arena image. A prefix-cache bundle
follows the same steps; see [Prefix-cache bundles](#prefix-cache-bundles).

## 1. Install a development checkout

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cpu,dev]"
```

For GPU checks use the arena's published image and local model. Keep its CUDA,
SGLang and kernel-library versions; a different installation tests a different
environment. Run candidate code only in the disposable development container.

## 2. Copy the CPU example

```bash
cp -R examples/miner_node_identity my_bundle
```

The [identity control](https://github.com/latent-to/cacheon/tree/main/examples/miner_node_identity)
claims `model.layers.*.mlp` and returns the original module's answer:

```python
def forward(module, *args, **kwargs):
    return module.forward(*args, **kwargs)
```

Choose an address supported by your model and arena. Change `bundle_id` and set
`competition.arena` to the published arena ID before submission. The optional
`prepare(module)` returns state passed as the entry's first argument.

## 3. Scan the source tree

```bash
python -m cacheon.cli scan my_bundle
```

Scanning covers declared sources and the rest of the bundle. Remove generated
artifacts from the submission directory. Writable compiler caches and ordinary
JIT work remain available inside the development container.

## 4. Verify the callable contract

```bash
python -m cacheon.cli verify my_bundle
```

For node bundles this scans and imports the entries in a fresh child, resolving
local helpers from the bundle root. It checks that entry accepts its first
positional argument and optional prepare accepts one module. It does not run
arbitrary preparation or forward calls without a model. `INTERFACE OK` establishes
no numerical correctness. CUDA dependencies need the published image for this
smoke test too.

## 5. Add a real specialization

Implement the computation inside the selected stock method interface. Keep its
arguments, return structure and state effects. Do not modify batching or
engine configuration. See [Kernel ABI](kernel-abi.md) and
[Finding a win](finding-a-win.md).

## 6. Move to the matching GPU environment

Inside the published image, mount the model and public arena inputs read-only,
with writable cache and result directories. Expose the full arena GPU topology.
The [GLM development inputs](https://github.com/latent-to/cacheon/tree/main/examples/arena_inputs/glm53)
provide two batches, 24 requests in all, that exercise prefill, decode and
prefix reuse on its published B300 configuration.
The [Qwen3.6-35B-A3B inputs](https://github.com/latent-to/cacheon/tree/main/examples/arena_inputs/qwen36)
provide a short and a long-context set, the H100 image recipe and the
`[competition]` lines for arena `qwen36-35b-h100-bf16-tp1`.
Use the input set for your arena; its model and topology must match. Then run:

```bash
python -m cacheon.cli check /bundles/my_bundle \
  --model /model \
  --engine-config /arena/engine-config.json \
  --requests /arena/development-requests.json \
  --output /work/check-001
```

`engine-config.json` is the published SGLang option object. The model path comes
from `--model`; a conflicting `model_path` fails. Each item in the requests file
is a keyword-argument object for `Engine.generate`, for example:

```json
[
  {
    "prompt": ["Explain why this loop terminates: for i in range(5): print(i)"],
    "sampling_params": {"temperature": 0, "max_new_tokens": 32, "ignore_eos": true}
  }
]
```

That short example explains the file format. Use the arena's real development
batches, lengths and widths for meaningful coverage; a short prompt is likely
to return `NO_DECISION`. Tokenized batches can use `input_ids` instead of `prompt`.

The command starts an eager audit engine first. Only after its receipts pass
coverage and numerical checks does it start a fresh graphs-on engine. It uses
the existing scheduler loader, node adapter and receipt gates. It changes graph
mode for the untimed audit role, not the published graph-pass configuration.
It does not generate a stock/candidate speed comparison or hidden-quality score.

The table contains one row per claimed node address and rank. A wildcard row
aggregates its bound modules, as the production receipt does. Audit windows,
violations, worst passing fraction and captured execution are shown; per-bound
module introspection belongs to the arena's supported-node information.

The output directory must be new and outside the bundle. It retains inputs,
`audit/engine.log`, `graph/engine.log`, stage results and the original receipt
files, including failures. A failed audit stops before the graph launch.
Default limits are 1,800 seconds per engine and four audit windows per
address/rank. Set `--timeout-seconds` and `--minimum-audit-windows` to the arena's
published development limits, not to values chosen after seeing a result.

Exit 0 means the development audit and capture checks passed, 2 means a failure
or execution error, and 3 means insufficient/incomplete audit evidence. The
[wrong control](https://github.com/latent-to/cacheon/tree/main/examples/miner_node_wrong)
has a valid interface but scales tensor-valued MLP outputs by 1.5 and should fail
the engine audit.

## 7. Decide whether the target is worth pursuing

A development check is not full-model quality, a speed win, qualification or
settlement. Measure the complete serving workload against the incumbent, then
follow [Submitting a bundle](submitting.md). Returning correct answers with no
speed improvement is a successful diagnostic, not a competitive contribution.

## Prefix-cache bundles

A cache bundle names `tree_cache` under the `prefix_cache` target and supplies a
factory that receives the runtime cache and returns a subclass of its type:

```toml
[competition]
target = "prefix_cache"
mode = "slot"
arena = "<published-arena-id>"

[[ops]]
slot = "tree_cache"
source = "cache/policy.py"
entry = "build"
```

```python
def build(cache):
    return type(cache)
```

Declare no `prepare`, dtypes, architectures or metadata. `verify` imports `build`
and checks that it accepts the runtime cache. Run `check` with the GLM development
inputs: their second batch reuses prefixes, so the cache audit sees enough
completions on every rank. A cache bundle cannot run on an arena whose engine
configuration disables radix caching. The table has one `tree_cache` row
per rank; the cache runs outside CUDA graphs, so it needs no captured execution.
The validator checks the bytes behind every prefix the cache serves; see
[Slots and targets](slots.md#the-prefix-cache).
