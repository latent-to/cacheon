# SGLang compatibility

Cacheon competes against and integrates with an exact SGLang runtime. The pin is
part of evaluation identity, not a loose minimum version. The
default compatibility pin is `0.5.20` in
[`cacheon/compat.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/compat.py). Different arenas may commission different exact runtimes:
the GLM arena runs `0.5.20` and the DeepSeek-V4.1-Flash arena `0.5.21`, whose
prefix cache hands a finished request through `insert_req` and `on_release`
instead of `cache_finished_req`; the cache seam guards both shapes.
Run `python -m cacheon.cli compat --sglang-version <pin>` inside the image
commissioned for that version. The runtime preflight independently compares the
installed version with its sealed `expected_sglang_version`. This does not
change an existing arena's authority or permit mixing measurements across versions.

Native MTP uses the existing `EngineSessionConfig.engine_kwargs`: set
`speculative_algorithm`, `speculative_num_steps`, `speculative_eagle_topk` and
`speculative_num_draft_tokens` in the validator-owned engine configuration.
These options are bound into its existing digest and forwarded to SGLang.
`EngineSessionConfig.engine_env` sets process environment for the engine and
its ranks the same way, bound into the same digest; only reviewed names cross
the boundary, currently the DeepSeek-V4.1 Engram host-table switches, which
SGLang reads from the environment rather than from its server arguments.
Target node addresses bind only to the target ModelRunner; the speculative
drafter retains stock execution. Miners keep the same node interface, including
the target-verification modes and shapes passed to it. Requalify the exact
model, runtime and workloads before activating this configuration; enabling
MTP does not carry forward a non-speculative speedup or change the scoring policy.

The node audit retains expected engine state in separate pinned host pieces and
streams comparisons on the GPU. It still restores and grades all selected state
rows; this storage choice does not change audit thresholds.

A green static seam canary establishes import and chokepoint compatibility only. A
runtime pin is eligible for evaluation authority only after end-to-end GPU
controls reject a deliberately broken bundle, accept a faithful bundle, and rebaseline
the registered champions under the exact new identity. Treat the source pin as the
compatibility target, not as evidence that these empirical gates passed.

## Why the pin matters

SGLang internals define the lower side of Cacheon's narrow waist: module paths,
call signatures, graph capture, attention/MoE dispatch, collectives, engine
arguments, and native dependencies. An upstream change can alter both whether
a contribution loads and what baseline it must beat.

Therefore a pin bump changes evaluation context. Existing performance evidence
cannot simply be relabeled with the new version.

### Validator consensus and the miner boundary

Validators independently score the same revealed contributions and publish weight
vectors. Bittensor's Yuma consensus compares those vectors with the stake-weighted
validator result. If validators run different SGLang revisions, the same contribution
can encounter different kernels, shapes, graph paths, throughput, or quality behavior;
the resulting weight divergence is a protocol fault rather than ordinary measurement
noise. Every validator for one arena must therefore use the same authenticated runtime
and evaluation identity and coordinate a pin transition.

That requirement binds authoritative measurement, not a miner's workstation. Miners may
develop with another SGLang revision against the node and cache contracts; `check` in the
published arena image is the first check at the pinned runtime, and qualification always
re-runs the submitted delta in the validator's pinned arena.
A pin bump remeasures existing contributions against a changed baseline and execution
context; it does not automatically make a portable contribution's source invalid.
Contributions that depended on an old runtime quirk can fail or lose their advantage
when remeasured, which is the intended outcome.

## What Cacheon depends on upstream

`compat` checks the three seam-table chokepoints in
[`cacheon/seams.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/seams.py)
and the supporting surfaces below:

| Upstream surface | Why Cacheon needs it | What movement can break |
|---|---|---|
| `run_scheduler_process` | Scheduler-only candidate load gate | Process roles, spawn path, rank identity |
| `ModelRunner.load_model` | Node binder: replaces the `forward` of named modules once weights load | Module tree, call arguments, capture order, engine-state layout |
| `create_tree_cache` | Prefix-cache binder and content check | Cache construction, handoff methods, KV pool and host-tier layout |
| `BaseFusedOp.forward_native` | The node audit's honest twin | Native reference paths and the twin's measured noise |
| `Engine.generate` logprob API | Reference scoring and quality observation | Request, logprob, and streaming schema |
| `ServerArgs` fields | Deterministic engine launch policy | Model, graph, memory, seed, backend, and logging controls |
| Kernel libraries (Torch, Triton, FlashInfer, CUTLASS DSL, sgl-kernel, DeepGEMM) | Kernel surface under every contribution | JIT products, numerics, throughput, validator agreement |

Kernel-library rows are record-only in the canary; the commissioned image pins them.

## Compatibility canary

```bash
python -m cacheon.cli compat
```

The canary imports the installed runtime, reports its version, and inspects every
registered seam. It does not load a model or require a GPU. A version mismatch
is printed as `DIFFERS from pin`, fails the version check, and makes
`python -m cacheon.cli compat` exit with status 2. A green result means the exact
pin and expected symbols and signatures are present; it does not mean behavior
or performance is unchanged.

### Interpret the output

- **Import or signature FAIL**: the version cannot be used for scoring until a
  reviewed adapter restores the same semantic boundary.
- **`DIFFERS from pin`**: the version row fails even when every inspected seam
  remains present. Restore the exact pin or complete a reviewed pin bump; do
  not treat signature compatibility as arena identity.
- **All seams intact**: imports and signatures match; proceed to behavioral
  negative controls and GPU/model validation.

Capture the complete output with installed package versions and image digest,
not only the final “ALL SEAMS INTACT” line; the recorded kernel-library versions
matter.

Run the Bittensor SDK canary separately:

```bash
python -m cacheon.cli chain-compat
```

This separation prevents an unrelated chain SDK change from being confused
with an Engine runtime change.

## Scheduled seam check

[`scripts/check_sglang.py`](https://github.com/latent-to/cacheon/blob/main/scripts/check_sglang.py)
runs weekly through the `sglang canary` GitHub Actions workflow and can be run locally
with `python scripts/check_sglang.py`. It separates notification from compatibility
failure:

- a newer PyPI release emits a warning but exits successfully; a bump is a
  reviewed rebaseline decision, not an automatic dependency update;
- an installed SGLang version that differs from `PINNED_SGLANG`, or a failed registered
  seam/API check, exits nonzero;
- when SGLang cannot be imported, the PyPI release check still runs but seam coverage is
  skipped, so the check does not claim compatibility evidence it could not produce.

Full seam coverage needs an environment where the pinned package imports. Behavioral
GPU/model proof remains a separate step.

## Pin-bump procedure

1. **Freeze the candidate upstream build.** Record the exact SGLang version,
   repository revision, base image digest, Torch/CUDA stack, native packages,
   and supported GPU architectures.
2. **Run the static canary.** Update seam adapters only where the registered
   semantic boundary is unchanged. A moved upstream symbol is not permission to
   widen a slot.
3. **Run contract tests.** Exercise manifests, target resolution, node and cache
   binding, graph capture, deterministic Engine-tree materialization, and
   qualification schemas.
4. **Run negative controls.** Confirm missing, ambiguous, wrong, graph-unsafe,
   and sabotage contributions still fail closed or route to stock as specified.
5. **Run real GPU/model qualification controls.** Establish stock stability,
   paired-replay noise, pristine T grading, worker cleanup, supported topology,
   and known-good/known-bad candidates under the new exact environment.
6. **Recalibrate the arena.** Freeze new workload, noise, quality, resource, and
   timing policy through the normal reviewed arena process. Do not reuse old
   empirical thresholds by assumption.
7. **Create new identities.** Runtime, base engine, arena, evaluation stack,
   and native build digests must reflect the bump.
8. **Publish migration state.** Mark prior crowns by their original pin; do not
   imply they reproduce on the new one until they do.

## Behavioral proof after the canary

A pin is ready for authority only after progressively stronger controls:

1. a faithful node bundle passes `check`: audit against stock and captured
   completion on every rank at the real TP width;
2. a deliberately wrong node bundle, such as `examples/miner_node_wrong`, fails the
   audit rather than falling through an unobserved upstream path;
3. a faithful cache passes the content check, and a cache serving corrupted state is
   refused;
4. two complete stock engines replay the arena workload in both lane orientations
   within the calibrated noise;
5. candidate C changes only the selected target delta;
6. pristine T regrades sealed outputs without candidate code present; and
7. known champions are rebaselined rather than inheriting old speedups.

The wrong-bundle control is especially important. A faithful candidate can appear
green even if the adapter never bound and SGLang silently served stock. The negative
control proves that the intended call path is under Cacheon's audit and receipt
authority.

## Seam design rule

The [slot contract](../architecture/slot-contract.md) is the stable miner-facing ABI;
SGLang adapters are version-specific glue. Keep upstream-specific imports and call
translation in the adapter layer. If the new runtime cannot preserve a validator-owned
call site and sampling, and correctness judged against stock in the running engine, do
not carry that target forward unchanged.

## Evidence migration rule

A new pin creates a new runtime digest and therefore new evaluation-stack,
arena, and native-build identities. Historical evidence remains valid
for the old context; it does not become corrupt, but it cannot authorize the
new one. Record whether each result is:

- static canary evidence;
- `check` evidence (audit and captured execution on public requests); or
- complete-engine qualification evidence.

This prevents a successful import check from being cited as a serving proof,
or a champion from another runtime identity from being quoted as the active baseline.
