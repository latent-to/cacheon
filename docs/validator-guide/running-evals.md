# Verification and diagnostics

Cacheon exposes two local contribution checks: `scan` and `verify`. Complete-engine
throughput and quality decisions belong to a registered validator arena; the CLI does
not provide a local qualification substitute.

| Check | Question answered | Authority |
|---|---|---|
| `scan` | Does the declared bundle tree satisfy the static intake policy? | Local admission diagnostic |
| `verify` | Do applicable variants satisfy the registered component ABI, reference, and graph checks? | Local component diagnostic |
| Paired replay (speed policy 17) + audit/T | Does the exact marginal delta clear execution, paired-replay speed, audit, and pristine-quality policy? | One qualification decision |
| Independent reproduction | Do two separately bound PASS attempts reopen and agree? | Settlement prerequisite |

Unknown bundles still execute code during verification. Run them only inside the minimum
[hostile-code isolation boundary](../security/isolation.md#operator-requirements) used for
the relevant device and contribution class; a Python environment or ordinary container is
not an adequate boundary for untrusted native GPU code.

## Static policy scan

```bash
python -m cacheon.cli scan ./my_bundle
```

The command parses the manifest, applies the Python policy to every declared and vendored
`.py` file, recognizes only manifest-declared CUDA sources and dependency patches, and
rejects symlinks, binary artifacts, undeclared executable material, and files outside the
benign metadata allowlist. It scans an extracted bundle tree, not a transport archive.
Archive extraction and resource limits belong to finalized intake. A clean result is
defense in depth, not a sandbox or a correctness proof.

The CLI additionally reports a separate Triton compilability heuristic and exits
2 on a finding. That heuristic sees syntax inside every `@triton.jit` body but
cannot prove the body is reachable; an unused broken kernel and the declared
live kernel look the same statically. It is useful miner feedback, not economic
authority. A production compile `FAIL` requires the reachable entry to fail in
the sandboxed build/execution path.

## Component verification

```bash
python -m cacheon.cli verify ./my_bundle
python -m cacheon.cli check --help
```

`verify` scans the bundle, then imports each entry in a spawned child and checks
its signature against the node's stock arguments. It does not fabricate a model
or run forward math. `check` runs in the published arena image with the arena's
engine configuration: it binds the bundle's nodes in a real engine, audits every
sampled call against the stock module on the same call, and captures graphs on
every required rank. Neither establishes end-to-end throughput, pristine
reference quality, or production isolation.

## Performance development

The repository has no public complete-engine benchmark command and no contributor command
that materializes a validator's incumbent evaluation stack. Local performance work is
therefore a contributor-controlled experiment built with external launch and profiling
tooling. It becomes comparable to a named arena only when the operator has published the
complete contract and the contributor can reproduce every disclosed input.

Before launching, freeze these inputs:

| Input class | Required identity or value |
|---|---|
| Candidate | Canonical bundle content hash, target ID, selected variant, and the exact source/native publication under test |
| Comparison stack | Exact incumbent manifest and engine-tree identities; if they are unavailable, identify the substituted stock/reviewed baseline and do not call the result an arena reproduction |
| Runtime | Container/base digest, SGLang revision, model content identity, launch arguments, environment, dtype, graph mode, and cache policy |
| Hardware | GPU model, driver/runtime, visible device set, TP/EP/DP degrees, rank mapping, clocks/power policy, and interconnect topology |
| Workload | Prompt/request corpus identity, batching/concurrency policy, warmup, measured iterations, token/work accounting, and timing boundary |
| Activation | Evidence that C activates only the selected target delta and that B/B′ use the identical stack without it |

Use the deployment's ordinary SGLang launcher and profiler only after those identities are
fixed. For a contributor-controlled diagnostic, materialize a fresh process lifetime for
each arm; do not present an ad hoc in-process toggle as matched evidence. The bracket is:

```text
B  = exact incumbent before the candidate
C  = the same stack with only the selected target delta replaced
B′ = the exact incumbent after the candidate
local_speedup = candidate_rate / mean(baseline_before_rate, baseline_after_rate)
```

For each arm, retain startup/activation evidence, complete warmup, then collect the same
number of measured samples under the same charged-work definition. Keep CUDA graphs in the
declared state and reject the bracket when B/B′ drift is comparable to the claimed gain.
For a prefill target, disable cache behavior that would silently convert repeated inputs
into cache-hit or decode work. A profiler range may explain a mechanism, but throughput
must use the declared end-to-end charged boundary rather than a hand-selected kernel span.

The local result record should contain:

- every frozen identity from the table above;
- raw per-sample B, C, and B′ work counts, durations, and rates;
- warmup count, measurement order, failure/retry history, and B/B′ drift;
- the speedup formula, charged-work denominator, and any exclusion rule;
- activation/fallback evidence for the selected slot on every expected rank; and
- tool versions plus an immutable location or digest for the raw logs.

If any required identity, raw sample, or activation signal is missing, label the result a
profiling observation rather than a matched bracket. Do not substitute the validator's
private workload, calibration, or incumbent identities with guesses.

This bracket is engineering evidence only. It is not the production resident protocol. A
contributor-controlled run cannot provide
finalized intake identity, validator-owned materialization, hidden work, frozen
calibration, no-egress worker authority, or a validator-bound durable attempt with its
aggregate speed witness and referenced graph/quality/T products. The production attempt
uses two isolated physical TP lanes, replays the sealed agent workload on both
concurrently in paired windows, validates richer raw frames and device state, and runs a distinct
audit-only role before pristine T. Those raw frames are not serialized into
`CohortQualificationAttempt`. A local run cannot supply the validator-owned qualification authority.

## Reading validator outcomes

- `PASS` means one complete attempt cleared every registered gate.
- `FAIL` requires complete evidence of a candidate-attributable violation.
- `NO_DECISION` covers infrastructure, drift, missing authority, or incomplete evidence
  and is eligible only for bounded retry policy.
- `qualified` means one complete audited PASS is retained.
- settlement reopens that exact contribution and its retained evidence.

Never infer rejection from absence in `chain-status`; that command sees public chain
state, not the validator's private lifecycle database.

## Evidence scope

| Evidence | Establishes | Does not establish |
|---|---|---|
| Clean scan | Static policy accepted the declared tree | Safety or correctness |
| CPU verify | Exercised component reference checks | CUDA graphs or performance |
| CUDA verify | Exercised component and registered graph checks | Model integration or crown authority |
| Local B/C/B′ bracket | A development performance hypothesis | Registered arena identity or quality authority |
| One arena PASS | Complete qualification under one authority | Settlement |
| One reopened complete audited PASS | Settlement eligibility for that exact context | Integration or release readiness |

See [Qualification](qualification.md), [Fidelity](fidelity.md), and
[Evidence and replay](../security/evidence.md).

## Source anchors

- [CLI](https://github.com/latent-to/cacheon/blob/main/cacheon/cli.py)
- [Static scanner](https://github.com/latent-to/cacheon/blob/main/cacheon/sandbox.py)
- [Miner development check](https://github.com/latent-to/cacheon/blob/main/cacheon/miner_check.py)
- [Node binder and audit](https://github.com/latent-to/cacheon/blob/main/cacheon/integrations/sglang_nodes.py)
- [Qualification runner](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification_runner.py)
- [Resident crossover runtime](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/crossover_runtime.py)
