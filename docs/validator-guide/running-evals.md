# Verification and diagnostics

Cacheon exposes three local contribution checks: `scan`, `verify` and `check`.
Complete-engine speed and quality decisions belong to a registered validator arena;
the CLI has no local qualification command.

| Check | Question answered | Authority |
|---|---|---|
| `scan` | Does the declared bundle tree satisfy the static intake policy? | Local admission diagnostic |
| `verify` | Does the bundle scan clean, resolve to a registered target, import, and match each entry signature? | Local interface diagnostic |
| `check` | In the published arena image, does every bound address pass the audit against stock and complete inside CUDA graphs on every rank? | Local engine diagnostic |
| Qualification | Does the exact delta clear execution, paired-replay speed (policy 17), audit, and pristine-quality gates? | One decision; one complete audited PASS qualifies for settlement |

Unknown bundles still execute code during verification. Run them only inside the minimum
[hostile-code isolation boundary](../security/isolation.md#operator-requirements) used for
the relevant device and contribution class; a Python environment or ordinary container is
not an adequate boundary for untrusted native GPU code.

## Static policy scan

```bash
python -m cacheon.cli scan ./my_bundle
```

The command parses the manifest, applies the Python policy to every declared and vendored
`.py` file, recognizes manifest-declared CUDA sources, and rejects `.patch`/`.diff`
files, symlinks, binary artifacts, undeclared executable material, and files outside the
benign metadata allowlist. It scans an extracted bundle tree, not a transport archive.
Archive extraction and resource limits belong to finalized intake. A clean result is
defense in depth, not a sandbox or a correctness proof.

The CLI additionally reports a separate Triton compilability heuristic and exits
2 on a finding. That heuristic sees syntax inside every `@triton.jit` body but
cannot prove the body is reachable; an unused broken kernel and the declared
live kernel look the same statically. It is useful miner feedback, not economic
authority. A production compile `FAIL` requires the reachable entry to fail in
the sandboxed build/execution path.

## Interface and engine checks

```bash
python -m cacheon.cli verify ./my_bundle
python -m cacheon.cli check ./my_bundle --model <MODEL_DIR> \
  --engine-config <ENGINE_JSON> --requests <REQUESTS_JSON> --output <NEW_DIR>
```

`verify` scans the bundle, resolves it to a registered target, then imports each
entry in a spawned child and checks its signature. It runs no forward math,
preparation or graph capture. `check` runs in the published arena image with the
arena's engine configuration and public requests. A fresh eager engine audits
its calls against stock on the same call; a second fresh, graphs-on engine
must complete every bound address inside a capture on every rank (the prefix
cache, which SGLang serves outside the graphs, counts on any completion). Logs,
inputs and raw receipts are retained under `--output`. Neither command establishes
end-to-end speed, pristine quality, or production isolation.

## Performance development

The repository has no public complete-engine benchmark command and no command that
materializes a validator's incumbent stack. Local performance work is a
contributor-controlled experiment built with external launch and profiling tooling.
It is comparable to a named arena only when the operator has published the complete
contract and the contributor reproduces every disclosed input.

Freeze these inputs first:

| Input class | Required identity or value |
|---|---|
| Candidate | Canonical bundle content hash, target ID, selected variant, and the exact source/native build under test |
| Comparison stack | Exact incumbent manifest and engine-tree identities; if unavailable, name the substituted baseline and do not call the result an arena reproduction |
| Runtime | Container/base digest, SGLang revision, model content identity, launch arguments, environment, dtype, graph mode, and cache configuration |
| Hardware | GPU model, driver/runtime, device sets, TP/EP/DP degrees, rank mapping, clocks/power policy, and interconnect topology |
| Workload | Request corpus identity, concurrency and release schedule, warmup, window count, work accounting, and timing boundary |
| Activation | Evidence that the candidate engine activates only the selected target delta and the incumbent engine runs the identical stack without it |

Then mirror the production comparison rather than timing one engine at a time:

```text
two disjoint, equally sized device sets, A and B
orientation 1: incumbent on B, candidate on A; replay the same work concurrently in paired windows
orientation 2: boot both engines fresh on the opposite sets; replay again
ratio(o) = pooled incumbent elapsed time / pooled candidate elapsed time in orientation o
speedup  = geometric mean of ratio(1) and ratio(2)
```

Swapping the device sets cancels a stable per-set speed difference. Keep CUDA graphs
and the prefix cache as the arena configures them: cache hits, MTP acceptance and
output generation are part of the measured cost. A gain no larger than the spread
between windows is unresolved. A profiler range may explain a mechanism, but the
speed claim uses end-to-end elapsed time for the fixed work.

Record every frozen identity; raw per-window elapsed time and completed work for both
engines in both orientations; warmup, ordering and failure history; the formula; the
activation evidence on every expected rank; and tool versions with an immutable location
for raw logs. Without these, label the result a profiling observation. Do not
substitute guesses for the validator's private workload, calibration or incumbent
identities.

This is engineering evidence only. A contributor-controlled run cannot supply finalized
intake identity, validator-owned materialization, hidden work, frozen calibration,
no-egress worker authority, the sealed stopping rule, or the audit and pristine-T
products that a qualification attempt retains.

## Reading validator outcomes

- `PASS` means one complete attempt cleared every registered gate.
- `FAIL` requires complete evidence of a candidate-attributable violation. A speed
  grade that has not established a gain by the last sealed window is `FAIL`, as is a sealed futility stop after the
  first orientation, or a measurably slower or attainment-missing read once both orientations exist.
- `NO_DECISION` covers infrastructure faults, missing authority, or incomplete evidence
  and is eligible only for bounded retry.
- `qualified` means one complete audited PASS is retained; settlement reopens that
  exact contribution and its evidence.
- A qualified V17 PASS earns reward only when it beats the best earlier rewarded PASS
  against the same arena and incumbent stack by 1.5%; see
  [Settlement and weights](settlement-and-weights.md).

Never infer rejection from absence in `chain-status`; that command sees public chain
state, not the validator's private lifecycle database.

## Evidence scope

| Evidence | Establishes | Does not establish |
|---|---|---|
| Clean scan | Static policy accepted the declared tree | Safety or correctness |
| `verify` | Target resolution, imports, and entry signatures | Numerical correctness, graph behavior, or speed |
| `check` | Audit against stock and captured execution on the public requests | Hidden workload behavior, paired speed, or pristine quality |
| Local paired replay | A development performance hypothesis | Registered arena identity or quality authority |
| One complete audited arena PASS | Qualification and settlement eligibility for that exact context | Reward eligibility, integration, or release readiness |

See [Qualification](qualification.md), [Fidelity](fidelity.md), and
[Evidence and replay](../security/evidence.md).

## Source anchors

- [CLI](https://github.com/latent-to/cacheon/blob/main/cacheon/cli.py)
- [Static scanner](https://github.com/latent-to/cacheon/blob/main/cacheon/sandbox.py)
- [Miner development check](https://github.com/latent-to/cacheon/blob/main/cacheon/miner_check.py)
- [Node binder and audit](https://github.com/latent-to/cacheon/blob/main/cacheon/integrations/sglang_nodes.py)
- [Paired replay runtime](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/goodput_runtime.py)
- [Statistical grade](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/service_capacity.py)
