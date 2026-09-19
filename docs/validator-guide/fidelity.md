# Fidelity and quality authority

Cacheon uses different quality checks at different stages. Confusing them creates a major
security error: a useful development diagnostic is not necessarily safe as a grading
oracle.

## The quality layers

| Layer | Purpose | Crown authority? |
|---|---|---|
| Typed slot verification | Fast ABI, output-layout, numerical, applicability, and graph preflight | No |
| Contributor-controlled model A/B or in-engine audit | Development feedback and integration diagnosis | No |
| Registered audit-only role | Exact slot × TP-rank live-call evidence graded by the trusted host | Yes, when the arena policy requires it |
| Registered fixed-stock exact-count gate | Candidate-only hidden-task generation compared with one sealed stock observation | Yes, when the registered profile selects it |
| Pristine T over sealed timed trajectories | Candidate-free, retained production quality evidence | Yes, as one required part of qualification |

## Slot verification

Each registered slot owns a typed input/output contract and correctness mode. Depending
on the slot, verification may use all-close tolerances, matched ratio, cosine similarity,
or top-k overlap. The verifier jitters eligible dimensions, checks variant routing, and
requires graph evidence where applicable.

This catches many broken kernels cheaply, but it samples a finite contract. It cannot by
itself establish end-to-end model behavior or serving quality.

## Development quality evidence

Engine developers may compare aligned rollout distributions, task behavior, and sampled
dispatcher calls while integrating a contribution. KL-like measurements are interpretable
only when stock-vs-stock variation is within the setup's calibration; launch
nondeterminism can otherwise dominate the signal. Tail and argmax-disagreement measures
help expose sparse corruption that a mean alone can hide.

An in-engine audit samples live candidate calls and compares them with a stock
calculation. It is valuable for debugging nondeterministic stacks, but the candidate
process contains the audit machinery and cannot grade a hostile engine. Framework-mode
token matching has the same limitation. These checks are engineering tools, not a
registered target or crown authority.

## Registered audit-only role

An arena may require a sealed audit plan after the resident speed stage. This is not the
candidate-side mechanism above. The audit role has its own plan and runtime identity,
executes outside the charged, versioned speed reads, and emits an exact slot × TP-rank/PID
witness. Trusted-host regrading checks expected rank coverage, unique processes, minimum
call counts, and retained violations or protocol errors without importing PyTorch.

The durable witness canonicalizes live floating-point facts into stable decimal strings
before computing receipt identity. A missing rank, duplicate/reused process, insufficient
coverage, tampered decimal fact, or unreopenable audit artifact cannot be waived by a
candidate-side report.

### Worker controls and comparator margin

The sealed `SlotAuditPolicy` owns the sampling rate, validator seed, expected slots,
expected TP member count, and minimum calls. For the separate audit-only candidate
session, the isolated engine worker maps that authority into two process-local controls:

- `CACHEON_SLOT_AUDIT` is the policy's integer parts-per-million sampling rate converted
  to a fraction in `[0, 1]`.
- `CACHEON_SLOT_AUDIT_SEED` is the policy's validator seed converted to an integer. Every
  rank receives the same seed so a sampled collective baseline is entered by all ranks
  rather than deadlocking on rank-divergent sampling.

The worker leaves both values empty when no audit policy is present. Charged
Timed speed sessions therefore have no audit sampling or audit receipt and retain
their sealed graph configuration. The audit-only session is eager and untimed; the
worker disables CUDA graphs for that role. An unexpected audit receipt in a charged
candidate session is a protocol error.

The tensor comparison still occurs inside the candidate engine. The node adapter,
[`cacheon/integrations/sglang_nodes.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/integrations/sglang_nodes.py),
grades each row of the candidate's result and engine-state rows against stock's on the
same call, under the tolerance and 75% window bar the
[slot contract](../architecture/slot-contract.md) defines, and hands
[`cacheon/audit.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/audit.py)
the fraction it measured for the rolling receipt. Both operands are the engine's own
low-precision results, so the tolerance is measured on an honest twin rather than
declared; changing the floor, the ceiling or the bar requires fresh honest and wrong
control evidence and review.

The environment variables and rolling receipt are only worker instrumentation. Running
that instrumentation during development does not create crown authority. Authority
requires the independently sealed audit-only plan, a distinct session from every timed
role, bounded transport out of the worker, and Torch-free host regrading through
[`cacheon/audit_gate.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/audit_gate.py).

### Audit outcomes

The host grades the receipts into one of three outcomes. A verdict about the kernel
requires compared calls; an audit that compared too little says nothing about the kernel.

| Outcome | Reason | Condition |
| --- | --- | --- |
| `PASS` | — | The exact slot × TP-rank/PID matrix is present, every member meets the per-member minimum, no candidate output was uncomparable, and any call outside tolerance is a rare near miss. |
| `FAIL` | `slot_audit_failed` | A compared call scored more than `0.03` under its recorded bar, near misses exceed one compared call in 100 for a slot and rank, or a candidate output could not be compared. Terminal for the bundle. |
| `NO_DECISION` | `audit_not_covered` | Receipts are missing, malformed, or under the per-member minimum. The audit role's shortfall; the bundle takes the ordinary requeue path and is not failed. |

A near miss is a compared call under its bar by at most `0.03`. Recorded honest receipts
sit within `0.004` of a `0.985` bar; recorded wrong kernels scored `0.18` or lower. Kernel
faults are graded before coverage, so a wrong kernel on a thinly covered run still fails.
The timed role has already proved the candidate executes, which is why an under-covered
audit is attributed to the audit role rather than to the bundle.

A stock baseline that raises inside the audit is counted as `baseline_refused`, not as a
comparison error: no candidate output was compared, so it can reduce coverage but cannot
fail a kernel. An error while comparing a candidate output remains a comparison error.

### Production audit canary

Before audit evidence can satisfy a production activation gate, run the complete current
qualification path on the exact production image, model, topology, target set, audit
policy, and runtime identities. Retain the following controls:

1. Run an honest candidate through one complete qualification. Every registered slot
   on every TP rank must meet the sealed minimum call count and grade `PASS` with zero
   comparison errors, and the retained attempt must reopen independently.
2. Run the registered residual-drop sabotage candidate through the same audit path. Its
   typed audit witness must make the aggregate qualification a nonretryable failure.
3. Inspect every charged speed session. Both audit environment values must be
   empty, no audit receipt may appear, and the charged result must retain the graph mode
   sealed by its speed plan.
4. In copies of the retained artifacts, remove or alter one slot/rank receipt and alter
   one audit-policy or request binding. Reopen and downstream settlement validation must
   reject every mutation rather than accepting an auditless or mismatched report.

The retained canary receipt is evidence only for those exact identities. Passing these
controls does not close in-process tampering, audit-role fingerprinting, timed-workload
fingerprinting, or other accepted residual risks; activation binds the canary separately
from its residual-risk acceptance.

## Pristine T

Production qualification launches a separate candidate-free reference session after the
resident speed executors are quiescent and the required audit stage completes. T:

- reopens an empty evaluation stack with no proposal contributions;
- binds the same runtime/model/reference identity and frozen calibration;
- receives sealed timed-read prompt and trajectory identities from the trusted controller;
- teacher-forces those trajectories;
- emits bounded token-level teacher evidence; and
- runs the registered hidden quality work.

The controller then regrades raw evidence under the frozen metric policy. Candidate C
does not choose prompts, support tokens, thresholds, or the hidden judge. Incumbent B′ is
also untrusted and is never substituted for T.

## Fixed-stock exact-count profiles

A registered profile may instead bind one retained stock observation and an exact-count
regression policy. The stock artifact contains every ordered output-token sequence and
its hidden-judge receipt, but no trusted aggregate score. Commissioning seals the
artifact reference, observation digest, prompt/generation/admission envelope, and policy;
runtime reopening rehashes the bytes and rejudges every retained row before comparing it
with the candidate.

Candidate evaluation uses the already-resident lanes at the profile's sealed admission
width. It does not rerun stock for each bundle. A missing, foreign, or envelope-mismatched
stock artifact is infrastructure and stops qualification; a configured historical score
is not a substitute for reopenable evidence. Only an intact candidate generation whose
hidden judge runs successfully can produce a quality PASS or FAIL.

## Calibration

A crown-authoritative `CalibrationManifest` is content-addressed and frozen. It binds:

- pristine reference manifest;
- arena, runtime, base engine, model, logical hardware, and workload;
- verification policy and controller distribution;
- raw calibration evidence and seeds;
- speed margin, noise multiplier, and maximum noise; and
- metric envelopes, candidate deltas, and any absolute floors.

Controls include expected stock/positive passes and negative failure. A provisional,
stale, incomplete, or context-mismatched calibration is not usable for a crown.

## Quality decisions

The registered policy can grade metrics such as argmax disagreement, support coverage,
teacher NLL, KL-derived measures, tail rate, and task score. Exact metrics depend on the
frozen arena calibration; documentation must not present one universal threshold as the
Cacheon quality contract.

Missing teacher coverage, wrong prompt/trajectory identity, tampered evidence, or
unreopenable calibration yields `NO_DECISION`. A measured candidate regression yields
`FAIL`. Quality `PASS` is still only one prerequisite alongside execution, speed, audit,
and identity evidence. It is also the leg of the graph proof that catches a stale replay:
a candidate that is captured but returns its capture-time answer passes the execution
check and fails here.

## Honest limits

- Finite prompts and hidden tasks cannot rule out all shape or workload overfitting.
- A pristine implementation can contain bugs; it is an independent authority, not a
  mathematical proof.
- Calibration is specific to the registered runtime, model, hardware, and workload and
  must be redone when those identities change.
- Contributor-controlled KL, task, or audit output is useful evidence for engineers but
  has no settlement effect.
- Passing fidelity does not establish license, security review, or release readiness.

See [Evidence and replay](../security/evidence.md) for how raw quality products are
retained and reopened.

## Source anchors

- [Typed slot contracts](https://github.com/latent-to/cacheon/blob/main/cacheon/slots.py)
- [Tensor output specifications](https://github.com/latent-to/cacheon/blob/main/cacheon/tensor_spec.py)
- [Qualification quality model](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification.py)
- [Torch-free audit gate](https://github.com/latent-to/cacheon/blob/main/cacheon/audit_gate.py)
- [Pristine wire protocol](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/reference_protocol.py)
- [Calibration authority](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/calibration.py)
