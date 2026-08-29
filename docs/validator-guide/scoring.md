# Measurement and decision policy

Production Cacheon does not reduce a proposal to one self-reported score. It derives a
three-way qualification decision from retained execution, graph, speed, and pristine
quality evidence, then requires independent reproduction before settlement.

## Screens are not scores

Static, build, ABI, graph, and abbreviated-serving screens protect scarce evaluator
capacity. They may promote, reject, retry, or hold a proposal, but are marked
non-crownable. Their timings and grades cannot update the evaluation stack.

## Marginal comparison

For a registered target, the production version-3 evidence protocol constructs
one exact marginal comparison and selects its speed subpolicy from candidate
features:

- B: the exact frozen incumbent stack;
- C: that same stack with one registered target replaced; and
- B′: a second incumbent read, conditional under v7 and mandatory under v8.

Hot-swappable candidates use v7 on the standing resident pair: B/C decides a
clear result under precommitted invariant bounds, and only an inconclusive ratio
authorizes B′. Both arms enter measurement through a swap/recapture in v7.
Non-swappable candidates use v8's separate baseline and candidate engine
processes and always read B/C/B′ because the quality gate consumes the second
stock read. C′/B″ are unreachable under both current policies; they survive only
in historical v2–v5 evidence. During execution, the controller fixes prompt
batches and token budgets, serializes timed GPU work, validates bounded batch
frames and token numerators, and records both charged intervals (registered
conditioning plus timed) and timed windows. The durable resident witness retains
the actual versioned schedule, physical-lane authority, operational timing, and
budget. After the speed lifetimes are quiescent, qualification runs registered
eager audit A when the plan requires it, destroys candidate lifetimes, and then
runs pristine T.
Reopen recomputes tokens/second and the frozen decision from typed counts and intervals,
not from raw session frames. Candidate-reported aggregate throughput and resident-screen
rates are never accepted as authority.

The speed estimate is conceptually:

```text
scored_rate = timed_tokens / timed_seconds
v7 clear speedup = C / B
bookended speedup = C / mean(B, B′)
noise       = relative spread of the baseline scored rates
bar         = 1 + max(min_margin, noise_multiplier * noise)
```

Current v7/v8 policy grades the median over steady-state timed windows.
The first read of a resident session pays residual cold-start inside its
conditioning window while a continuation read does not; a scored rate that
charges conditioning turns that positional split into apparent baseline noise
and biases both the measured speedup and the bar. Conditioning therefore stays
outside the scored rate and remains bounded by the sealed operational timing
budget, so a candidate cannot hide work in warmup. Version-1 witnesses, which
graded the charged rate, regrade only under their own sealed arithmetic; the
policy version is part of the witness digest and cross-version splicing is
refused.

The exact thresholds come from a frozen `CalibrationManifest` bound to the
measured reference, arena, runtime, model, hardware, workload, and verifier.
Provenance still records the exact controller, but measurement reuse is not
invalidated by an unrelated controller revision. Under historical v2/v3,
excessive baseline disagreement or per-read scatter could yield
`NO_DECISION`. V4 made every retained read gradable and terminates an
undetermined final spread as `FAIL valid_not_faster`; v5 additionally excludes
later baseline brackets that drift past the sealed ceiling and decides from the
adjacent C/B pair. V6 made B′ conditional, v7 added the symmetric baseline
swap, and v8 precommits B′ on the two-process substrate. Retained evidence
always regrades under the version that produced it.

Policy version 3 replaces each read's single timed aggregate with the median
over per-batch timed windows. The window is the timed batch because host
wall-clock spans at batch boundaries are the only timing the trust model
accepts; every window recomputes exactly from sealed batch evidence and is
retained in the witness rows. A version-3 read also carries a sealed
per-read window-scatter bound (median absolute deviation about the median,
relative): a read whose own scatter exceeds the bound refuses to produce a
scored rate at all, in live grading and in every reopen, so an unfit
measurement cannot be graded anywhere. Version-3 timed reads request no
log-probability collection (`top_logprobs_num` 0): quality becomes the
teacher-NLL-only mode, digest-bound by a zero top-k width in the
qualification profile and the raw quality binding. The pristine engine
teacher-force-scores the exact retained token stream (target NLL and the
teacher's own argmax per position) and hidden tasks grade the same retained
outputs — the text the candidate was fast at is the text it is judged on,
and no candidate code executes during scoring. No candidate distributions
are retained, so no distribution evidence exists: absence is explicit
(null, uniformly enforced at every layer), never zeros, and a threshold
policy naming a distribution metric (`topk_kl`, `argmax_rate`,
`coverage_dev`) against teacher-NLL-only evidence refuses outright.
Distribution-level numerics coverage remains with the in-engine slot audit
stage. Evaluation work never shares the clock with a speed measurement. A version-3 policy also seals a
conditioning slowdown bound: the conditioning span is the only place a
candidate's prefill cost is host-visible, so the candidate's conditioning
seconds must stay within the bound of the baseline's, compared at equal
warmth position. Historical repeat schedules compare C′ with the matching
warm baseline; current v7/v8 have no C′ and grade the initial C/B pair.
Conditioning spans carry warm/cold session structure and positions must never
be mixed. A violation is a clear
candidate `FAIL`: a decode win cannot hide a prefill regression. The check
grades numbers already sealed in every read and adds no measurement time.
Version-1 and version-2 witnesses keep their exact historical bytes and
regrade only under their own sealed arithmetic.

## Complete qualification decision

A candidate can pass only when all required products agree:

| Product | Failure meaning |
|---|---|
| Execution evidence | Wrong/missing role, launch, device, protocol, or completion; current source requires the complete pair-native per-generation rank count before grading |
| Graph evidence | Missing target member/variant/shape coverage or capture/replay failure |
| Speed evidence | Below the calibrated bar, or missing/unfit evidence that prevents a valid decision |
| Audit-only evidence | Missing slot × rank/PID coverage, retained violation, or protocol error |
| Pristine quality evidence | Regression against frozen metric envelopes or hidden work |
| Identity checks | Evidence does not describe the committed arena, stack, target, or delta |

Attributable violations yield `FAIL`. Infrastructure, missing evidence, or stale
identity yields `NO_DECISION`; current v5+ bracket drift is retained and handled
by the registered exclusion rule rather than automatically becoming a non-answer.
Only complete green evidence yields `PASS`.

## Independent reproduction

The first `PASS` moves the reservation to `reproduction_pending`. A second `PASS` must
match the economic identity while using distinct authority, attempt, report, and
selection evidence. The settlement candidate's conservative speedup is:

```text
settled_speedup = min(primary_speedup, reproduction_speedup)
```

There is no single-pass fast path to a crown.

For version-3 resident-family evidence, including current v7/v8 witnesses,
reproduction must also swap the baseline and candidate physical TP-lane
orientations exactly while retaining the same speed-policy and
settlement-control digests.

Here “independent” means the seven required authority, plan, attempt, report, commitment,
secret-commitment, and selection-evidence digests differ. It does not by itself prove
separate operators, hosts, or infrastructure failure domains.

## Settlement cohort over one incumbent authority

The store leases one economically unblocked group sharing a qualification authority and
one exact incumbent stack. Stale candidates are held. Across all current registered rows
in that leased group—even rows for non-overlapping targets—the planner selects one winner
by conservative speedup and uses finalized arrival order as the tie-break. The shared
incumbent advances once, so every other current row is held for a fresh qualification
against the new stack rather than treated as an independent per-target argmax.

The winning transaction may emit crown, retirement, neutralization, adoption, and stack
transition events. Atomic targets explicitly displace overlapping singleton targets;
manifest order and bundle packaging never decide overlap.

## Reward policy follows the activated generation

Under retained legacy V1 authority, each active registered target defines one
reward family. The policy derives standing credit from reproduced marginal
improvement and age. The normative conversion, decay equation, and integer
rules live in
[Legacy V1](../reference/emissions-policy.md#legacy-v1).

Standing-claim age begins at the proposal's finalized submission block, which
settlement stores as `crowned_block`; discovery lifetime likewise begins at
that submission block via `awarded_block`. Qualification or settlement delay
never resets reward age.

An active atomic target suppresses overlapping singleton families. Packaging,
integration, and release records do not create additional families. Discovery bounties
are non-renewable, expire, and share a policy-bounded pool.

The final multi-arena projection is exact integer ppm and is built only after
every active family reopens against current stack and metagraph authority. A
stale, incompatible, or unreopenable claim holds the complete projection. If a
valid active claimant is merely absent from the current metagraph, that family's
allocated ppm is sent to the validator hotkey for the tick; other families keep
their allocations, and the claimant resumes receiving its decayed share if it
returns.

Finite-debt V2 is a retained design, not an implemented lane; it does not reuse this
standing-decay formula. See [Emissions policy](../reference/emissions-policy.md).

Read [Settlement and weights](settlement-and-weights.md) for transaction and publication
details.

## What a result means

A crown means: under the registered arena, workload, calibration, and two attempts meeting
the seven digest-distinctness checks, the exact delta improved the exact incumbent with
acceptable measured quality.

It does not mean:

- the contribution improves every model, topology, or traffic mix;
- the measured speedup is a service-level capacity guarantee;
- the proposal is licensed, maintainable, reviewed, or ready to ship; or
- any score produced outside registered, retained qualification has economic effect.

## Source anchors

- [Raw speed recomputation](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/scoring.py)
- [Frozen calibration](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/calibration.py)
- [Qualification runner](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification_runner.py)
- [Resident crossover](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/crossover_runtime.py)
- [Settlement](https://github.com/latent-to/cacheon/blob/main/cacheon/settlement.py)
- [Economics](https://github.com/latent-to/cacheon/blob/main/cacheon/economics.py)
