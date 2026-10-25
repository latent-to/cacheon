# Measurement and decision policy

Production Cacheon does not reduce a proposal to one self-reported score. It derives a
three-way qualification decision from retained execution, graph, speed, and pristine
quality evidence, then reopens that complete audited PASS for settlement.

## Marginal comparison

For a registered target, the production version-3 evidence protocol constructs
one exact marginal comparison:

- B: the exact frozen incumbent stack; and
- C: that same stack with one registered target replaced.

B and C run one engine at a time on the whole commissioned lane and replay the
same sealed agent slice (see
[Finite agent replay](qualification.md#finite-agent-replay)). B is measured once
as a retained stock reference of four fresh boots of N windows each; C gets up to
two fresh boots of N windows on the same lane; after a speed PASS one fresh stock
bracket boot rechecks the reference. The controller fixes
the slice, the operating load and the output budgets, releases requests in
lockstep rounds, and timestamps every request on the host. The durable resident
witness retains each boot's turn records, physical-lane authority, operational
timing, and budget. After the speed lifetimes are quiescent, qualification runs
registered eager audit A when the plan requires it, destroys candidate lifetimes,
and then runs pristine T. Reopen recomputes the costs and the frozen decision
from the retained records, not from raw session frames. Candidate-reported
aggregate throughput is never accepted as authority.

The policy-19 speed estimate is:

```text
cost(boot)  = log(mean elapsed serving seconds over the boot's N windows)
speedup     = exp(mean cost of the 4 reference boots - mean cost of the k candidate boots)
SE          = sqrt(sum of reference boot variances / 16 + sum of candidate boot variances / k^2)
PASS        = exp(log speedup - z * SE) > 1 at a sealed look
```

A boot's variance is the larger of `null_noise²/N` and its within-boot jackknife
variance, plus `boot_noise²`; requests sharing a window are not treated as
independent replications. The reference variance is shared across candidates.
Looks fall at candidate boot boundaries and spend the sealed 12.5% error budget
(90% one-sided at the final look). After boot 1 the stage can end only on the
sealed `futility_margin` (FAIL); a PASS needs both boots and the pooled
attainment bound. A stock bracket that reads faster than the reference beyond a
one-sided 1% test turns the PASS into `NO_DECISION` and retires the reference.

The exact thresholds come from a frozen `CalibrationManifest` bound to the
measured reference, arena, runtime, model, hardware, workload, and verifier.
Provenance still records the exact controller, but measurement reuse is not
invalidated by an unrelated controller revision. Invalid or incomplete
measurement yields `NO_DECISION`, never a fabricated miner loss or reward.
Retained evidence regrades under the version that produced it. Evidence sealed
below version 16 (the batch-cell policies 8–15 and the schedules before them) is
refused rather than regraded.

Timed replay requests collect no log-probabilities (`top_logprobs_num` 0):
quality is the teacher-NLL-only mode, digest-bound by a zero top-k width in the
qualification profile and the raw quality binding. The pristine engine
teacher-force-scores the exact retained token stream (target NLL and the
teacher's own argmax per position) — the text the candidate was fast at is the
text it is judged on, and no candidate code executes during scoring. No
candidate distributions are retained, so no distribution evidence exists:
absence is explicit (null, uniformly enforced at every layer), never zeros, and
a threshold policy naming a distribution metric (`topk_kl`, `argmax_rate`,
`coverage_dev`) against teacher-NLL-only evidence refuses outright.
Distribution-level numerics coverage remains with the in-engine slot audit
stage. Evaluation work never shares the clock with a speed measurement.

## Complete qualification decision

A candidate can pass only when all required products agree:

| Product | Failure meaning |
|---|---|
| Execution evidence | Wrong/missing role, launch, device, protocol, or completion; current source requires complete per-rank execution evidence before grading, and on a graphs-on run a completion counts only when the dispatcher recorded the candidate inside a CUDA-graph capture |
| Speed evidence | Below the calibrated bar, or missing/unfit evidence that prevents a valid decision |
| Audit-only evidence | Missing slot × rank/PID coverage, retained violation, or protocol error |
| Pristine quality evidence | Regression against frozen metric envelopes or hidden work |
| Identity checks | Evidence does not describe the committed arena, stack, target, or delta |

Attributable violations yield `FAIL`. Infrastructure, missing evidence, or stale
identity yields `NO_DECISION`; no window is ever discarded. Only complete green
evidence yields `PASS`.

## One qualification per bundle

One complete audited `PASS` becomes `qualified` and supplies the settlement
candidate. No independent second qualification is scheduled. Settlement reopens
the exact speed, graph, audit, and pristine-quality artifacts before accepting
its score. Historical pairs retain their original identities and lower score.

## Settlement cohort over one incumbent authority

The store leases one economically unblocked group sharing a qualification authority and
one exact incumbent stack. Stale candidates are held. Across all current registered rows
in that leased group—even rows for non-overlapping targets—the planner selects one winner
by conservative speedup and uses finalized arrival order as the tie-break. The shared
incumbent advances once, so every other current row is held for a fresh qualification
against the new stack rather than treated as an independent per-target argmax.

The winning transaction may emit crown, retirement, neutralization, adoption, and stack
transition events. Targets have disjoint node roots; manifest order and bundle
packaging never decide overlap.

## Reward policy follows the activated generation

Under retained legacy V1 authority, each active registered target defines one
reward family. The policy derives standing credit from qualified marginal
improvement and age. The normative conversion, decay equation, and integer
rules live in
[Legacy V1](../reference/emissions-policy.md#legacy-v1).

A qualified PASS earns reward only when its speedup beats the best earlier
rewarded PASS against the same arena and incumbent stack by the reward margin:
1.5% for V17, V18 and V19, the sealed `min_margin` for historical policies. The first PASS
against a commissioned baseline is eligible; see
[Settlement and weights](settlement-and-weights.md).

Standing-claim age begins at the proposal's finalized submission block, which
settlement stores as `crowned_block`; discovery lifetime likewise begins at
that submission block via `awarded_block`. Qualification or settlement delay
never resets reward age.

Packaging, integration, and release records do not create additional families. Discovery bounties
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

A crown means: under the registered arena, workload, and calibration, one complete
audited qualification found that the exact delta improved the exact incumbent with
acceptable measured quality.

It does not mean:

- the contribution improves every model, topology, or traffic mix;
- the measured speedup is a service-level capacity guarantee;
- the proposal is licensed, maintainable, reviewed, or ready to ship; or
- any score produced outside registered, retained qualification has economic effect.

## Source anchors

- [Speed grade](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/service_capacity.py)
- [Reward comparison](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/evaluation_order.py)
- [Frozen calibration](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/calibration.py)
- [Qualification runner](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification_runner.py)
- [Resident crossover](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/crossover_runtime.py)
- [Settlement](https://github.com/latent-to/cacheon/blob/main/cacheon/settlement.py)
- [Economics](https://github.com/latent-to/cacheon/blob/main/cacheon/economics.py)
