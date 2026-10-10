# DeepSeek-V4.1-Flash B300 arena

Production-path results from 2026-10-09 UTC on one eight-GPU B300 host serving
DeepSeek-V4.1-Flash through the
[paired replay qualification](../validator-guide/qualification.md), using
source revisions `3f41699b` (diagnosis), `51a359db` (fix) and `9cbce877`
(audit of registered controls). They establish the arena's end-to-end wall
clock, the cost the node seam charged the candidate lane, the sealed noise
boundary, and the audit's rejection of wrong output. They are not a crown or a
serving claim.

## Runtime

Two commissioned pairs, each a TP2/EP2 candidate lane and an equal baseline
lane, ran SGLang 0.5.21 with the Engram host tables resident in 2 MiB pages,
CUDA graphs captured up to batch 24, a static memory fraction of 0.85, and the
checkpoint held in RAM behind the sealed model path. Every scheduler thread was
pinned to its sealed vCPU; the host's fast vCPUs follow the pattern
`{0, 1} + 15k`, and the pin table bound at commissioning matched the measured
rates. GPU clocks were locked at 1905 MHz throughout.

## Qualification wall clock

The first complete run of every stage on this model was the whole-model
identity control on pair 4-5 (request `851d5e9a`), 75.3 minutes from request
acceptance to verdict.

| Stage | Measured | Composition |
|---|---:|---|
| Speed, two orientations, four windows | 55.0 min | Two engine boots of 9.5 min including conditioning; windows of 8.8 to 9.2 min |
| Eager audit | 11.4 min | One engine boot; 872 audited calls, 0 violations |
| Pristine reference T | 8.0 min | One engine boot and the reference replay |
| Request to verdict | 75.3 min | Intake, the three stages above, teardown and publication |

A run that stops at the futility rule after the first orientation takes
28 minutes. GLM-5.3's last production qualification on the same protocol took
99 minutes, of which 15 were the candidate build and 20 were four 5-minute
boots; V4.1 boots are longer because the Engram tables load on top of the
weights, and its windows are shorter.

Engine boots re-ran the FlashInfer MoE kernel autotune (16 profiles, 76 s)
because the runtime seed was built from a development check that tuned nine
profiles. That cost is per lifetime and is the next boot-time and
boot-to-boot-variation item.

## The candidate lane's dispatcher cost

With the `3f41699b` image, the faithful identity control on
`model.layers.*.mlp` ran 4.4 to 8.0 percent slower than stock on either GPU
pair and failed the futility rule at the second window. Timing the dispatcher
in the commissioned image on a 40-module dummy model isolated the cause to
host time per dispatched call:

| Piece | Per call, `3f41699b` | Per call, `51a359db` |
|---|---:|---:|
| Capture-authority probe (failed import of the removed legacy module) | 28 µs | 0.8 µs |
| Receipt root resolve on every completed call | 35 µs | 2.6 µs |
| Call descriptor and variant match | 12 µs | memoized |
| Whole dispatched call | 85 µs | 9 to 12 µs |

Forty calls per eager step cost 3.5 ms, about five percent of a prefill chunk
step, which matched the measured slowdown. A single-layer identity control on
`model.layers.3.mlp` with the old image read 1.02, 1.02, 1.01 and 1.00 against
stock over four windows, confirming the per-layer scaling. The fix imports each
capture authority once per process, resolves an absolute receipt root once, and
memoizes the registry selection per graph mode, dtype, width and device.

## Noise and the sealed boundary

The whole-model identity control, a bundle that executes stock unchanged,
received `PASS` at a pooled speedup of 1.0127. Its per-window
candidate-to-stock elapsed ratios were 1.000 and 0.995 in the first
orientation and 0.987 and 0.953 in the second, where the stock engine booted
for that orientation ran 2.5 percent slow for one window and 7 percent slow at
the start of the next, recovering as the other pair's run ended. The sealed
noise model inherited from GLM, 1 percent per window and 0.4 percent per boot,
put the final-look pass line at +0.74 percent.

Both pairs now seal window and boot noise at 0.02, the ceiling the frozen
speed policy allows. The final-look pass line is +2.2 percent and the
third-look line +4.3 percent; the futility rule is unchanged at -1.5 percent
after the first orientation. The controls below ran under this seal.

## Controls on the commissioned image

All controls ran with the 0.02 noise seal, both pairs loaded concurrently as
in production, on the `51a359db` image; the registered-control audit rows ran
on `9cbce877`.

| Control | Pair | Expected | Result |
|---|---|---|---|
| Faithful identity on `model.layers.*.mlp` | 0-1 | `FAIL`, every window within noise of 1.00 | `FAIL` at the final look; window ratios 1.000, 1.001, 0.977, 0.947 |
| Skip-MLP, returns the hidden states untouched | 4-5 | Speed `PASS`, rejected by the eager audit | `FAIL` at the futility rule, candidate slower; never audited |
| Candidate engine killed mid-window | 0-1 | Infrastructure `NO_DECISION`, evidence retained | `NO_DECISION` three minutes after the kill, reason `outer_session_process`, result retained |
| Pod service restarted mid-run | 0-1 | Interrupted attempt resolved as infrastructure, fresh attempt to a verdict | Restarted service published the interrupted attempt as `no_decision` / `pod_service_restart` on its own; resubmitted wrong-MLP reached `FAIL` after four windows at 1.007, 1.015, 1.012, 1.020 |
| Skip-MLP under the registered-control audit policy | 0-1 and 4-5 | Speed non-`PASS`, then audit `FAIL` with violations | `FAIL` at the audit on both pairs: speed stopped at the first-orientation look (window ratios 1.024 and 1.051 on 0-1, 1.058 and 0.954 on 4-5), then the eager audit graded 34,880 calls with 34,800 violations and a worst fraction of 0.0000; 47 minutes from submission to verdict, 10.6 of them the audit |
| Faithful identity on `model.layers.*.mlp` | 4-5 | `FAIL`, every window within noise of 1.00 | `FAIL` at the final look; window ratios 1.017, 1.025, 0.989, 0.995 |

Each pair is one complete measurement: the stock and candidate engines on its
two lanes replay concurrently, then swap. The host runs two pairs so two
submissions qualify in parallel. The runs below show the neighbouring pair's
transitions as an interference source: two of three identity runs leaned
toward the candidate in the orientation during which the other pair booted or
tore down, while the identity run whose neighbour stayed loaded throughout
read 1.017, 1.025, 0.989 and 0.995. That skew is the next noise item.

The identity control's second orientation again favoured the candidate lane,
by 2.3 and 5.3 percent, with the last two quarters of the final window at
0.883 and 0.900 while the other pair tore down. Its pooled estimate of about
+1.9 percent sat 0.3 points under the +2.2 percent pass line. Two identity
runs on two pairs have now shown the same orientation-two skew, so it is a
cross-pair effect rather than window noise; it is recorded here as the next
noise item, not corrected by this change.

The first skip-MLP control did not reach the audit because nothing wrong can
be faster on this workload: the sealed replay fixes every request's token
count, ignores end-of-sequence, samples at temperature zero and runs DSPARK
speculative decoding, so any change to the model's answer collapses draft
acceptance and the candidate decodes slower (78 against 93 tokens per second
here). A sealed submission that fails speed is terminal, so the audit's
grading had only been exercised on the identity run, 872 audited calls with
every row graded and 0 violations. Revision `9cbce877` lets the sealed
qualification policy name control bundle deltas whose speed non-`PASS`
continues into the eager audit; pristine T never runs after a failed speed,
the verdict stays `FAIL`, and any other delta remains terminal at speed. The
registered-control rows above are the production rejection of wrong output on
this arena.
