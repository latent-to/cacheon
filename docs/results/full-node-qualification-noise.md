# Full-node qualification and replay noise

Results from 2026-10-25 UTC on one eight-GPU B300 host serving Muse Spark
through speed policy 19 of the
[replay qualification](../validator-guide/qualification.md), driven through the
production intake, coordinator and remote-worker code over a scratch off-chain
intake store, using source revisions `a1477f7a` (first stock reference
campaign and identity control) and `5e3ead79` (second campaign and identity
control). One engine owned all eight GPUs at TP8/EP8, a configuration no arena
has commissioned. The page also records simulated grade power under the TP4
Muse spreads and a census of arena outcomes since replay grading began. None
of it is a crown or a serving claim.

## Runtime

The stage used the commissioned Muse arena's image, sealed workload and engine
configuration with tensor and expert parallelism changed from 4 to 8, booted
from a TP8 [runtime seed](../validator-guide/new-arena.md), with CUDA graphs.
Each replay window carries 317 turns, of which 301 are timed, with 45.37
million prompt tokens and 306,924 output tokens per window. Fixed-work elapsed
is the wall time to serve the timed turns. Stock service attainment, the share
of turns that meet the sealed decode-rate and first-token bounds, was 0.915 to
0.927 over the 16 stock windows here, against 0.73 to 0.79 (median 0.77) over
44 retained TP4 stock windows of the same workload.

## Wall clock

| Step | Measured |
|---|---:|
| Stock window, fixed-work elapsed | 258.9 to 281.4 s, median 266.5 s (16 windows) |
| Same workload on a TP4 stock lane | 344.8 to 369.7 s, median 357.3 s (44 windows) |
| One engine lifetime: seeded boot, warm-up, two windows, teardown | 14.4 to 14.7 min start to start; about 15.7 min for a request's first |
| Four-lifetime stock reference campaign, from request acceptance | 61 min |
| Identity control, request to verdict, including that reference | 90.3 min (both controls) |
| Identity control's two candidate lifetimes to verdict | 29.5 min |
| Candidate that fails at prepare, request to verdict | 6 to 8 min |

A TP8 window takes 25 percent less time than a TP4 window of the same work.
One TP8 engine times 13.5 windows of one arm per hour of window time; a TP4
pair times 10.1 candidate windows in that hour beside 10.1 concurrent stock
windows. A PASS adds one stock bracket lifetime, the eager audit and pristine
T (see the [qualification timeline](../validator-guide/qualification.md#current-qualification-timeline));
no PASS ran on this host, so that cost is derived from the TP4 audit timing and
the longer TP8 boot, about 28 min.

## Identity controls

An identity bundle, which executes stock unchanged, ran once against each
reference campaign. Speedup is the reference's mean log cost minus the
candidate's, exponentiated; Required is the ratio the speedup must exceed to
pass, `exp(z·SE)` in the v19 grade the qualification guide defines. Both rows use the `5e3ead79` revision of the v19 grader, which
splits the sealed noise between the arms; the first control's verdict under
`a1477f7a` was also `FAIL` at the same speedup.

| Reference | Candidate lifetime means | Speedup | Required | Verdict |
|---|---|---:|---:|---|
| First campaign | 267.0 s, 271.0 s | 0.9897 | 1.0197 | `FAIL` |
| Second campaign | 269.4 s, 267.9 s | 1.0026 | 1.0244 | `FAIL` |

## Stock noise at TP8

Two four-lifetime stock reference campaigns ran 1.5 hours apart, with two
windows per lifetime. Components are standard deviations of log elapsed time.
A window-1 offset is a fixed slowdown of each lifetime's first window, fitted
either once for all lifetimes or once per campaign; the next section explains
why the choice matters. The TP4 column comes from 14 retained Muse
evaluations, with 90 percent intervals from resampling whole evaluations.

| Component | TP8, this host | TP4 Muse |
|---|---:|---:|
| Window | 1.11% (df 6) with one offset per campaign; 1.77% (df 7, 90% CI 1.25–3.19) with one offset for all lifetimes | 1.32% [0.87, 1.49] |
| Lifetime mean of two windows | 1.45% (df 6, 90% CI 1.00–2.79) | 1.20%, from the two components |
| Boot | 1.22% with one offset per campaign; 0.74% (90% CI 0–2.48) with one offset for all lifetimes | 0.76% [0, 1.08] |

The lifetime-mean spread, which sets each boot's contribution to the grade,
was not lower with one engine on the whole node: 1.45 percent against 1.20.
The TP8 figures rest on eight lifetimes, so their intervals contain the TP4
values.

The second campaign's mean cost was 1.16 percent higher than the first's
(standard error 1.03 percent, 90 percent CI −0.83 to +3.16). Every candidate
graded against one reference shares that reference's error, so a reference
that reads slow raises the false-pass rate of every candidate graded against
it.

## The first window

Every lifetime of the second campaign ran its first window slower than its
second: by 3.7, 3.5, 5.9 and 1.8 percent. The first campaign's lifetimes read
+1.0, −2.0, −1.0 and +1.3 percent, and the second identity control's two
lifetimes +3.4 and −2.4 percent. Token counts are identical across windows
and median first-token times match to within 0.03 s. The 90th-percentile first-token time is about 5
percent higher in every first window of both campaigns, so it does not
separate them. Summed decode time is higher in every first window, by 1.5 to
3.0 percent in the second campaign against 0.45 to 0.96 percent in the first,
so it tracks part of the second campaign's 1.8 to 5.9 percent excess. Kernel-autotune log
counts matched across the first campaign's lifetimes; the second campaign's
were not checked.

GPU telemetry sampled every 5 s over the second identity control's first
lifetime, whose first window ran 3.4 percent slow, showed window 1 against
window 2 at a mean SM clock of 2026 vs 2024 MHz, 53.0 vs 53.2 °C and power-cap
throttling in 10 vs 12 percent of samples, but GPU utilisation of 87 vs 93
percent. The slow window spent more time with the GPUs idle. The cause is not
identified.

The v19 grader has no window-position term. With two windows per boot, its
within-boot jackknife absorbs a first-window offset into that boot's variance,
and an offset that differs between the reference and a candidate shifts the
estimate by half the difference.

## Candidates at TP8

The ten most recent distinct Muse bundles as of 2026-10-25 were started on this
stage or inspected in source, and none executes its optimized path as written
at TP8:

- Two started on this stage and failed at prepare. One accepts only TP4
  attention without DP attention; the other expects 64 of 256 experts per rank
  under EP4.
- Three more whole-forward bundles carry the same TP4 check, and one per-layer
  bundle requires 32 query heads per rank.
- By source inspection, four per-layer bundles would run at TP8, but their
  kernels gated on a group size of four return to stock code; their other
  fused paths are gated differently. One of the four was started on this stage
  and the host became unreachable before it finished building.

Grading these bundles at TP8 would measure different code from what the arena
graded at TP4.

## Grade power

Simulated with ports of the v18 grader at `0141578f` and the v19 grader at
`5e3ead79`, each checked against that revision, with 200,000 to 400,000
trials per cell. The sealed noise is the commission's per-window and per-boot
log-cost standard deviation, 2 percent each on Muse. The true variation is the TP4
Muse components: stock window 1.32% and boot 0.76%, candidate window 1.40% and
boot 1.02%, first-window penalties of 0.74% (stock) and 1.47% (candidate), and
a lane correlation of 0.11 for v18's concurrent pair. A true gain scales every
candidate window, the first included, on top of its first-window penalty. The
TP8 spreads measured above are not used.

| True gain | 0 | +1.5% | +2.25% | +3% |
|---|---:|---:|---:|---:|
| v18: two TP4 lanes, up to four windows, lanes swapped between orientations | 1.8% | 17.2% | 35.2% | 57.3% |
| v19: 4 reference boots and 2 candidate boots, two windows each | 1.8% | 19.6% | 40.6% | 64.7% |

The median required speedup is 1.021 for v19, against 1.0224 at v18's fourth
look and 1.0428 at its third.

Among 110 simulated variants (up to 8 reference boots, up to 6 candidate
boots, up to 4 windows per boot, three look schedules), the one with the
fewest speed-stage node-minutes that passes a true +2.25 percent at least 80
percent of the time, with false passes averaging at most 10 percent over
references, sets each arm's sealed noise to that arm's simulated true noise,
takes three windows per boot, and adds candidate boots, two to four, only
while the estimate sits within one z-unit below the bar. It passes +2.25
percent 81 percent of the time, +1.5 percent 53 percent, and an identity
bundle 6.2 percent, against 1.8 percent for v19. Priced at 15.5-minute
lifetimes with a fifth of a reference per candidate and the bracket, audit and
pristine T after each PASS, a candidate with a true +2.25 percent gain costs
about 93 node-minutes against about 55 for v19; on that basis a variant with 8
reference boots, two windows per boot and two to six candidate boots is
slightly cheaper, at 91. Under the chosen design, an identity bundle passes
more than 20 percent of the time against 7.7 percent of references; the share
rises to 17 percent of references when the gain is defined on lifetime means.
No retained boot has more than two windows, so the independence of a third
window is untested.

Node-minutes per candidate, not counting the v19 reference's 61 minutes,
which every candidate graded against it shares: v18 ends at its third look in
a median 33.1 minutes (12 retained TP4 Muse stages) and at its fourth in 40.6
(2 stages), including its concurrent stock lane, and a PASS adds 11.0 for the
audit, pristine T and publication (measured once). v19 ends after one
candidate lifetime (14.7) on futility or after two (29.5), and a PASS adds
about 28.

## Arena outcomes since replay grading

All 563 submissions on the public dashboard were classified from its
per-submission records on 2026-10-25: GLM 358, DeepSeek-V4.1-Flash 78, Qwen3.6
81, Qwen3.8 20 and Muse Spark 26. Qwen3.6 was never replay-graded, and 266 GLM
rows predate GLM's switch to replay grading on 2026-09-28; DeepSeek, Qwen3.8
and Muse were replay-graded from the start. Of the remaining 216 rows, 158
carry a measured speedup.

| Outcome, replay-graded arenas | Rows |
|---|---:|
| PASS | 111 |
| Attainment veto | 25 |
| Candidate exception | 16 |
| Payment invalid | 16 |
| Statistical non-establishment | 14 |
| Quality, audit or correctness | 12 |
| Not evaluated (infrastructure, operator, baseline closed, no executable delta) | 18 |
| Build, scan or copy | 4 |

- 133 of the 158 measured rows read at or above 1.0225, the reading a true
  2.25 percent gain would produce; 11 read between 1.0 and 1.0225; the median
  is 1.064.
- Twenty-six FAILs read at or above 1.0225: 19 attainment vetoes and 7
  quality or audit FAILs. A statistical FAIL at the final look reads at or
  below its required speedup, 1.0224 to 1.0273 on the arenas sealed at 2
  percent noise and 1.012 to 1.013 on GLM, so a statistical FAIL above 1.0225
  is nearly excluded by construction.
- Twelve DeepSeek rows are invalid measurements but remain in the counts on
  this page: five statistical FAILs from a stage whose node seam covered 3 of
  40 layers, and seven candidate exceptions caused by an image change that
  removed a module the bundles imported.
- An empirical-Bayes fit of each arena's distribution of true gains to the
  158 measured rows, using measured rather than sealed noise, places 0.3 to 0.9
  bundles truly at least 2.25 percent faster among the statistical FAILs,
  including the five invalid rows, and 18.6 to 18.8 among the attainment
  vetoes; the range spans the fitted priors. Of the roughly 123 to 127 such
  bundles not failed for quality, noise alone failed under 1 percent, and
  noise plus the attainment veto 15 to 16 percent.
- The attainment gap is the candidate's attainment minus stock's, in fractions
  of turns; its noise is the standard deviation of that gap across one
  evaluation's windows. Twelve of the 13 Muse attainment vetoes had a mean
  first-token time of 2.13 to 2.19 s against stock's 1.91 to 1.93 s, with gaps
  four to six times Muse's gap noise of 0.008. On GLM and DeepSeek, six of the
  seven vetoed rows above 1.0225 had gaps 1.5 to 1.8 times their arenas' gap
  noise of 0.022 to 0.024.
- Under each arena's measured noise from its two lane orientations, a simpler
  model than the Grade power section's, an identity bundle passes v18 14 to
  17 percent of the time on GLM, sealed at 1 percent window and 0.4 percent
  boot noise, against 1.4 to 4.4 percent on DeepSeek and Muse.
