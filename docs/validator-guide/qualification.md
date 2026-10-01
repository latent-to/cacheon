# Authoritative qualification

Qualification asks a narrow question: does one exact submitted delta improve one frozen
evaluation stack, in one registered arena, at acceptable quality?

The production answer comes from the version-3 qualification protocol executed
by a trusted host controller. Every candidate is measured by separate baseline
and candidate engine processes on two isolated TP lanes under one sealed
physical-lane authority. The two engines replay the same sealed agent workload
concurrently, one paired window at a time. The answer does not come from a local diagnostic
launch, candidate-side self-audit, miner report, or arbitrary evaluator command.

## Identities before execution

Before a candidate runs, the validator binds:

- finalized reservation and hotkey;
- arena service and workload;
- target catalog and exact registered target;
- submitted-delta digest;
- incumbent and candidate `EvaluationStackManifest` digests;
- materialized engine-tree and launch identities;
- model, runtime, topology, native build, seccomp, and worker distribution;
- calibration, resident speed, physical-lane, and slot-audit requirements; and
- selection commitment, private selection secret reference, and candidate order.

For a registered candidate, C is the incumbent stack with exactly one target replaced.
Every other contribution, adapter, fallback, and engine setting is supplied by the
validator.

There are three nested identities to keep straight:

| Identity | What it fixes | Why it matters |
|---|---|---|
| Reservation | Finalized arrival, hotkey, publication, target members, submitted delta | Prevents a later file tree or miner from inheriting the attempt |
| Qualification authority | Frozen source/plan, candidate order, selection commitment, arena/calibration/runtime identities | Prevents the evaluator from changing the experiment after admission |
| Contribution identity | Arena, target, delta, hotkey, and exact incumbent/candidate stack and tree digests | Binds the retained measured contribution |

Paths are not identities. Moving the same publication or evidence store does not change
the content digests, while rebuilding “equivalent” source under new bytes does.

## Speed-policy versions

Retained attempts identify the speed policy that created them:

| Version | Timed reads | Purpose |
|---|---|---|
| v17 | Two to five paired replay windows, split across both lane orientations | New commissions: pooled elapsed serving cost with statistical eligibility |
| v16 | Paired replay windows on one orientation | Retained: fastest complete pass against a fixed margin |

Both are described in [Finite agent replay](#finite-agent-replay). New
commissions seal v17; v16 remains readable only to reproduce its original
evidence.

Versions 8–15 measured batch cells with a B/C/B′ schedule, and versions 1–7
preceded them. Their graders are not in the tree: a speed witness below version
16 does not decode as qualification authority, and a commission that seals no
replay is refused. The dashboard shows retained batch-cell attempts from their
stored JSON — raw lane rates and per-cell delivery times — without regrading them.

The commission seals the policy version, and the worker executes the version the
sealed plan carries, so the plan and the execution substrate cannot disagree.
Every calibrated threshold is the one the provider sealed.

Fresh execution is resident-only: the runner refuses any other speed-evidence
policy at entry, and the constructor default is the resident policy — the only
one a fresh plan can run. A reopen binds the retained evidence's own policy
explicitly, and retained v16 artifacts regrade byte-for-byte without
reinterpretation. Merely changing the policy label does not upgrade old
evidence.

## First-token and delivery timing

Replay requests always stream. The worker sends a first-token and a final-token
boundary for each request; the controller timestamps their arrival, checks their
request identity and token IDs against the final evidence, and retains the
relative host times. Worker-supplied timestamps are never accepted. Missing,
duplicate, stale or inconsistent boundaries are measurement failures, not
candidate speed failures. A sealed commission may also set
`session.measure_phase_latency` to `true` so the engine-conditioning batches
stream the same boundaries; each streamed request must then generate at least
two output tokens.

SGLang's intermediate stream rows may share a growing token-ID list with an
earlier usage-count snapshot. The adapter accepts that documented shape within
the requested budget; final token IDs and final usage must still match the
exact budget. Intermediate usage is never substituted for delivered tokens.

The isolated worker uses SGLang's persistent async engine loop for generation.
It can accept another disclosed request while an earlier request awaits output;
complete binary evidence and streaming boundaries retain their original request
IDs and nonces. Frames remain intact even when responses complete out of order.
Conditioning batches are still submitted one batch at a time. Eager audit
requests also remain serial so their rank receipts retain one request boundary.

The dashboard's replay detail reports mean and P95 TTFT, the median per-user
decode rate and unsuccessful requests from the retained turn records. TTFT
includes queueing, tokenization, prefill, sampling and delivery. These are
serving latency measurements, not isolated GPU phase durations, and they do
not change a qualification verdict.

Retained batch-cell attempts keep their per-cell `cells` table, recomputed from
the host times in their retained timed windows and grouped by input tokens,
output tokens and request concurrency:

| Field | Definition |
|---|---|
| `mean_ttft_seconds` | Mean time from batch dispatch to each prompt's first delivered token |
| `mean_tpot_seconds` | Mean `(last delivery − first delivery) / (output tokens − 1)` across prompts |
| `end_to_end_output_tokens_per_second` | Cell output tokens divided by its timed batch spans |
| `timed_batches` | Number of retained timed batches for this cell |

## Current qualification timeline

The version-3 protocol binds two non-overlapping physical TP lanes, A and B,
with equivalent topology, separate runtime namespaces, lane-specific NUMA policy,
exact workload, and a total qualification budget. Policy 17 splits its windows
between two orientations. The first boots the incumbent on lane B and the
candidate on lane A; both condition and flush their caches, then replay the same
sealed slice concurrently, one paired window at a time. Both engines then close,
and fresh engines boot on the opposite lanes for the second orientation, so a
stable lane factor cancels out of the score. Both lanes publish native builds to
one shared store under the commissioned root, so the swapped orientation reuses
the first orientation's builds.

Every replay record carries the engine-observed prompt token count and the
controller's host timing, and the regrade rejects a window whose records differ
from its completed host requests. A nominal host-side token count is never
authority, and a count mismatch is an infrastructure fault — it can hold the
leg, never mint a candidate verdict.

```mermaid
sequenceDiagram
    participant H as Trusted host
    participant LA as Lane A
    participant LB as Lane B
    participant A as Audit-only role
    participant T as Pristine reference
    H->>LB: orientation 1, boot incumbent, condition, flush
    H->>LA: orientation 1, boot candidate, condition, flush
    par each paired window
        H->>LB: replay the sealed slice
        H->>LA: replay the sealed slice
    end
    H->>H: grade, a sealed futility margin may FAIL here
    H->>H: close both engines
    H->>LA: orientation 2, boot incumbent, condition, flush
    H->>LB: orientation 2, boot candidate, condition, flush
    par each paired window
        H->>LA: replay the sealed slice
        H->>LB: replay the sealed slice
    end
    H->>H: grade at each sealed look
    H->>LB: on PASS, close the candidate engine
    H->>H: bind entropy to lane B quiescence, select quality prompts
    H->>LA: generate the selected incumbent controls
    H->>H: prove both speed executors quiescent
    H->>A: run sealed audit-only plan
    A-->>H: exact slot × rank witness
    H->>H: reveal the selection again; entropy must match
    H->>T: run candidate-free quality authority
    T-->>H: pristine quality evidence
    H->>H: prove final quiescence and regrade
```

The candidate cannot request extra windows; the sealed stopping rule decides
how many run. The retained witness records every window's turn records, lane
identities, operational timing, and the stage and total budgets.

Speed is graded before the expensive audit and pristine-reference stages. An ordinary
speed non-PASS emits a durable stage-exit and does not run audit or T. A separately bound
calibration-observation disposition may continue after a speed failure to collect
diagnostic audit and T evidence, but it cannot crown the candidate.

A speed FAIL names what the round proved, graded with the verdict itself rather than
derived from the bare decision. A bar of 1 + u can only call a candidate slower once
its measured speedup falls below the mirrored bound 1 − u; that failure is
`candidate_slower`. A miss inside the band is `speed_threshold_not_met`: the bar was
not cleared, and the candidate was not measurably slower either. A candidate that
clears the bar but fails the service-attainment gate is `service_contract_not_met`.
Reports settled before this split carry the retained coarse code `speed_regression`,
which remains valid for them and is never recorded on a new verdict.

The audit-only role is distinct from both timed lanes. Trusted-host grading imports no
PyTorch and returns `PASS`, `FAIL` (`slot_audit_failed`: compared calls show a wrong
kernel), or `NO_DECISION` (`audit_not_covered`: the audit compared too little to grade
the kernel, which requeues the bundle instead of failing it). The rule is in
[Audit outcomes](fidelity.md#audit-outcomes). Live floating-point facts are
canonicalized into stable decimal strings before they enter the durable witness.
The audit policy names the selected target's slots. A composed engine also runs
the incumbent contributions; their audit receipts are excluded from the selected
delta's grade, while execution coverage still requires every active slot on every rank.

MoE audits run stock on the original inputs before invoking the candidate and
retain a copy of the stock outputs. This preserves the input-address binding of
upstream FP4 outputs while preventing candidate writes from changing the reference.
A failed stock call is a baseline refusal: it adds no coverage and is never a
successful comparison, and it is not evidence against the candidate.
When MoE consumes gathered DP projection outputs, its audit uses the same original
per-rank token counts as the projection audit. Unused padding is excluded from
comparison; entirely idle calls and tuner batches without token counts add no
audit coverage. Candidate execution and returned buffers are unchanged.
The slot's numerical comparison and acceptance thresholds are unchanged.

The eager audit preserves the charged workload's prompt batches, concurrency,
and per-batch input-token expectations, including mixed-length workloads.
It runs one warmup batch, then the first charged batch of each distinct concurrency
and request shape, in charged order. Generation
length remains bounded by the audit policy. Reducing these batches to single
prompts can select a different DP padding or dispatch path and leave a serving
collective completely unaudited. The host verifies the exact derivation; missing
slot/rank coverage never passes the audit and grades `NO_DECISION`. Semantic quality remains the separate
pristine T reference's responsibility.

Reference errors and incomplete audit coverage produce an infrastructure HOLD,
preserving completed speed evidence. Only complete, attributable numerical
violations produce a candidate FAIL. The validator may correct its own audit
failure with `qualification_recovery.authorize_recovery`: an append-only child
continuation retains the original speed and historical disposition, binds the
corrected audit execution, and runs only missing pristine T. A passed audit can
also be reused when only T is missing. The operator must authenticate the original
request and establish executor quiescence before authorizing recovery.

Recovery charges completed speed, audit and reference execution against the
existing qualification budget; downtime between durable stages is excluded.
Original timestamps, speed policy, numerical thresholds and bundle identity stay
intact. Conflicting evidence, exhausted time or a clock reset stop before launch. Recovery grants no PASS or reward by itself: the
normal grader, evidence import and settlement still consume the completed result.

T remains untimed and candidate-free. The host owns role assignment, monotonic clocks,
token numerators, conditioning windows, absolute deadlines, device observations, audit
grading, selection entropy, and teardown. Candidate wall-clock reports and aggregate
throughput are ignored.

## Cohorts and selection

A service may freeze one incumbent and qualify a chain-ordered cohort `C1..Ck`, sharing
one pristine reference lifetime where the policy permits. This is an
operational optimization, not a semantic relaxation:

- every C remains one exact marginal delta;
- candidate ordering is sealed in finalized cohort/plan order before entropy is observed;
- post-commit entropy selects the hidden prompt/task work, and the selection receipt binds
  that choice without reordering candidates;
- drift outside calibration produces `NO_DECISION`; and
- retained evidence must still support each candidate independently.

The contract does not require cold model loads for every timed read. It requires resident
lane identity, the sealed window schedule, audit authority, and pristine-reference
authority to remain causally and cryptographically separable.

For registered cohorts, a recognized cohort-level factory, runner, raw-speed,
outer-session, or OCI-backend failure that the intake boundary normalizes into a
qualification failure product produces `NO_DECISION` for every affected reservation and a
persisted bisection plan. Subsequent passes halve the cohort to isolate a poisoning or
resource-sensitive candidate in logarithmic retry groups. A per-candidate `NO_DECISION`
after a complete shared attempt is requeued individually. Other provider, controller, or
evidence-publication exceptions abort the pass and recover through controller
restart/hold handling rather than this typed batch product. Neither mechanism changes
finalized arrival order.

A typed candidate-worker error is attributable only when rank receipts bind the exact
registered target arm and identity. It publishes a terminal candidate failure product
instead of entering the generic retry path. Baseline-lane, shared-controller, audit/T,
multi-candidate, or untyped worker failures are infrastructure authority failures; they
are never assigned to a convenient candidate.

## Gates and three-way decisions

Qualification reopens and grades several evidence products:

1. **Execution:** required roles completed under the expected launch and device state,
   and on a graphs-on run every claimed slot on every rank completed inside a CUDA-graph
   capture, as recorded by the dispatcher.
2. **Speed:** C's paired replay cost beats the incumbent's under the sealed
   eligibility bound, and C passes the service-attainment gate.
3. **Audit:** the sealed audit-only plan has complete exact slot × rank authority.
4. **Quality:** pristine T validates sealed trajectories and hidden work under the
   registered calibration.
5. **Whole-stack identity:** the report still describes the frozen incumbent and exact
   candidate stack.

There is no separate offline graph stage. The graph proof is those three products read
together: the captured completions of the timed run show the candidate served from inside
the graphs, the audit checks each slot's output against the stock computation, and the
quality gate catches a candidate that is captured but replays a stale answer. Reports
before `cacheon.qualification.candidate-report.v3` carried a graph grade from a separate
capture-and-replay probe; this source does not produce or reopen them.

The result is one of:

- `PASS` — all required evidence is complete and green;
- `FAIL` — attributable candidate evidence violates a registered requirement; or
- `NO_DECISION` — infrastructure, drift, missing authority, or incomplete evidence makes
  a fair result impossible.

`NO_DECISION` is retryable under bounded policy. It is not a loss and must not be
converted to zero reward for convenience.

The evidence-to-verdict mapping is fail closed:

| Observation | Decision | Example |
|---|---|---|
| Complete, bound, and green across every required product | `PASS` | C clears calibrated speed bar; audit and pristine quality pass |
| Complete attributable violation of a frozen candidate requirement | `FAIL` | Wrong output, never invoked inside a capture, no speed gain established by the last sealed window, or measured quality regression |
| Authority incomplete, stale, unreopenable, timed out, or infrastructurally invalid | `NO_DECISION` | Missing evidence bytes, controller/worker failure |

An unexpected exception is not evidence of candidate guilt. The intake projection turns
recognized plan, runner, and raw-speed authority failures into typed failure products and
retry plans. Other controller exceptions are contained by the pass loop and recovered
conservatively on restart.

### Execution evidence

Registration is not execution: a bundle can load, register its slot, capture, and
then never dispatch, and such a run still produces a complete speed number. The
candidate engine therefore proves execution from dispatcher receipts. Every
scheduler rank must activate the same registered slot set at launch and complete
every registered slot during the run. On a graphs-on run, only completions the
dispatcher recorded inside a CUDA-graph capture count.

| Observation | Decision |
|---|---|
| Ranks do not all activate the same slot set | Launch fails; infrastructure HOLD unless rank receipts record the candidate's own load or invocation failure |
| No completion at all, or completions only outside a capture | `FAIL` (`candidate_never_executed`) under the one-target attribution rule above |
| Some ranks or slots complete and others do not | Infrastructure HOLD / non-verdict |
| Every rank completes every registered slot | Speed evidence may be graded |

In the eager audit role, missing execution grades `NO_DECISION`
(`audit_not_covered`) instead. Partial evidence may not be converted into
candidate PASS or FAIL. The durable store represents a non-verdict as a
reservation HOLD with no candidate decision, which is semantically
`NO_DECISION` without reviving the retired literal decision field.

A worker HOLD retains the exception type and a bounded cause chain in the
existing failure fields, shown in the dashboard's evaluation forensics. The three
graph HOLD reasons remain readable so durable rows written by earlier source
reopen; no worker on this source produces them.

This evidence is written from inside the candidate's own process. It closes
accidental non-invocation as a verdict condition, but it is not proof against a
deliberate forger; complete-engine isolation and external qualification remain
the boundary.

## Qualification acceptance

One complete audited PASS becomes `qualified` and creates a `SettlementCandidate`
in the same intake transaction. Settlement reopens the exact retained attempt,
report, disposition, audit, and quality evidence. No second qualification
is scheduled.

New single-run candidates use `cacheon.settlement.candidate.v5` and encode only
`primary`. Their existing evidence record uses `cacheon.settlement.evidence.v2`
and omits reproduction fields. Historical paired records keep their exact wire
bytes, distinct-authority and lane-swap checks, and lower accepted speedup.
A complete primary PASS left in `reproduction_pending` by an older controller
is accepted on restart after its evidence reopens; GPU work is not repeated.

## Reopen and regrade

Full regrade requires more than the persisted attempt reference. The caller must
reconstruct the exact `CausalQualificationInput`, including prepared plans, candidate
authorities, calibration references, runtime policy, reference
authority, and commitment. SQLite's authority manifest and
`CohortQualificationAttempt` bind identities but do not embed that complete private
provider/plan object. Settlement restart authenticates attempt bytes and stored PASS
dispositions; it does not invoke the full causal regrader.

The final report is derived from the serialized attempt, referenced graph/quality
artifacts, and calibration manifests. Reopen can regrade graph and raw quality evidence.
Speed regrading uses the retained `ResidentSpeedWitness`, which retains every
paired window's turn records, physical-lane authority, operational timings, and
budget. A witness below version 16 is batch-cell or MiniMax-M3 history and is
refused rather than decoded. Regrading recomputes costs and the frozen
decision from those typed facts; it does not reconstruct them from raw session
frames. A summary JSON line without these products is not authority.

See [Evidence and replay](../security/evidence.md) for retention and audit requirements.

An authoritative attempt is not one headline. Durable authority includes the authority
manifest; selected plan and commitment/entropy/selection receipts; referenced graph
evidence; the aggregate speed witness; the pristine-T execution
witness and raw quality artifact/binding; per-candidate reports; and the enclosing attempt
artifact. Settlement keeps every accepted attempt reference.

The live outer session validates richer per-request protocol frames, lifecycle order,
device state, and cleanup before constructing that attempt. Those raw frames and per-arm
device samples are not serialized into `CohortQualificationAttempt`. The witness's turn
records retain per-request host timing, not the engine frames behind them.

Reopening verifies hashes and expected bindings before grading. If the attempt artifact
reopens but a referenced graph, calibration, or raw quality product does not, authority is
still incomplete. Operators must retain every referenced evidence-store object and test
restores, not merely archive the final report digest. If policy requires raw engine-frame
replay, the attempt schema must first be extended to retain and bind those products.

## Qualification incident handling

| Incident | Required disposition |
|---|---|
| Candidate engine exceeds deadline or violates protocol with attributable evidence | Grade under the frozen requirement; `FAIL` only when attribution is complete |
| Typed worker failure binds one exact candidate arm | Contain that candidate; retain its attributable outcome and preserve unaffected cohort results |
| Recognized worker, Docker, GPU, driver, plan, runner, or raw-speed authority failure | HOLD with the original evidence; automatic retry requires authenticated proof that resident execution never began |
| Evidence-store publication failure | Abort the pass; recovery holds an interrupted `qualifying` row as `controller_restart_during_qualifying` rather than manufacturing a typed `NO_DECISION` |
| Speed gain not established by the last sealed window | `FAIL`; do not add windows or tune the bar after seeing C |
| Either resident speed executor survives past its quiescence proof | Abort authority; never launch audit or T into the contaminated lifetime |
| Audit role misses a slot/rank, reports a violation, or cannot reopen | `FAIL` only for a complete attributable violation; otherwise `NO_DECISION`; never substitute candidate-side audit output |
| T identity/session mismatch | `NO_DECISION`; T cannot be replaced with an incumbent read or a candidate-side audit |
| One member poisons a registered cohort | Preserve cohort failure digest and execute the stored bisection groups |
| Accepted PASS evidence root lost | No settlement; restore exact bytes or hold |
| Historical reproduction differs in contribution identity or reuses an independence digest | Reject the retained pair; do not reinterpret its original contract |

Never rerun only the favorable arm, splice evidence from different authorities, or lower
a threshold after seeing the outcome. A fresh attempt must be a complete, newly bound
qualification under the registered policy.

## Standing two-lane composition and remote products

The commissioned deployment surface for standing mainnet qualification lives in
`eval/b300_qualification_deployment.py` and `eval/b300_registered_qualification.py`.
Registered per-target profile authorities are sealed ahead of time; at plan time the
deployment layer independently re-derives the profile authority for the finalized
reservation and rejects a plan whose authority, marginal arm, secret, pristine
binding, or resident lane executors differ from the sealed construction inputs. The
two lanes are disjoint, equally sized device sets on the commissioned node, each at
the arena's TP width, and the physical lane pair, device identities, and role swap
are validated against the READY receipt before any engine work.

Qualification is the only operation of remote-evaluation protocol v4. It returns
a sealed `RemoteQualificationProduct` (schema version 2): size-bounded evidence artifacts are
rehashed on capture and on import, and the coordinator's durable
`commit_remote_qualification_result` pins the incumbent stack and tree identity per
arena on first commit and rejects any later mismatch atomically. Transport,
authentication, and identity-check failures release the durable lease as
infrastructure outcomes; they are never converted into a candidate verdict.

The persistent production consumer is
`eval/b300_remote_worker_adapter.py`, whose only mode is `--serve`. It loads one
digest-exact qualification-capabilities factory, calls
`build_commissioned_b300_qualification_service`, and derives the worker and the
primary and reproduction commissions from the same registered READY authority.
Without those capabilities or an injected commission it refuses to start.
Each authenticated qualification request resolves its leased cohort,
derives a candidate-local `B300RemoteQualificationAdapter`, and runs through
`B300MainnetWorker.run_remote_qualification`.

Registered B300 qualification seals only the canonical retained-support policy
digest derived by `retained_support_policy_digest()`. A stale support-policy
digest fails commissioning before GPU execution; retained quality validation
checks the same binding again after execution.

The commission measures the candidate against the durable incumbent stack the
capabilities factory declares (`incumbent_entries`, resolved through the same
closed source resolver); at genesis the declaration is empty and the baseline
is the stock tree. The two-process schedule's baseline process boots the
materialized incumbent tree, and the worker executes the sealed version the
plan carries, so plan and execution cannot disagree. Pristine T stays anchored
to the empty stock stack regardless of the declared incumbent, so the untimed
audit reference never inherits crowned contributions. A declaration that does not reproduce the
durable stack identity fails closed at the dispatcher's incumbent pin and at
the durable commit.

`eval/crossover_runtime.py` owns the resident plan and evidence types, and
`eval/goodput_runtime.py` runs and grades the paired replay. Deployment-private
capability bytes still supply sealed identities and must match their configured
source digest; they do not create a second evaluator.

## Nonclaims

- Passing proves the registered arena/workload and policy, not universal model quality or
  performance.
- T is an independent semantic reference, not proof that the reference implementation is
  bug-free.
- Isolation and protocol checks reduce candidate influence; they are not a formal proof
  against GPU, driver, kernel, or container-runtime compromise.
- A crown records measurement and attribution. It does not satisfy integration, license,
  provenance, maintainability, or release review.
- Deployment must supply a reviewed production provider that constructs this work for the
  registered arena. Structural qualification fixtures can test the authority path but cannot
  establish an empirical GPU crown or production calibration.

Next: [Settlement and weights](settlement-and-weights.md).

## Source anchors

- [Qualification evidence model](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification.py)
- [Causal qualification runner](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification_runner.py)
- [Resident crossover runtime](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/crossover_runtime.py)
- [Qualification deployment composition](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/b300_qualification_deployment.py)
- [Registered qualification profiles](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/b300_registered_qualification.py)
- [Remote qualification adapter](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/b300_remote_qualification_adapter.py)
- [Torch-free audit gate](https://github.com/latent-to/cacheon/blob/main/cacheon/audit_gate.py)
- [Finalized-intake projection](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification_intake.py)
- [Pristine reference session](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/oci_reference_session.py)
- [Qualification tests](https://github.com/latent-to/cacheon/blob/main/tests/test_qualification_runner.py)

The isolated worker and controller share one request-addressed pipe exchange.
Ordinary generation requests may overlap and finish out of order; eager audits
remain serial. A request can carry either text prompts or canonical token IDs
from the validator's chat template, plus an explicit attention-DP rank for
session affinity. Token IDs avoid re-tokenizing rendered chat text with an extra
BOS token. First/final token events are timed by the controller and checked
against the existing final binary evidence; one-token requests still have TTFT.
No HTTP listener or candidate network access is added inside the container.

## Finite agent replay

`SessionExecutionPlan.replay` selects `AgentReplayPlan` for a finite load
window through the ordinary OCI executor. The plan seals exactly one load, the
operating load of the paired fixed-work comparison.
Install the controller's `replay` extra
and supply a separate AIPerf 0.13.0 virtual environment executable, the local model tokenizer, a
sealed slice manifest, loads, output directory and service contract. Replay uses
the complete sealed engine template, including its declared
`engine_config.engine_kwargs.context_length`. Engine conditioning
runs first with at most the declared session count per conditioning batch and
16 output tokens per request. The replay plan retains only those conditioning
rows; AIPerf supplies the measured work. AIPerf starts exactly the declared number
of root sessions and drains their turns and children; there is no duration cut
or request-count truncation. `--ignore-trace-delays` removes recorded user and
tool waiting; the bridge releases ready requests in lockstep rounds.
Turn order and child joins remain intact. The loopback chat adapter passes canonical input IDs and
sticky DP ranks, fixed at each session's first turn in release order, through the isolated-worker pipes.

The slice loader verifies the named files in sealed order. A per-load directory
contains only the first `load` files. Output budgets come from those traces;
context overflow or an incomplete export fails the read. The client preserves
`X-Request-ID`; the collector joins it to source trace and child coordinates, numbers each root's
turns by release round and conversation coordinate (never by client send time), then requires the
exact main/inner turn counts for every root. All scoring timestamps use host nanoseconds in the
same epoch clock domain, with the monotonic-to-epoch anchor retained in `clock.json`.

Each engine stays loaded across the windows of its orientation. Before each load, the controller
requires an acknowledged SGLang cache flush covering device radix state and
the HiCache host pool. Failure stops the window. Each load's fresh output
directory retains the AIPerf command, log and raw export,
`bridge.jsonl` with canonical inputs, actual output IDs and host timing,
`turns.jsonl` in the service-capacity record format, and `read.json` with fixed
work rate and attainment. `window.json` retains the workload identity and the
read. The existing engine-session evidence also carries
the typed `LoadRead` records, so continuation retains them with the token evidence.
A window does not by itself provide the paired capacity comparison, quality audit or
authoritative qualification result.

Policy v17 retains the lockstep replay schedule: requests are released together
in session-key order after the preceding round drains. The cold opening round
establishes the declared warm-cache operating point before measurement.
The measured cost spans the first warm release through the last warm completion,
including handoffs and tails. Fixed workload value and GPU allocation make its
inverse monotone in profit for this workload and pricing scenario. Native output
generation, MTP acceptance costs, and prefix-cache behavior remain in this cost.
The first opening group must fill within 120 seconds of its last arrival.
V16 reads retain lockstep rounds and inverse mean request latency;
that quantity is not wall-time throughput and is not the v17 score.
The driver uses the executable's sibling Python interpreter to call AIPerf's
single-run API. The sealed slice digest supplies its benchmark identity, keeping
cache-buster tokens identical between arms and windows. Each
client retains separate memory-mapped dataset files under its output directory;
concurrent clients share no writable dataset state. Each client's ZMQ IPC
sockets live in a short private temporary directory under `/tmp`, because Unix
socket paths are limited to 107 bytes and the spool root may be deep. The client requires exactly
AIPerf 0.13.0 because it consumes that version's API.

The driver runs AIPerf at concurrency `load` with `--no-fixed-schedule` and
without the AgentX scenario; the bridge, not the client, sets the arrival
schedule. The AgentX trajectory warmup is designed for
recorded timestamps; stripping those timestamps can omit intended turns.
Omit `--warmup-request-count` for no client warmup; version 0.13.0 rejects an
explicit zero. The engine still receives its ordinary conditioning requests.

Replay qualification is commissioned with `session.replay`: absolute `manifest_path`,
`aiperf_binary` and `tokenizer_path`, the manifest's computed `slice_digest`, a
single operating `load`, and the number of paired `windows`. Optional
`max_work_seconds` bounds replay work, split across orientations in proportion to
their window budgets; expiry aborts, never grades incomplete work. Engine startup
retains its separate OCI initialization deadline. `resident_speed.goodput`
seals the service `contract`
(`decode_floor_tps`, `ttft_bound_s`, `attainment`), `required` ratio, paired
`null_noise`, fixed `attainment_tolerance`, and its calibrated one-sided
`attainment_margin`. These values use canonical decimal strings. The
window-scatter, conditioning-slowdown and minimum-window fields of the retired
batch-cell policies stay sealed as zero, because the policy digest binds them. V17 seals `required=1`, `error_rate=0.01`, paired per-window log-cost
standard deviation `null_noise`, and paired boot standard deviation `boot_noise`.
An optional `futility_margin` fails a stage whose complete first orientation
reads more than that fraction slower, without booting the swapped orientation.
At least one uncertainty component must be positive. These are calibrated
uncertainty bounds, not a desired detection threshold. Greedy sampling and zero
rollout top-k width are required. Coding replay uses a teacher-only quality
profile (`hidden_tasks_per_prompt=0`, `hidden_tasks_required=false`), not numeric
answer tasks. Its reference token maximum equals the largest output budget in
the selected slice; conditioning generates 16 tokens.

New replay commissions require V17 statistical eligibility. The commissioner
rejects a legacy fixed-threshold goodput declaration before staging a run;
V16 remains readable only to reproduce its original evidence. A noise estimate
for summed request latency cannot calibrate elapsed serving cost. Calibration
must cover the commissioned workload, arrivals, engine state and lane schedule.
An error budget is conditional on those noise bounds; merely setting 1% does
not establish a measured false-positive rate.

Both OCI engines condition and flush before concurrent reads. V17 permits two
to five paired windows; the first orientation runs half of them, rounded down,
and the swapped orientation the rest. Costs
pool within each orientation; the score is the geometric mean of the two pooled
cost ratios. This cancels a stable multiplicative lane factor without dropping
slow windows. The standard error retains paired boot variance and the larger of
the calibrated window variance or observed whole-window jackknife variance.
Requests sharing a window are not treated as independent replications.

Eligibility requires a positive one-sided log-gain bound. At most four looks
share the 1% error budget: 5%, 5%, 10%, and 80% of that budget, with unused looks
unspent. Both orientations must exist before a PASS; a futility FAIL keeps one.
The last sealed window yields `PASS` or `FAIL`, never `NO_DECISION`. This normal-model guarantee
depends on the sealed noise bounds and independent boot contrasts; it does not
establish runtime resolution by itself. A fixed 1% gain floor is absent. Reward
credit uses the point estimate, not a confidence-bound haircut; a later V17 PASS
is paid only when that estimate beats the best earlier rewarded PASS against the
same arena and incumbent stack by 1.5%. Regrading checks
the exact stopping point, all engine identities, host timings, and controls.
V16 retains its fixed margin on one orientation. Each paired window is one complete
pass over the sealed basket, and the V16 score is the candidate's fastest pass over
the incumbent's fastest pass. Every turn keeps its cost weight because a pass is
never split. Both engines drop their slower passes alike, so the first pass after
an engine load, whose allocator and cache warm-up differs between engines, does not
decide the verdict, while a slowdown that recurs in every pass remains a cost. The
attainment gate still applies to every window.

The dashboard reports V17 replay measurements as elapsed serving seconds and
**completed warm turns per second**, with lane-balanced rates matching the
scorer. V16 retains seconds per warm turn (lower is better).
Both include each arm's per-window latency and service attainment, the
operating load, completed window count, workload identity and required gain.
V16 marks each engine's fastest complete pass and uses those pass latencies in
the baseline and candidate summaries. The detail includes mean and P95 TTFT,
median per-user decode rate, unsuccessful request count and speed-stage duration.
The attempt's retained workload and policy identify its regime, regardless of
submission date. V17 uses pooled elapsed cost across lane orientations. Retained
batch-cell evaluations keep their token-throughput units and are shown as stored,
without a regrade.

The economic interpretation assumes identical billable work, fixed GPU
allocation and cost, and demand for the measured capacity. At fixed basket
value `V`, measured duration `T` and GPU count `g`, revenue capacity per GPU-hour
is `3600 V / (g T)`. Subtracting the same hourly GPU cost preserves its ordering.
A positive revenue-capacity gain understates percentage profit gain only when
the incumbent's profit is positive. It does not guarantee realized profit when
demand, prices, cache-billing rules or the traffic schedule change. Lockstep
replay characterizes its declared arrival schedule, not arbitrary continuous
traffic. Service attainment is a relative non-inferiority check; the retained
`attainment` target does not certify an absolute SLA or change billable value.

On a speed PASS, the swapped orientation's candidate engine closes and lane B
proves quiescence. The entropy provider binds selection entropy to that
candidate-lane quiescence receipt and selects source occurrences from the swapped
orientation's incumbent and candidate trajectories. The incumbent, still loaded on
lane A, generates only those selected controls. Canonical input digests bind
all three rollouts, and the speed continuation retains the controls before the
separate eager audit and pristine T run. Reopening selection after the audit must
return the same retained entropy. Source identities use trace/outer/inner indices,
so parallel child arrival order does not change which turn T evaluates.

Before it judges the candidate, the quality gate bounds the incumbent's drift from
pristine stock per metric; a bound outside the calibrated envelope is `NO_DECISION`.
A rollout's worst token is an extreme value, so from four scored prompts on the
worst-token drift bound may set aside its single most extreme prompt, and never
rises by doing so. The mean and tail-rate bounds, and every candidate regression
bound, use all scored prompts.
