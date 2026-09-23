# Arena service

An arena service binds an immutable proposal publication to a validator-owned
runtime, workload, capacity policy and qualification authority. A submission
selects a registered arena and target; it cannot select Python providers, shell
commands, model paths, workloads or qualification factories.

## Manifest identity

`ArenaServiceManifest` binds:

| Part | Bound facts |
|---|---|
| Runtime | Arena ID, runtime/base engine/overlay/worker identities, model revision/manifest/content, architecture, topology, GPU count and TP size |
| Workload | Prompt corpus, seed scheme, serving cells and timed reads |
| Capacity | Queue depth/age, concurrent screens and qualifications, cohort size and retry budgets |
| Screens | Ordered non-crownable stages and deadlines |
| Authorities | Reviewed provider digest, qualification policy and closed targets |

The manifest digest identifies one immutable commission. Changing the model,
image, runtime, workload, hardware allocation or policy creates new evidence
identity. A provider must match the declared digest and typed interface;
deployment review establishes that its installed bytes match that declaration.
This check is not remote attestation.

## Scored workload

Each cell declares engine-observed input tokens, output budget, concurrency and
timed reads. Sealed prompt batches must match these cells. Context length,
admission width, graph/eager mode, seam bindings and watchdog settings derive
from that authority. Qualification rejects inconsistent sessions before launch.

Mixed-cell scoring uses timed output tokens over the complete host-observed
mixture makespan. Optional `resident_speed.prefill_lane` commissions v15 and
appends a one-token prompt pass after B′; see
[Qualification](qualification.md#prefill-lane-v12). Evidence retains the version
and arithmetic under which it was measured.

The sealed input selects `registered_targets`, `model_profile_key` and engine
configuration. Commissioning derives `closed_targets` from the shared catalog,
so another arena adding a target cannot silently change this arena's policy.
Unmeasured submissions for closed targets leave as NO_DECISION and release
payment before dispatch; see [Why submissions fail](../miner-guide/why-submissions-fail.md).

## Non-crownable screens

Stages run in order: `static`, `build`, `abi`, `graph`, `abbreviated_serving`.
Each emits bounded typed evidence: `pass`, `fail`, or `no_decision`. Receipts
must be a contiguous prefix; a provider cannot skip a stage or exceed its
registered timeout while retaining a pass. Preserve the bytes named by each
evidence digest.

A resident screen keeps stock loaded and executes stock/candidate/stock swaps.
Each candidate must reach its registered dispatch, recapture/replay graphs and
prove stock restoration on every rank. Reads bind to the current generation;
stock-canary drift withdraws the lane. These measurements only route work and
cannot crown, settle or authorize rewards.

Standing-resident deployments defer separate ABI/graph lifetimes to this
all-rank swap/read, preserving their fixed stage positions. Full qualification
still supplies the independent audit, the pristine quality gate, and the
captured completions of the timed run, which together are the crownable graph
proof.
Non-swappable AOT artifacts, dependency patches, native rebuilds and setup
hooks receive an explicit routing waiver into full qualification, never a
performance pass.

## Admission and capacity

The controller supplies a durable queue snapshot. Capacity comes from measured
operational budgets; finalized chain order remains authoritative.

| Decision | Effect |
|---|---|
| `admit` | Claims a screen or qualified cohort within available capacity |
| `queue` | Leaves work in its durable lane and stops selection this pass |
| `hold` | Retains work when age/depth/cohort limits require intervention |

A first screen NO_DECISION retries. An inconclusive final serving canary routes
to isolated qualification. Candidate-caused screen failure rejects; provider,
baseline, teardown and incomplete-evidence failures remain infrastructure
failures. Qualification HOLD has an explicit recovery path. Monitor held rows
and preserve evidence; deleting them is not recovery.

## Boundary types

`ArenaCandidateBinding` binds publication, reservation and screen attempt.
`ArenaScreenReceipt` retains routing evidence. `ArenaQualificationRequest` and
`ArenaQualificationWork` bind authoritative qualification. `ArenaServiceRegistry`
resolves exact logical arena IDs.

## Provider interface

`ArenaServiceProvider.run_screen(manifest, stage, candidate)` returns the
requested stage's evidence. `build_qualification(request, state=None)` preserves
promoted reservation order and supplies the frozen plan factory, isolated
executor, post-commit entropy, hidden judge and absolute deadline. The controller
checks these identities before importing retained outcomes.

`B300ArenaServiceProvider` is the shared implementation. Historical names and
signed digest domains remain stable; hardware comes from commissioned inputs.
Private capabilities remain validator-owned.

## Deployment composition

The screen materializer, qualification commissioner and worker consume one
sealed definition per arena. READY's `gpu.inventory` describes the host;
`lane.devices` selects the screen lane and `lane.tensor_parallel_size` must
equal its width. Optional `lane.baseline_devices` selects an equal-width,
disjoint baseline lane of the same GPU model and memory capacity. If omitted,
the complete host complement must have exactly that width.

TP1 therefore needs two GPUs, including an explicit pair within an eight-GPU
host. TP4 uses two four-GPU lanes. Spare devices are never allocated implicitly.
Pass the materializer's output as `--commissioned-root` to the remote adapter
(or `commissioned_root` to its Python builders). Omission preserves the
historical GLM path.

Optional sealed `resources.runtime` and `resources.prebuild` objects override
existing OCI capacity fields: `cpu_millis`, `memory_bytes`, `pids_limit`,
`nofile_limit`, `tmpfs_bytes`, `shm_bytes`, `cache_bytes`, `cache_inodes`,
`stage_bytes`, and `stage_inodes`, where supported by the respective policy.
UID/GID, executable and deadlines retain their existing authorities. Absent
resource objects preserve prior policy. Both authority and measurement inputs
must agree; screen, graph and qualification executors consume the same values.

Commissioning requires faithful, broken and infrastructure controls through the
production entrypoint, with graphs, audit/T roles, retained evidence and restart
recovery. CPU composition and compatibility checks do not establish GPU success.

## Registry and CLI boundary

Run one finalized `chain-validate --intake-only` process with a shared store.
Each arena runs the same dispatcher/supervisor with its own manifest, READY,
registration, worker, spool and evidence paths. Exactly one supervisor enables
settlement and weights. Settlement follows each arena's completed arrival prefix;
later active evaluations do not block an earlier completed result. Weights
aggregate eligible arena claims.
Additional supervisors set `enable_settlement` and `enable_weights` to `false`.

New arenas require `competition.arena` in the hash-bound bundle. Every dispatcher
config states `accept_legacy_bundles` (an omitted key is refused): `true` for the
default arena, `false` for the others. The default alias binds once to the
verified runtime's `arena_id` and cannot be claimed by another arena. Explicit
selectors equal to that ID resolve to the historical namespace. Routing becomes
known at publication; fetch and pre-publication admission remain shared.

Queues, baselines, target admission, lineage and qualification recovery are
scoped to the logical arena. Service epochs retain that arena's lineage.
Commissioning Qwen cannot retire GLM's baseline. Two arenas can qualify
concurrently. Within one arena, commission disjoint pairs with the same model,
runtime, workload, incumbent and GPU execution policy. Physical GPU addresses
belong to each worker's READY, registration and launch binding; equivalent pairs
share the arena service identity. Calibration and complete B/C/B′, audit and T
evidence remain specific to each job and physical pair.

Run one supervisor and relay per pair with distinct `owner`, registration, spool,
evidence and runtime paths. Each supervisor uses the existing screen and
qualification entrypoints, with `enable_settlement=false` and
`enable_weights=false`. A separate supervisor runs economics with
`enable_screen=false` and `enable_qualification=false`, so long evaluations do
not stop finalization. `enable_screen` is optional and defaults to `true`.
The sealed service's `max_active_screens` and `max_active_qualifications` bound
transactional claims. One worker can hold only one active job, and one reservation
can belong to only one active lease. Recovery reopens by worker owner.

To free an allocation, stop its supervisor with SIGTERM or SIGINT. It stops
claiming new work, continues renewing and importing any active qualification,
then exits. A bounded result-wait timeout does not end this drain. Keep the relay
and pod service running until import finishes; then stop the relay and interrupt
the idle pod service with SIGINT so its adapter closes any resident screen engine.
Confirm the commissioned devices are idle before using them elsewhere. An idle
worker releases immediately; an active job must finish first.

The process manager must preserve an intentional stop, allow the drain to finish
without a kill timeout, and restart the same worker configuration on resume.
For systemd, use `KillMode=process`, `TimeoutStopSec=infinity`, and
`Restart=on-failure`; `systemctl --no-block stop UNIT` requests a drain and
`systemctl start UNIT` resumes. Disable the unit as well to keep it off across a
host reboot. Its stop hook must refuse GPU teardown while its owner still holds
an active lease. Resume starts the pod service and relay before the supervisor.
These controls act on the complete commissioned allocation: a TP1 pair has two
GPUs, while a TP4 pair has eight. Capacity is a ceiling, so stopping a worker does
not require changing the arena manifest or affect other workers.

Completed jobs release their pair immediately. An authenticated terminal
infrastructure HOLD also releases its pair while retaining the unresolved
submission; it is not a candidate FAIL. Later jobs continue, but their final
ranking and reward eligibility wait for every preceding submission in that arena
to finish or receive an existing terminal disposition. `settlement_candidates`
retains `reward_eligible` once the completed prefix reaches a PASS, so reopening
an older result cannot retract another miner's finalized credit. The dashboard
and weight producer consume this same eligibility.

Lease and recovery schemas migrate to version 3 without replacing retained
requests or evidence. Stop old database consumers before opening with the new
version, and restart every writer on the new source. Keep the database backup,
all worker spools and evidence together for recovery. Start with two pairs and
verify concurrent completion, ordered economics and restart without duplicate
evaluation before commissioning the remaining pairs. Raising capacity alone does
not commission GPU workers.

Schema version 2 preserves existing signed lease, request and event bytes.
Upgrade all CPU controllers at an idle boundary before enabling a second arena;
old controllers cannot operate the new schema. Keep a database backup and
commissioned packets for rollback.

Python `run_pass` / `run_validator` accept an injected `ArenaServiceRegistry`,
`arena_id` and `accept_legacy_bundles`. Standalone `chain-validate --arena-id`
cannot construct executable capabilities. `retained_only=True` processes the
queue at a fresh finalized head without advancing reveal history; it requires
injected arena authority and conflicts with `intake_only`.

## Operating signals

Monitor queue age/depth, capacity, stage latency, verdicts, holds, restarts,
stock-canary drift and evidence reopen failures. Label these by service digest,
runtime/model identity and lane; arena ID alone does not distinguish epochs.

The CPU relay refreshes its heartbeat during request and result copies; this
reports dispatcher liveness, while the pod heartbeat and request deadlines
retain their separate checks. An adapter timeout closes and reaps that adapter
before another request can use it. The existing cooldown then permits one fresh
boot; the timed-out request retains its infrastructure outcome and diagnostics.
Reopening retained qualification evidence preserves the audit's PASS, FAIL or
NO_DECISION verdict, including insufficient coverage.

## Nonclaims

Registration is not a performance win or proof of a representative workload.
Screens, waivers and optional sampled audits cannot replace qualification.
Qualification remains bound to one arena/stack and does not authorize release.

Next: [Authoritative qualification](qualification.md).

## Source anchors

- [Arena service types and registry](https://github.com/latent-to/cacheon/blob/main/cacheon/arena_service.py)
- [Resident screen bridge](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/resident_screen_lane.py)
- [Resident screen queue](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/resident_queue.py)
- [Resident OCI session](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/oci_resident_session.py)
- [Arena service tests](https://github.com/latent-to/cacheon/blob/main/tests/test_arena_service.py)
- [Qualification intake projection](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification_intake.py)
