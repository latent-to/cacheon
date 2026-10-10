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
| Capacity | `max_queue_depth`, `max_queue_age_blocks`, `max_active_qualifications` and `max_cohort_size` |
| Authorities | Reviewed provider digest, qualification policy and closed targets |

The manifest carries schema version 3, and its digest identifies one immutable
commission. Changing the model, image, runtime, workload, hardware allocation or
policy creates new evidence identity. A provider must match the declared digest
and typed interface; deployment review establishes that its installed bytes
match that declaration.
This check is not remote attestation.

## Scored workload

The scored workload is a sealed agent-replay slice. The commission's
`session.replay` names the slice manifest and its digest, the operating load and
the number of paired windows; see
[Finite agent replay](qualification.md#finite-agent-replay). The arena's declared
workload cells set the input-token geometry and batch routing of the
engine-conditioning and audit batches; their output-token and timed-read fields are
retained identity only. Context length,
admission width, graph mode and watchdog settings come from the sealed engine
template. Qualification rejects a session that differs from its sealed authority
before launch.

A commission that seals no replay — the batch-cell speed policies 8–15, including
the v15 prefill lane — is refused. Evidence retains the version and arithmetic
under which it was measured.

The sealed input selects `registered_targets`, `model_profile_key` and engine
configuration. Commissioning derives `closed_targets` from the shared catalog,
so another arena adding a target cannot silently change this arena's policy.
Unmeasured submissions for closed targets leave as NO_DECISION and release
payment before dispatch; see [Why submissions fail](../miner-guide/why-submissions-fail.md).

## Admission and capacity

A published candidate enters qualification directly; the qualification's first
window is the only screen. The controller supplies a durable queue snapshot.
Capacity comes from measured operational budgets; finalized chain order remains
authoritative.

| Decision | Effect |
|---|---|
| `admit` | Claims a qualification cohort within available capacity |
| `queue` | Leaves work in its durable lane and stops selection this pass |
| `hold` | Retains work when age/depth/cohort limits require intervention |

Admission runs before the first claim: an exact copy of a loser inherits its
FAIL, closed targets release payment as NO_DECISION, and commitments after the first crown on the
commissioned baseline expire as `baseline_closed_at_submission` (NO_DECISION, payment released).
Candidate-caused qualification failure rejects; provider, baseline, teardown and
incomplete-evidence failures remain infrastructure failures. Qualification HOLD
has an explicit recovery path. Monitor held rows and preserve evidence; deleting
them is not recovery.

## Boundary types

`ArenaCandidateBinding` binds reservation, publication and the one-based
qualification attempt; a retry after an infrastructure hold is a new binding.
`ArenaQualificationRequest` and `ArenaQualificationWork` bind authoritative
qualification.

## Provider interface

`ArenaServiceProvider.build_qualification(request, state=None)` preserves the
claimed reservation order and supplies the frozen plan factory, isolated
executor, post-commit entropy, hidden judge and absolute deadline. The controller
checks these identities before importing retained outcomes.

`B300ArenaServiceProvider` is the shared implementation. Historical names and
signed digest domains remain stable; hardware comes from commissioned inputs.
Commissioning seals only `B300DeclaredAuthorities` (runtime identity plus the
declared qualification) into the manifest; executors, judge, entropy and
deadline exist only inside the qualification worker. Private capabilities remain
validator-owned.

## Deployment composition

The deployment materializer (`python -m cacheon.eval.b300_deployment
materialize`), qualification commissioner and worker consume one sealed
definition per arena. The materializer writes `deployment.json` (schema
`cacheon-b300-deployment-v3`), `arena-service-manifest.json` and
`worker-readiness.json` under its output root. READY's `gpu.inventory` describes
the host; `lane.devices` selects the commissioned TP lane and
`lane.tensor_parallel_size` must equal its width. Optional
`lane.baseline_devices` selects an equal-width, disjoint baseline lane of the same GPU model and memory capacity. If omitted,
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
`resources.runtime.cpu_pins` maps each physical GPU to a vCPU list: the first
entry pins that GPU's SGLang scheduler, and the rest join the lane's pool for
every other engine process. Pins bind the runtime digest; a lane GPU without a
pin, or an engine whose scheduler ranks do not match the plan, fails the launch.
Use it on hosts whose vCPUs differ in speed. Both lanes share one native
publication store under the commissioned root, so the swapped orientation reuses
the first build; lanes take turns building and reopening it.
UID/GID, executable and deadlines retain their existing authorities. Absent
resource objects preserve prior policy. Both authority and measurement inputs
must agree; every qualification executor consumes the same values.

Commissioning requires faithful, broken and infrastructure controls through the
production entrypoint, with graphs, audit/T roles, retained evidence and restart
recovery. CPU composition and compatibility checks do not establish GPU success.
The broken control must be faster than stock, so that it clears the speed gate
and the correctness audit is what rejects it; a control that only corrupts its
answer at stock speed fails the paired speed test first and proves nothing about
the audit.
The exception is a control listed by selected delta digest in the sealed
`qualification.policy.audit_control_delta_digests`, a sorted list of distinct
lowercase SHA-256 digests (omitted means none; the registered policy digest
changes only when it is non-empty). Its plan carries the calibration-observation
disposition: the audit still runs after a speed failure, pristine T does not, and
the verdict stays the speed grade unless the audit grades `FAIL` or `NO_DECISION`,
which exits at the audit stage. Intake and the deployment check refuse that
disposition for every unlisted delta.

A host that keeps the checkpoint in RAM reloads it at boot from an enabled
oneshot unit ordered before the container runtime, and reapplies the GPU clock
lock in the same unit, so a reboot cannot silently change the measured engine.
After a host reboot, restart the CPU-side pair units: their start hook
re-launches the pod service over the registered transport and then starts the
relay. Retiring a pair archives its whole tree except the materialized replay
slices and worker scratch, and the validator VM pulls those archives over the
pinned host key, so required evidence survives both pair retirement and pod
loss.

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
Commissioning a second arena cannot retire the first arena's baseline. Two arenas can qualify
concurrently. Within one arena, commission disjoint pairs with the same model,
runtime, workload, incumbent and GPU execution policy. Physical GPU addresses
belong to each worker's READY, registration and launch binding; equivalent pairs
share the arena service identity. Calibration and complete speed, audit and T
evidence remain specific to each job and physical pair.

Run one supervisor and relay per pair with distinct `owner`, registration, spool,
evidence and runtime paths. Each supervisor uses the existing qualification
entrypoint, with `enable_settlement=false` and `enable_weights=false`. A
separate supervisor runs economics with `enable_qualification=false`, so long
evaluations do not stop finalization. The sealed standing config has no screen
gate; a config that carries `enable_screen` is refused. The sealed service's
`max_active_qualifications` bounds transactional claims. One worker can hold
only one active job, and one reservation can belong to only one active lease. Recovery reopens by worker owner.

To free an allocation, stop its supervisor with SIGTERM or SIGINT. It stops
claiming new work, continues renewing and importing any active qualification,
then exits. A bounded result-wait timeout does not end this drain. Keep the relay
and pod service running until import finishes; then stop the relay and interrupt
the idle pod service with SIGINT.
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

`chain-validate` runs intake only. The standing supervisor's dispatcher binds the
commissioned `ArenaService` and selects its arena with `accept_legacy_bundles`; no
command-line flag selects an arena.

## Operating signals

Monitor queue age/depth, capacity, stage latency, verdicts, holds, restarts,
speed-stage `NO_DECISION` rates and evidence reopen failures. Label these by service digest,
runtime/model identity and lane; arena ID alone does not distinguish epochs.

The CPU relay refreshes its heartbeat during request and result copies; this
reports dispatcher liveness, while the pod heartbeat and request deadlines
retain their separate checks. After a successful verified SSH heartbeat read,
the relay saves the original payload as `state/worker-heartbeat.json` for dashboard
observation. Failed reads leave that worker timestamp unchanged; the relay's own
`state/heartbeat.json` remains independent. An adapter timeout closes and reaps that adapter
before another request can use it. The existing cooldown then permits one fresh
boot; the timed-out request retains its infrastructure outcome and diagnostics.
Reopening retained qualification evidence preserves the audit's PASS, FAIL or
NO_DECISION verdict, including insufficient coverage.

## Nonclaims

Registration is not a performance win or proof of a representative workload.
Optional sampled audits cannot replace qualification. Qualification remains
bound to one arena/stack and does not authorize release.

Next: [Authoritative qualification](qualification.md).

## Source anchors

- [Arena service types and registry](https://github.com/latent-to/cacheon/blob/main/cacheon/arena_service.py)
- [B300 deployment materializer](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/b300_deployment.py)
- [Arena service tests](https://github.com/latent-to/cacheon/blob/main/tests/test_arena_service.py)
- [Qualification intake projection](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification_intake.py)
