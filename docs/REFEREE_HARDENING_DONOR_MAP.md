# Referee-hardening donor map

This is the extraction inventory for the frozen donor tree. `PORT` means transplant the
mechanism with its tests; it does not bless an uncommitted file wholesale. Mixed files are
always selected hunk-by-hunk.

| Owner | Donor modules | Disposition | Extraction rule |
|---|---|---|---|
| Vendor prerequisite | `eval/seccomp_moby_v0_2_1.json`, two `arena_assets/minimax_m3/sglang_patch/**` files, `NOTICE`, package-data declarations | VENDOR/extracted | Preserve the three donor bytes exactly; bind distribution, derivation, runtime-preimage, license, size, and hash in `vendor_provenance.json`. Do not port donor runtime/model-provision code. |
| PR 1 contracts | `capabilities.py`, `tensor_spec.py` | PORT | Strong standalone typed foundations. |
| PR 1 contracts | `slots.py`, `registry.py`, `dispatch.py`, `verify.py`, `verify_collective.py`, `receipts.py` | ADAPT/high reuse | Keep typed outputs, variants, domain routing, stock fallback, capture/replay, completed/fallback evidence; prepare loader for composite stacks. |
| PR 1 target catalog | `competition.py`, target portions of `manifest.py` | ADAPT | Keep singleton/atomic semantics. Remove device-only crown restriction and duplicate whole-serving titles. PR 1 defines targets, not economics. |
| PR 1 loading | `seam.py`, `audit.py`, selected `_launch.py`/`ipc.py` | ADAPT | Keep setup gating, shared-module behavior, rank coverage, safe result transport; in-engine evidence remains diagnostic. |
| PR 1 atomic patch | `dep_policy.py`, `patchers/apply_dep_patch.py`, `integrations/flashinfer_overlay.py` | PORT/ADAPT | Preserve the proven deep atomic replacement and fail-closed overlay handling. |
| Deferred | `device_component.py` and `validator_device` branches in parser/loader/rebuild/CLI | DEFER | Raw pointers add complexity without memory safety and are unnecessary for isolated component attribution. |
| PR 2 executor | `eval/device_state.py`, `eval/runtime_preflight.py`, `source_release.py`, `referee_runtime.py`, `model_provision.py`, `runtime_overlay.py` | PORT | Strong device/runtime/model/source sealing and release-resolution primitives. |
| PR 2 prebuild | `patchers/build_cuda_ext.py`, generic `rebuild.py`, `eval/oci_prebuild.py` | PORT/ADAPT | Preserve build-without-dlopen and load-only-inside-untrusted-engine; remove device-component hooks. |
| PR 2 streaming | `eval/oci_backend.py`, `eval/oci_outer_session.py`, session protocol/worker | ADAPT/very high reuse | Keep external timing, conditioning, device receipts, sandbox, watchdogs, artifact publication, and cleanup; input becomes `EngineLaunchSpec`. |
| Deleted after PR 2 | `eval/oci_protocol.py`, `eval/oci_worker.py`, one-shot/HMAC/result-file paths | DELETE | Move shared validation first, then remove superseded execution lane. |
| PR 2/4 attestation | `eval/host_attestation.py` | ADAPT | Runtime/device publication in PR 2; qualification and stack/delta binding in PR 4. |
| Shared policy | `arenas.py`, `compat.py` | ADAPT | Keep immutable arena identities; separate runtime/workload, quality, settlement, and release concerns as their owners land. |
| PR 3 stack | no complete donor module | NEW using existing primitives | Add catalog/stack/contribution/materializer/marginal transaction only; reuse manifest, registry, rebuild, publication, and OCI. |
| PR 4 scoring | `eval/scoring.py` | PORT | Keep bookended noise and three-way speed decision. |
| PR 4 quality | `eval/external_quality.py`, `eval/qualification.py`, quality session/outer code | ADAPT/high reuse | Preserve post-C sealing, teacher forcing, raw publication/reopen, three-way quality; bind PR 3 identities and add pristine T authority. |
| PR 4 orchestration | `eval/throughput_kl.py`, `eval/prompts.py`, `eval/capability.py` | ADAPT | Keep batching, token accounting, and hooks; remove duplicate local/one-shot crown paths. |
| PR 5 discovery | `system_patch.py`, `system_overlay.py`, system parser/bootstrap/sandbox/worker hunks | PORT/ADAPT | Keep bounded source deltas and post-spawn activation; change permanent whole-serving title into discovery/promotion. |
| PR 6 chain intake | `chain/payload.py`, `chain/fetch.py`, history/canary portions of `chain/__init__.py`/`validator_loop.py` | PORT/ADAPT | Preserve strict/finalized/hostile transport; reserve priority prefetch and publish rehashed worker-readable trees. |
| PR 6 provenance | `copy_fingerprint.py` | ADAPT | Fingerprint only submitted delta; exact/containment authoritative, shared fragments advisory. |
| PR 7 state machine | `commit_reveal.py`, settlement/retry/publication portions of `validator_loop.py`, weight portions of `chain/__init__.py` | ADAPT/high reuse | Keep durable semantics, pending recovery, holds, global projection, publication journal; replace whole-serving title assumptions and JSON production authority. |
| PR 8 releases | `source_release.py`, `referee_runtime.py`, `referee_release.py`, `model_provision.py`, packaging/legal hunks | PORT/ADAPT | Freeze nonzero identities only after clean extracted builds; release contains integrated code, never proposal bundles. |
| Every PR | `cli.py` | HUNK ONLY | Each command lands with its owner or moves into the subsystem; never transplant the mixed file wholesale. |

## Entanglement hotspots

- `manifest.py`: variants, dependency patches, device experiment, and discovery products
  share one parser.
- `competition.py`: useful target semantics are mixed with the wrong crown restriction.
- `oci_backend.py`: strong streaming/device code coexists with legacy one-shot and
  system-specific branches.
- `throughput_kl.py`: local launches, OCI orchestration, scoring, and teacher quality are
  coupled.
- `arenas.py`: runtime, workload, quality, settlement, and release policy are one object.
- `commit_reveal.py` and `chain/validator_loop.py`: valuable durability and incorrect
  product identity occupy the same files.
- `cli.py`: spans the entire extraction and is never a file-level transplant.

The extraction rule is therefore: port standalone mechanisms, adapt policy-bearing files,
and delete only after every still-used primitive has an explicit owner and regression.
