# Referee hardening audit

State date: 2026-07-11. Frozen donor branch:
`codex/referee-hardening-donor-20260711` (not for merge).

Product authority: `docs/PRODUCT_CONTRACT.md`. Extraction and finite exit criteria:
`docs/REFEREE_HARDENING_SPLIT_PLAN.md`. File/hunk reuse inventory:
`docs/REFEREE_HARDENING_DONOR_MAP.md`.

Operational clarification: B/C/B'/T are authoritative logical arms, not four cold model
loads per reveal. The product contract now requires staged screening, frozen-incumbent
cohorts, amortized B/B' bookends, batched post-destruction T grading, TP4 dual-half use where
registered, and measured queue/crown-latency budgets before launch.

Donor-freeze test receipt (2026-07-11, `pyenv activate sn120`, exact moving tree):
**1050 passed, 1 skipped, 25 failed in 36.37s**. Most failures are the already-observed
test-module arena object/fingerprint mismatch after the v12 registration changed. Distinct
failures preserve three known unfinished migrations: `EvalOutcome.kl_mean` aliases teacher
NLL, the raw-quality publication test omits the new required arena/full payload, and one
system-patch test still requests the superseded one-shot gate. The donor is compile-clean
and `git diff --check` clean, but intentionally not green; extraction branches must earn
their own clean receipts rather than repairing this archive in place.

This is the finding -> implementation -> test -> GPU-receipt index for the
post-blockscore review and the later joined no-go audit. It deliberately separates
local proof, portability proof, and B300 proof. The current tree now has a complete
non-crown RTX TP8 B/C/B' **runtime** receipt, but the stock-equivalent candidate exposed
an uncalibrated external token gate and therefore did not qualify. There is still no
approved source-digest freeze, registered B300 v11/v4 qualification, settlement receipt,
or new crown. Prior wins must requalify under this referee before they can emit.

Moving-tree versions: Qualification v12 (compile-clean, not yet test-certified),
host-attestation v4, active-device receipt v2, ledger v10. The last complete local
checkpoint, on the superseded v11 tree, was **1023 passed, 1 skipped in 34.25s**;
it does not certify the current tree.

## 2026-07-11 product-invariant correction and scope freeze

The current branch must **not** be merged or digest-frozen in its present shape. A
second independent audit identified both launch blockers and a product-level error in
the hardening direction. Root verified the following directly against the moving tree:

- chain archives are extracted as owner-only `0600`, while production prebuild mounts
  them read-only into an OCI process running as UID/GID 65532;
- chain/copy priority is materialized only after transport, manifest resolution, and
  fingerprinting, so network availability can reverse finalized reveal priority;
- target settlement filters invalid standing champions and overwrites the title map,
  which can silently delete a title and redistribute its emission;
- whole-product copy detection treats any fingerprint-set intersection as identity,
  allowing shared utilities or an early poison fragment to demote distinct products;
- ordinary untrusted-host slot and atomic bundles are explicitly denied component
  crowns and redirected to `sglang.inference.bundle.v1`, while the source-overlay lane
  separately crowns `sglang.inference.v1`.

The last item confuses the **execution security boundary** with the **economic/product
identity**. The corrected invariant is:

> Every B/C/B' arm executes as a complete untrusted engine in an isolated container,
> but C is rewarded for the smallest validator-controlled stack delta that distinguishes
> it from B/B'.

Therefore the intended product is a validator-assembled inference data plane:

1. a versioned validator-owned `StackManifest` selects the incumbent implementation for
   each registered slot or atomic target;
2. B and B' run that identical current stack; C runs the identical stack with exactly
   the submitted target replaced;
3. all candidate code remains inside the untrusted engine container—component reward
   authority does not imply importing miner code into the trusted controller;
4. a successful marginal substitution updates the canonical stack, followed by a full
   system regression and compatibility check;
5. bounded whole-system overlays remain an incubator/prototype lane with a distinct,
   limited promotion/bounty policy, not duplicate equal-emission serving titles.

This correction preserves the useful OCI referee, host-owned timing, B/C/B' lifecycle,
external quality, retained evidence, typed ABI/capabilities, and bounded source-overlay
work. It requires redesigning competition resolution, assembly, attribution, copy
scope, and settlement before launch. `validator_device` remains non-crown and should be
deferred; raw CUDA pointers are not needed to restore component economics because the
component runs inside the isolated engine.

**Scope freeze:** no new feature surface, GPU campaign, digest freeze, or weight
submission is authoritative until (a) in-flight qualification edits reach an atomic
checkpoint, (b) all writers stop, (c) the intended tree is committed, and (d) tests run
from a detached clean worktree. The next work product is a keep/delete/redesign matrix
and a reduced stack-assembly vertical slice—not another layer on the current hybrid.

Moving-tree size at this checkpoint: 57 tracked files changed by approximately
`+15,156/-1,111`, plus 59 untracked paths containing approximately 37,492 lines. This
is approximately 52,648 added lines in aggregate and is not reviewable as one change.

### Verified second-audit delta

The independent read-only audit classified the moving tree as roughly 31k authored
production lines, 16.6k tests, and 4.6k vendored assets. In addition to the five
findings above, it verified these remaining authority gaps:

- pass-start chain height is still reused after model-sized GPU work for retry,
  reconciliation, intent, pending, and confirmation authority; refresh immediately
  before publication and again after submission;
- quality has a three-way decision internally, but `EvalOutcome` does not preserve the
  quality `NO_DECISION`, so it cannot reliably enter the retry/hold path;
- calibrated arenas require hidden tasks, but there is no production hidden corpus,
  judge, secrecy/rotation policy, or retained judge evidence;
- throughput evidence retains derived rates rather than the raw token numerators,
  trusted elapsed intervals, charged-tail boundaries, and constituent rates needed for
  independent replay;
- post-C quality selection uses fresh randomness without retaining a commitment/reveal
  or deterministic post-commit entropy sufficient to reproduce the selection;
- the moving v12 schema currently aliases `QualificationReport.kl_mean` to teacher NLL,
  conflating it with the advisory rollout-KL quantity used by existing consumers.

Three attachment findings are improved in the latest untested v12 checkpoint: raw
teacher evidence now has content-addressed publish/reopen/regrade code; candidate token
IDs are checked against the pinned tokenizer vocabulary with typed candidate-only OOV
attribution; and weight publication writes a durable intent before calling the SDK.
These are code observations, not a green receipt.

### Keep / delete / redesign disposition

| Disposition | Subsystems |
|---|---|
| Keep | Streaming outer OCI referee and disposable prebuild; non-root/read-only/no-egress/seccomp boundary; host timing, conditioning, and device receipts; typed tensor ABI; capabilities/variants; execution-completion receipts; arena profiles; canonical qualification/attestation concept; bounded source-overlay incubator; finality/history/fetch hardening; retry leases and pending recovery; vendored assets with provenance. |
| Delete or defer | `device_component`; legacy one-shot OCI request/HMAC/result path; dead close protocol and production `candidate_audit` branches; superseded exact-token/V1 crown logic; dirty-branch compatibility helpers; redundant fixtures/tests for deleted paths. |
| Redesign | `StackManifest` assembly and marginal target substitution; one canonical reward family; transactional priority/intake/title/weight state; immutable worker-readable bundle publication; copy identity; three-way quality and raw evidence; shared immutable `SettlementEvidence`; common overlay builder; OCI backend decomposition. |

A review budget—not a cosmetic line target—is approximately 18–20k authored
production additions, at most 10k tests, and the 4.6k vendored assets in a separate
provenance commit. Reaching it requires removing/consolidating roughly 10–13k production
lines and 6–7k test lines while adding the missing stack-assembly authority.

Everything except final hardware calibration can be completed on RTX/testnet: the clean
checkpoint; chain fixes; one slot plus one atomic `StackManifest` vertical slice;
qualification stabilization; joined chain-to-OCI-to-settlement proof; permitted testnet
publication/restart; and branch reduction/splitting. B300 is reserved for consensus
constants, SM103/NVLink/P2P behavior, real blockscore/deep-fusion performance, and final
independent-seed crown reproduction.

## Original review findings

Claude's structured independent review of the original Codex findings returned six
`CONFIRMED`, three `PARTIAL`, and zero refutations. `PARTIAL` narrowed severity; it
did not make the underlying defect disappear.

| Finding | Review | Implemented boundary | Primary code | Local proof | B300 proof / residual |
|---|---|---|---|---|---|
| `setup()` ran outside its stated lane | Confirmed | `setup()` requires explicit framework arming; validator-device components forbid it; whole-serving setup remains inside the untrusted candidate container | `optima/seam.py`, `optima/eval/_launch.py`, `optima/manifest.py` | `tests/test_isolation.py`, `tests/test_eval_fidelity_mode.py`, `tests/test_device_component.py` | Current RTX outer bracket passed the lifecycle; B300 crown path pending. |
| Miner native code loaded in the trusted timing driver | Confirmed | Trusted host owns the clock outside the entire SGLang container. Native build occurs in disposable prebuild; load occurs only in candidate ranks. Raw-pointer CUDA components are development-only and cannot settle | `optima/eval/oci_outer_session.py`, `optima/eval/oci_prebuild.py`, `optima/patchers/build_cuda_ext.py`, `optima/rebuild.py`, `optima/competition.py` | `tests/test_oci_outer_session.py`, `tests/test_oci_backend.py`, `tests/test_cuda_ext_cache.py`, `tests/test_rebuild.py`, `tests/test_device_component.py` | Current RTX prebuild + B/C/B' completed with exact cleanup; B300 qualification pending. |
| `fired` receipt preceded successful candidate execution | Partial: accounting defect, not an independent crown exploit | Separate completed/fallback receipts and per-rank coverage; controller-observed output remains crown authority | `optima/receipts.py`, `optima/registry.py`, `optima/dispatch.py`, `optima/audit.py` | `tests/test_dispatch_execution_receipts.py`, `tests/test_seam_receipts.py`, `tests/test_slot_audit.py` | Diagnostic only; no process-local receipt can crown. |
| Missing/incomplete evaluator report could fail open; bench/margin-zero were crown footguns | Partial: default positive margin prevented a 1.0 crown | Qualification v11 is exact-schema and independently regraded; the controller recomputes decisions; command evaluation is development-only; bench cannot record; registered margin is immutable and positive | `optima/eval/qualification.py`, `optima/chain/validator_loop.py`, `optima/cli.py`, `optima/arenas.py` | `tests/test_qualification_report.py`, `tests/test_chain_validator_loop.py`, `tests/test_scoring.py` | RTX bracket exercised the path; registered B300 qualification pending. |
| Offline verify and live output ABI differed | Confirmed | Typed output dtype/layout/stride contract is shared by offline and live dispatch; graph replay poisons outputs between replays | `optima/tensor_spec.py`, `optima/slots.py`, `optima/verify.py`, `optima/verify_collective.py`, `optima/dispatch.py` | `tests/test_tensor_spec.py`, `tests/test_verify_cpu.py`, `tests/test_msa_prefill_block_score.py`, `tests/test_collective.py` | Historical B300 FP32 noncontiguous MSA slab and single/NCCL graph replay passed. |
| One implementation per slot and weak eligibility forced miner-owned fallback | Confirmed | Multiple named variants, declarative named-dimension capabilities, ambiguity rejection, N/A off-domain verify, validator-owned stock fallback | `optima/capabilities.py`, `optima/manifest.py`, `optima/registry.py`, `optima/verify.py` | `tests/test_capabilities.py`, `tests/test_kernel_variants.py`, `tests/test_msa_prefill_block_score.py` | Historical five-shape MSA proof: three in-domain pass, two off-domain N/A. Canonical product remains the score sheet, not `topk_idx`. |
| Multi-op bundle identity and rewards were inferred from `ops[0]` | Confirmed | Validator resolves an explicit `slot`, indivisible `atomic`, or whole-serving `system` target. There is no false per-member attribution | `optima/competition.py`, `optima/manifest.py`, `optima/commit_reveal.py`, `optima/chain/validator_loop.py` | `tests/test_competition.py`, `tests/test_per_slot_settle.py`, `tests/test_chain_validator_loop.py` | Marginal per-slot rewards still require ablations and are intentionally not claimed. |
| Eager audit did not prove graph-captured behavior | Confirmed | Real candidate capture/replay verification plus external controller-observed B/C/B' token/top-k fidelity; in-engine audit is diagnostic | `optima/verify.py`, `optima/verify_collective.py`, `optima/eval/oci_protocol.py`, `optima/eval/qualification.py` | `tests/test_verify_cpu.py`, `tests/test_collective.py`, `tests/test_oci_outer_session.py`, `tests/test_qualification_report.py` | Historical capture-conditional adversaries failed on B300; current RTX system bracket captured/generated. B300 serving qualification pending. |
| Submission axioms prohibited useful engine/topology changes; review claimed no safe middle ground | Partial: the existing FlashInfer overlay was already a bounded prototype | Validator-owned device ABI for non-crown development, bounded SGLang source-patch products, and inspectable whole-serving bundles under external qualification | `optima/device_component.py`, `optima/system_patch.py`, `optima/system_overlay.py`, `optima/manifest.py`, `optima/competition.py` | `tests/test_device_component.py`, `tests/test_system_patch.py`, `tests/test_dep_patches.py`, `tests/test_competition.py` | Historical device ABI and FlashInfer overlay proofs remain useful; the current SGLang/whole-serving lane still needs full qualification. |

## Joined no-go findings and current closure

The first green integration was not a freeze gate. A joined adversarial pass found
coherent retained-evidence rewrites, warmup/timed dilution, a candidate-controlled
conditioning gap, unsafe raw-pointer component authority, incomplete historical chain
state, and per-arena weight writers that could overwrite one another. The current tree
closes those code-level paths as follows.

| Finding | Current authority | Primary regression |
|---|---|---|
| Coherent miner/round/target/score/decision rewrite reused genuine retained evidence | Host-attestation v4 binds the complete settlement projection: miner, chain/seed/evaluation/round/block identity, arena/bundle, target/mode/members, score, phase-quality decisions, speed/confidence/crownability, quality summary, and qualification digest | `test_settlement_projection_rewrite_cannot_reuse_sidecar`, `test_cross_chain_validator_or_evaluation_transplant_fails` |
| Correct warmups or clean batches diluted a bad timed batch | Qualification v11 retains warmup and timed fidelity separately and grades every batch against its paired stock control; output IDs are an independent product in every lane | `test_correct_warmups_cannot_subsidize_corrupt_timed_system_batches`, `test_clean_batches_cannot_dilute_one_bad_timed_topk_batch`, `test_system_token_match_uses_exact_stock_control_without_new_margin` |
| Sleeping/cooling in setup warmups or between charged work and timing could subsidize a candidate | Each arm's settlement throughput is `min(timed median, conditioning rate)`. Earlier setup warmups are throughput-free but quality-graded. The charged tail begins after the final free response (or ready when none), spans every declared conditioning warmup, gap, sampled readiness, and first timed response, and retains every constituent floor | `test_final_warmup_conditioning_rate_caps_crownable_throughput`, `test_slow_final_warmup_caps_otherwise_fast_timed_throughput`, `test_slow_earlier_warmup_cannot_buy_a_discarded_cooldown_window` |
| Failed retries polluted authoritative device receipts | Only a successful arm's exact adjacent pre-idle / active-final-warmup / post-idle triplet is published. Failed attempts remain in a diagnostic stream; sequence and ordinal gaps are accepted, while reuse or non-monotonic evidence is rejected | `test_failed_attempt_receipts_remain_diagnostic_and_retry_publishes_one_triplet`, `test_failed_attempt_sequence_gaps_are_allowed_but_reuse_is_rejected` |
| Raw CUDA pointers were treated as a safe component crown boundary | Validator-device CUDA remains available for isolated development verification, but component settlement is rejected; crownable host-code work must compete as an externally graded whole-serving system product | `tests/test_device_component.py`, `tests/test_competition.py`, `tests/test_qualification_report.py` |
| Head-only reveal reads lost copy-priority history | Production reads the exact finalized block, paginates saturated per-hotkey history backward, preserves global `(block, hotkey, payload)` order, and fails closed on pruned, malformed, non-progressing, or over-budget history | `test_read_reveal_history_preserves_every_row_in_global_order`, `test_saturated_chain_reveal_history_paginates_to_genesis`, `test_historical_reveal_pagination_does_not_skip_quieter_hotkeys`, `test_unfinalized_reveal_never_fetches_or_enters_submission_ledger` |
| Per-arena daemons could replace the chain's complete weight vector | Ledger v10 computes one global vector across all registered `(arena, target)` titles, revalidates every retained crown, and refuses partial redistribution if any title loses authority | `tests/test_per_slot_settle.py::test_global_weights_never_redistribute_an_invalid_arena_title`, `tests/test_chain_validator_loop.py` |
| Active receipt could omit all pre-release work or lose its mandatory ready sample | Active receipt v2 binds an exact release index, the policy-sized consecutive pre-release active run, and exactly one post-release ready pass; the monitor thread has a start handshake and cancellation wakes its release wait | `tests/test_device_state.py`, `tests/test_oci_backend.py`, `tests/test_host_attestation.py` |
| More than one reveal at a page-boundary block could be skipped | Historical reads query the oldest boundary block itself, preserve distinct same-block rows, and fail closed when a saturated ten-row same-block page cannot prove progress | `test_historical_reveal_pagination_preserves_boundary_block_duplicates`, `test_historical_reveal_pagination_fails_closed_on_same_block_overflow` |
| Early system-overlay activation could receipt stock SGLang | Scheduler children validate the immutable overlay during site startup but force the exact package through a post-`spawn.prepare` meta finder; the loader receipts exact origin only after package execution, permanently blocks retries after failure, propagates overlay `PYTHONPATH` only to scheduler descendants, consumes one-child role markers, and serializes every armed process start | `tests/test_system_patch.py` (post-reset stock-vs-overlay import, post-loader failure/retry, marker consumption, concurrent helper race) |

## Settlement authority and durability

Qualification schema v11 and host-attestation schema v4 use a non-circular,
fail-closed construction:

1. The controller independently grades canonical B/C/B' timed samples, the charged
   conditioning floor for every arm, and phase- and batch-separated paired fidelity.
2. It projects every settlement-affecting identity and decision into canonical
   qualification evidence.
3. It immutably publishes that evidence with the stock-runtime preflight and exact
   successful pre/active/post device triplets in a controller-owned,
   content-addressed sidecar. Failed retries are diagnostic only.
4. It binds the sidecar digest without changing the qualification-evidence digest,
   then reopens and regrades both before settlement and emission.

Ledger schema v10 carries chain scope, the externally-known validator identity, the
fsynced evaluation lease, full settlement projection, qualification digest, and host
digest through `EvalOutcome -> Score/EvalRecord -> PendingSettlement -> Champion ->
global weights`. Settlement recovery is idempotent across both persistence boundaries
and never replays GPU work. One owner-controlled lock spans finalized chain intake,
ledger load, evaluation, settlement, retained-evidence verification, and weight
publication. Corrupt or cross-scope state fails closed; durable validator-fault holds
require explicit operator release.

Primary proof:

- `tests/test_qualification_report.py`
- `tests/test_host_attestation.py`
- `tests/test_host_attestation_integration.py`
- `tests/test_pending_settlement.py`
- `tests/test_ledger_durability.py`
- `tests/test_per_slot_settle.py`
- `tests/test_chain.py`
- `tests/test_chain_validator_loop.py`

## OCI runtime boundary

Every candidate-bearing Docker path (legacy launch, streaming session, and disposable
prebuild) explicitly selects `--runtime=runc` and uses the vendored Moby v0.2.1
seccomp profile. The exact profile SHA-256 is
`de1f5327ca42b80be02daba8d39c0d087a530dc3c16f7028170fe068c9d66e61`;
it is included in source releases and wheels and rehashed immediately before every
launch. The remaining portable policy is non-root UID/GID, no network, read-only root,
no-new-privileges, all capabilities dropped except `SYS_NICE` and `SYS_RESOURCE`,
private IPC and Docker's default-private PID namespace, bounded resources, exact GPU
selection, read-only untrusted inputs, and independently verified container removal.

The RTX provider reports no portable AppArmor confinement, so this audit does not claim
a MAC boundary. NVIDIA driver ioctls remain a host-kernel trust surface; deployment on
dedicated validator hosts should add a provider-tested MAC policy where available.

Primary proof: `tests/test_oci_backend.py`, `tests/test_runtime_preflight.py`,
`tests/test_isolation.py`, `tests/test_source_release.py`, and
`tests/test_packaging.py`.

The RTX integration found defects that the mocked boundary tests did not: JSON-phase
worker errors were hidden behind a generic ready-marker failure; three read-only-mount
helpers used nonexistent `stat.ST_RDONLY`; the driver inspected SGLang's lazy `Engine`
object as a concrete class; top-level `multiprocessing.Process` left `_start_method=None`
while using the selected spawn context; and early `.pth` activation was erased by
`multiprocessing.spawn.prepare`. Each is fixed and regression-tested. The first 309.62
tok/s candidate run used the early receipt and is quarantined as a phantom; it is not a
valid overlay receipt.

The current immutable RTX release is source
`sha256:34f50f1a4bebecf6cc757e2ea2a6594311c3c2567e64fd0113c9d81ff21dea34`, tree
`sha256:d619041e8273bcb1ebb50aeaacefc927df6a2cb2dc694d2f331972e88e520096`
(89 files), remote `/root/optima-referee-d619041e8273bcb1`. These are development
identities, not the approved arena freeze constants.

## Merge-size audit

An independent moving-tree snapshot measured +47,710/-1,072 across 111 files:
+27,864/-951 authored production, +15,007/-92 tests, +4,610 vendored assets, and
+229/-29 docs/legal/packaging. Main contained only 11,965 tracked `optima/*.py` lines;
the working tree is about 3.25x main. Green tests do not justify that shape.

Disposition gate before merge/digest freeze:

- retain the typed ABI/capability/competition design, streaming outer referee,
  immutable qualification/host evidence, and bounded whole-serving overlay;
- delete the superseded one-shot OCI timing/request/HMAC result lane, dead close
  protocol, production `candidate_audit`, and unused compatibility helpers;
- defer/remove the non-crown raw-pointer `device_component` experiment;
- consolidate repeated runtime/arena/evaluation/competition/decision fields into one
  immutable settlement-evidence value while preserving v10 bytes and pending recovery;
- factor repeated test fixtures and split chain/accounting, evaluator, evidence,
  whole-serving overlay, and packaging/ops into reviewable commits;
- target 18-20k authored production additions and at most 10k test additions, then
  rerun this necessity audit. This is a review budget, not permission for cosmetic cuts.

## Current receipts

| Receipt | Location / identity | What it proves | What it does not prove |
|---|---|---|---|
| Full joined local suite | `pyenv activate sn120 && pytest -q tests/` -> **1023 passed, 1 skipped in 34.25s** | Current integrated CPU/mocked authority, isolation-policy, packaging, chain, ABI, system-overlay spawn/import, and failure-path regressions | Real Docker/NVIDIA/SGLang behavior or a digest freeze |
| RTX SM120 exact-seccomp CUDA proof | Via `experiments/minimax_m3/pod_exec.sh`; Docker 27.3.1; image ID `sha256:a238d3da9bf518ff54bd356b5946177521856ee49fc1fe19d84391619a391625`; profile SHA above; recorded in `WORKLOG.md` and the node ledger | The explicit `runc` plus exact custom seccomp and production-shaped non-root/read-only/no-network/cap-dropped policy allowed Torch 2.11.0+cu130 to run simultaneous three-iteration 8192x8192 BF16 matmuls on all eight RTX PRO 6000 Blackwell SM120 GPUs without error | It is **not** MiniMax/SGLang, NVLink, B300, performance, B/C/B', settlement, or crown evidence |
| Corrected RTX system candidate | Current release/tree above; prebuild publication v13; 8 prompts x 64 tokens, TP8 | Exact system overlay built/published, post-loader package-origin check passed, M3 loaded/captured/generated, effective 314.7783 tok/s (timed 317.6548/380.7939), exact pre/active-v2/post receipts, zero residue | One arm only; no bookend quality/noise decision; not B300 |
| Complete RTX system B/C/B' | Same release, fresh engines; B=315.8942, C=315.2577, B'=316.3577 tok/s; noise=0.1466%, speedup=0.99725 vs 1.005; nine receipt hashes in `WORKLOG.md`/node ledger | Production-shaped prebuild plus fresh B/C/B' streaming/device/container lifecycle completed and a no-op overlay correctly did not clear speed; cleanup was exact | Quality did **not** pass: stock B/B' output IDs matched only 315/1024 and candidate B/C 210/1024. This is a calibration failure of the current exact one-control token floor, not a qualification/crown receipt |
| Independent RTX bracket + per-batch quality | Prompt seed `0xA11CE002`; B=316.0376, C=313.8740, B'=314.5187 tok/s; noise=0.4818%, speedup=0.99555 vs 1.00964; nine new contiguous receipts | Reproduced the full lifecycle and no-op speed rejection. Every paired top-k distribution check passed in all two timed and three warmup batches | Exact-token gate false-failed again: timed candidate matched 360/1024 while its stock control matched only 313/1024, but batch 2 rejected 146/512 vs 148/512. This second seed proves threshold tuning on one autoregressive sample is not a defensible fix |
| Current-schema live testnet 307 | Finalized history through block 7532504, hardened HTTPS archives `5414ff5d...` and `40d05152...`, ledger v10; immediate restarts | Same-hotkey history preservation, fetch/re-hash, legacy rejection, explicit system identity, development-only evaluation, later-hotkey exact-copy demotion, arena-fingerprint re-evaluation, and `new=0` idempotence against the public chain | No current-schema GPU qualification, crown, or weight push |
| Live stale-weight and writer-lock negatives | Default validator uid 3/permit=true; active sparse row `{uid4: 43690, uid5: 21845}`; same ledger restarted and contended | Current SDK sparse-weight decode, read-to-dry-run UID mapping, durable no-refetch restart, fail-closed stale-emission guard, and exclusive whole-pass lock all exercised against live chain state | It deliberately did not sign or neutralize historical weights |
| RTX system isolation negatives | Immutable v15 release; disposable post-prebuild tamper, import-failure, clock/power-mutation, prebuild-timeout, and init-timeout bundles | Artifact mutation rejected before launch; 0.25 s prebuild watchdog force-removes in 0.345 s with no partial files; deterministic candidate failures publish no authoritative receipt; non-root clock/power attempts leave all eight GPUs byte-identical; 12 s init watchdog reaps in 15.87 s; every case leaves zero container/GPU residue | SGLang's fatal grandchild path kills the worker before preserving the exact import traceback; failure remains terminal but diagnostic cause is generic exit 137 |
| RTX pinned teacher API | Stock TP8 MiniMax-M3; batched `input_ids`, per-row `logprob_start_len`, `max_new_tokens=0`, top-5 and targeted sentinel | For a 7-token prompt with start 6 and four response IDs, all three input-logprob arrays return one leading prompt-boundary row plus exactly four response rows; slicing the final four is exact, and the sentinel logprob equals the first target logprob | Protocol mechanics only; no acceptance thresholds, hidden-task utility, qualification, or crown |
| Historical non-root OCI/resource probe | `experiments/minimax_m3/frontier_2026-07-09/artifacts/codex_grounding_20260710/oci_nonroot_resource_probe.json`, SHA-256 `4169a826672c684061f6d691d052879eabebbcc7a354c11af9b53a3de2d5d4e4` | UID 65532 CUDA init, read-only/cap-dropped container, topology/JIT tmpfs facts on B300 | Current exact-seccomp policy or final scoring path |
| Historical validator-device B300 smoke | `experiments/minimax_m3/frontier_2026-07-09/artifacts/codex_grounding_20260710/validator_device_smoke_b300_receipt.json` | Disposable compile, cubin ABI, numeric/graph/load/cache/drain on B300 | Crown safety for raw-pointer components; that lane is now development-only |
| Historical B300 graph probes | Same artifact directory: `graph_verify_single_gpu_probe.log`, `graph_verify_collective_probe.log`, `graph_verify_fused_epilogue_probe.log` | Real capture/replay and capture-conditional attack resistance | End-to-end serving score under the current referee |
| Historical isolated deep engine | Log SHA-256 `7a69700eaf70348a664d32e49c4579d3fc9f75a5c52dce5065a761ace552cb17` recorded in `WORKLOG.md` | TP4 B300 model load, tuning, capture, generation, and drain in isolated OCI | Current qualification/sidecar/runtime authority |
| Historical stock runtime preflight | Canonical receipt SHA-256 `7273357cac69893b90dcc15772490f524e3e0a165d59525969f2ac7dae7f5254` | Pinned image/local ID and installed runtime metadata without candidate/model/GPU exposure on B300 | Current candidate-bearing custom-seccomp path or B/C/B' performance |
| Historical B300 device guard | Both TP4 halves passed three consecutive stable idle samples; recorded in `WORKLOG.md` and the node ledger | Host-owned GPU identity/configuration and pre/post drain behavior on all eight B300s | Current successful pre/active/post triplets or conditioning-rate calibration |

## Remaining RTX/testnet work and irreducible B300 gates

- On RTX, expose every per-batch top-k/output-token metric and repeat stock/no-op brackets
  across independent prompt seeds. The current output rule requires a candidate batch to
  meet the exact single observed stock-control match rate; the stock-equivalent full bracket
  false-failed under autoregressive M3 divergence. Design and adversarially test a versioned
  hidden external task/teacher-forced distribution gate rather than loosening an exact-token
  threshold until sabotage passes.
- Carry an RTX result through Qualification v11 creation, host-attestation v4 publication,
  retained reopen/regrade, EvalOutcome/Score/EvalRecord/Pending/Champion, crash recovery,
  and zero-emission rejection for non-crown hardware.
- Current-schema live testnet-307 intake/restart, hardened HTTPS, sparse weight
  reconciliation, stale-emission refusal, dry-run UID mapping, and whole-pass contention
  are now proven. Still required: carry a real authenticated GPU qualification through
  settlement and an actual weight extrinsic when current authority exists. Wallet `default`
  has uid 3 and permit=true. Keep chain keys on the control box; use a production
  remote-executor boundary rather than blessing arbitrary `--eval-cmd` output as crownable.
- RTX post-publication tamper, candidate import failure, container/GPU cleanup, prebuild
  and init timeouts, and non-root GPU clock/power mutation are now live-proven. Still run
  retained-evidence corruption/reopen after the new qualification product is
  available. Wrong-origin/import-retry logic has focused spawn-path regressions; reproduce
  on hardware only if a legal system patch can reach that pre-import boundary.
- On B300, calibrate the external quality policy, charged-tail/poll overhead, pre-ready
  cooldown sabotage, B/B' drift, and false-crown rate. RTX data informs implementation but
  cannot set consensus constants.
- Build and verify a fresh source release and wheel, freeze their approved identities,
  then run the complete current MiniMax/SGLang outer B/C/B' bracket on B300. Retain,
  reopen, regrade, and settle the resulting v11/v4 evidence before any crown claim.
- Validate registered TP4 SM103/NVLink topology, the real blockscore/deep-fusion candidates,
  capture/replay, and custom-all-reduce scheduler-descendant/P2P behavior. Future PP/DP
  arenas must derive scheduler coverage beyond `tp_size`.
- Treat one validator/netuid as one coordinated writer: all arena workers must share the
  ledger, lock, retained-artifact root, registered-arena set, and weight publication.
  Separate per-arena ledgers or hosts are not safe without a real shared coordinator.
- Use an archive-capable chain RPC for historical pagination. Confirm the protocol cannot
  hide more than ten same-hotkey reveals at one identical block, or replace SDK pagination
  with an indexed event-history source.
- Validate the deployment host's LSM/MAC policy and continue to treat NVIDIA-driver ioctls
  as a dedicated-host kernel residual rather than claiming container isolation alone is a
  complete kernel-security boundary.
- MSA fused score-to-`topk_idx`, `StackManifest` composition, ablation-aware marginal
  rewards, and a resident/cheap hill-climbing tier remain subsequent product work. They are
  not hidden claims of this hardening branch.
