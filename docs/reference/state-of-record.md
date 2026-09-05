# State of record

This page is the dated capability and evidence ledger for Cacheon. Evergreen
pages define contracts and procedures; this page identifies the implementation
revision, evidence class, and unresolved limits behind readiness claims.

Passing tests, completing an empirical qualification, activating an incentive
policy, publishing weights, and producing a deployable engine release are
different events. Evidence for one does not authorize another.

On **2026-09-05**, the GLM branch incorporated PR #110's source-build-only
execution path while preserving the GLM catalog's exclusive overlap policy and
retirement of the M3 attention and dependency-patch surfaces. Reward projection
now reads each retained stack's own catalog, preserving historical v1 and current
v2 active relationships and exact claim bindings in one global vector. New remote
qualification requests bind the commissioned incumbent stack and tree before
resident entry; completed retained requests reopen under their original binding.
These source changes require a fresh GLM runtime commission.

The distribution also drops the retired MiniMax SGLang overlay copies and their
packaging exceptions. The GLM worker consumes the pinned upstream SGLang image.

The GLM-5.3 branch moved the source compatibility target to SGLang `0.5.18` on
2026-08-30. Exact-image 8xB300 controls now establish full-model generation,
all-rank seam activation, graph capture, deliberately broken and faithful
bundle behavior, and one complete mixed-cell B/C/B′ arena read. They do not
establish a winning miner implementation, independently reproduced PASS pair,
commissioned mainnet cutover, or deployable Engine release. The dated source
snapshot below records the earlier `0.5.13.post1` repository state and is kept
unchanged as historical evidence.

## Source snapshot

Snapshot date: **2026-08-19**

| Item | Value |
|---|---|
| Repository | [`latent-to/cacheon`](https://github.com/latent-to/cacheon) |
| Implementation revision | [`d94713a7`](https://github.com/latent-to/cacheon/commit/d94713a74a78f8123a18338aca0e7239f988ba6b), merge commit for PR #95; PR head [`53aaff30`](https://github.com/latent-to/cacheon/commit/53aaff3072a9c5a81898fe26f2922c25ed22ec3e) |
| Production Python | 207 files and 144,474 lines under `cacheon/` |
| Tests | 199 Python files and 98,322 lines under `tests/` |
| Merge validation | All required PR #95 checks were non-failing at `53aaff30`: seven passed and the frontend rebuild was skipped; the CPU suite completed in 4m44s alongside distribution, hygiene, documentation, and code-analysis checks |
| Test command | `python3 -m pytest -q tests`; CI used Python 3.11 |
| SGLang pin | `0.5.13.post1` in `cacheon/compat.py` |
| Bittensor raw-reveal storage ABI | `10.3.2` in `cacheon/chain_canary.py` |
| Public CLI | 28 commands |

The public product, Python package, CLI, environment-variable, HTTP-header,
and protocol-identity names formerly branded Optima are now Cacheon
(`cacheon` / `CACHEON` / `X-Cacheon-*` / `cacheon.*` digest domains /
`cacheon-op-abi-v0`). Since 2026-08-03 new publications and every mutable
protocol surface use the Cacheon vocabulary: the shared-weight transport reads
a single strict Cacheon dialect, other pre-rename `optima.*` identifiers are
refused fail-closed, and domain-stamped digests (model provision, weight
projections, intake scope) rotated with the vocabulary. One reader-only
exception preserves the exact hash-bound `optima-op-abi-v0` manifest spelling
for already-finalized bundles; readers do not rewrite its committed bytes.
The rename does not alter kernels, timed
evaluation arithmetic, or crown/settlement formulas. File and line counts describe the accompanying
change set; they are not quality metrics. The suite is
CPU/non-empirical validation and does not establish GPU performance,
container-runtime isolation, chain finality, or serving readiness.

## Authority order

When sources disagree, apply this order:

1. executable code, closed registries, schemas, and tests in the referenced
   revision;
2. the normative [product model](../architecture/product-model.md),
   [slot contract](../architecture/slot-contract.md), and
   [emissions policy](emissions-policy.md);
3. authenticated retained evidence and immutable publications;
4. this dated ledger; and
5. campaign notes, plans, console output, and historical narrative.

The evidence classes are intentionally non-substitutable:

| Evidence | Establishes | Does not establish |
|---|---|---|
| Unit/focused suite | Implemented invariants under test fixtures | Real GPU, chain, or serving behavior |
| `scan` / `verify` | Static policy and component correctness | Complete-engine speed, pristine quality, or settlement |
| Resident screen | Registered routing decision | Qualification PASS, crown, or reward speedup |
| One qualification PASS | Decision under one frozen authority | Crown or release readiness |
| Two independently bound PASSes | Settlement eligibility for one exact contribution/context | Integration or serving authorization |
| Confirmed weight journal | Exact vector read back under the publisher's policy | Qualification, activation, or release authority |
| Signed release verification | Descriptor, artifact, model, and signature consistency | Successful registry build or production serving |

## Implemented surfaces

### Submission, intake, and transport

- Finalized timelock commit-reveal establishes ordering and content identity.
- The fetcher enforces HTTPS/network policy and bounded gzip/tar preflight,
  including PAX/GNU extension payloads, before extraction.
- Current bounds include 64 MiB compressed, 256 MiB extracted, 4,096 members,
  16 MiB per file, 8 MiB per inspectable file, and 32 MiB aggregate inspectable
  content. Extension headers are bounded separately.
- Deterministic re-hash, cumulative copy disposition, immutable worker
  publication, and reopen-before-use remain required.
- `chain-publish` packages a miner bundle into a content-addressed key on any
  S3-compatible endpoint and proves anonymous HTTPS reopen through the production
  validator fetcher. Hippius and MinIO are optional configuration presets, not
  protocol identities.
- Legacy schema-3 single-PASS migration holds are non-crownable and have an
  evidence-preserving archive command.
- Optional eval-cost admission (default off) requires a coldkey
  `Balances.transfer_keep_alive` of the published TAO amount to the current
  subnet owner coldkey, bound by a content-hash remark and consumed once. v1
  quotes freeze that amount for 300 blocks (~1 hour). A later reveal of the
  same bundle may attach an unused payment pointer; intake consumes the pointer
  only on reserved or deferred admission.
- `chain-eval-cost-credit` can grant one audited, hotkey-scoped artificial
  make-good. It admits the oldest matching unpaid reveal exactly once, consumes
  transactionally only on admission, and never masks an invalid payment pointer.
- `chain-reservation-status` and `chain-miner-report` expose privacy-safe private
  diagnostics without taking the controller's writer lock. They report typed
  retained causes and evidence limitations rather than inferring candidate blame
  from a status string.
- Candidate load/invocation exceptions now cross the OCI boundary as a distinct
  candidate-owned type only when scheduler receipts prove ownership. Qualification
  publishes a content-addressed failure product containing the exact arm, launch,
  reservation, rank/slot/phase/file/line error, and retained diagnostic hash.
  Fresh registered qualification is singleton-bound, so that proved exception is
  terminal `FAIL` without a retry. Validator-runtime receipts remain infrastructure
  outcomes.
- Engine/native stdout and stderr share the retained bounded OCI diagnostic after
  the framed protocol descriptor is reserved. `explain --log <artifact>` works
  without a product. Fetch and manifest failures retain safe generic guidance
  without exposing the validator's private path or transport detail.
- Every new remote request also publishes the existing `worker_log` output role.
  It joins adapter phase events and every OCI diagnostic receipt/raw prefix to the
  request ID on both success and infrastructure failure. `chain-miner-report
  --remote-spool-root` combines it with immutable lease/recovery and transport
  events, so the displayed failure names its observed component and exact traceback.
  The read-only submissions dashboard now renders the same explanation and exposes
  a per-request raw diagnostic log, with partial retention stated in the section
  headers.

### Validator recovery archive

- `chain-validate` appends a redacted, fsynced JSONL operational chronology by
  default. It records finalized positions, content-derived identifiers, bounded
  disposition classes, and fault types without URLs, hotkeys, candidate bytes,
  exception messages, wallets, credentials, or ambient environment. SQLite
  remains transition authority. The audit append heals its own canonical
  parent directory to the owner-only mode when a sibling component created it
  first under the ambient umask (2026-07-30), instead of refusing every append
  on a split storage layout.
- `chain-snapshot` uses SQLite's online backup API and publishes a closed,
  digest-bound private recovery manifest. It includes the consistent database,
  redacted journal when present, database-referenced worker publications and
  retained settlement qualification artifacts, plus only explicitly named sealed
  inputs. Models, OCI images, wallets, credentials, caches, unredacted logs, and
  unrelated evidence are not auto-discovered.
- Every archive read is byte-bounded; uploaded objects are reopened; and
  `chain-snapshot-verify` semantically restores into a fresh private staging root,
  checking SQLite, worker receipts/content hashes, evidence references, and audit
  structure. The object store is neither the live database nor the live evidence
  filesystem, and restore never overwrites live state.

### Qualification continuation recovery (2026-08-09)

Resident qualification now commits completed eager-audit and pristine-T evidence
to the existing fsynced continuation records before the producer returns. A crash
after that commit reopens the durable result without entering the evaluator again.
An armed evaluator with no completion record still holds fail closed; the CPU
contracts do not establish recovery before any host byte becomes durable or prove
the behavior on B300 hardware.

### Fresh-pod model reopening (2026-08-09)

Deployment commissioning can require a provisioned model tree to be publicly
readable and read-only while it reopens the canonical receipt, complete file
inventory, and actual file bytes. This is a local fail-closed input check; it
does not establish that a paid OCI lifetime has mounted or executed that model.

### Slots and targets

The executable catalog contains 11 slots and one registered atomic target:

| Kind | Registered identifiers |
|---|---|
| Op | `activation.silu_and_mul`, `norm.rmsnorm` |
| Block | `linear.dense`, `moe.fused_experts`, `moe.fused_routed_experts`, `norm.fused_add_rmsnorm` |
| Collective | `collective.all_gather_into_tensor`, `collective.all_reduce`, `collective.ar_residual_rmsnorm`, `collective.reduce_scatter_tensor`, `moe.fused_experts_reduce` |
| Atomic target | `collective.dp_attention_exchange.v1` over all-gather and reduce-scatter |

On 2026-08-30 the four M3 attention slots (`attention.sdpa`,
`attention.decode`, `attention.msa_block_score`,
`attention.msa_prefill_block_score`), their seam adapters, ABIs, catalog rows,
example bundles, and the MSA verification-probe synthesis were retired from the
tree as part of the generic slot-contract work; `moe.fused_routed_experts` (the
fat routed-MoE boundary), `linear.dense`, `norm.fused_add_rmsnorm`, and the two
DP-attention exchange members were registered in the same change. The obsolete
deep-finalize/atomic-MoE-epilogue and dependency-patch surfaces were removed.
This is a reviewed target identity epoch: catalog and settlement digests moved,
and each arena's registered set is now pinned arena data rather than derived
from the cross-arena catalog. Historical evaluation records embed their own
catalog snapshots and are unaffected. The deployed M3 lane runs from its pinned
source trees and is unaffected by repository-side retirement.

The table records contracts, not deployment availability. In the retired
MiniMax-M3 arena three targets were unavailable: `norm.rmsnorm` (GemmaRMSNorm
outside the registered seam), `activation.silu_and_mul` (the model activated
inside the MoE GEMM epilogue; the dense-layer swigluoai callsite was unpatched,
measured never-called on 2026-08-23), and `moe.fused_experts_reduce` (sealed
closed pending a full-engine outer-reduction proof that was never produced).
Their computation stayed claimable inside open fused targets; closure keys on
the submitted target name only. The miner-facing notice is
[Arena availability](../miner-guide/slots.md#arena-availability).

The full GLM-5.3 arena instead registers exactly
`moe.fused_routed_experts`, `linear.dense`, `norm.fused_add_rmsnorm`,
`collective.all_reduce`, and atomic
`collective.dp_attention_exchange.v1`. Pure pointwise activation/norm and
engine-owned KV-cache, batching, radix, and speculative-decoding policy are not
separate GLM reward lanes.

On 2026-09-01 target-catalog v2 removed MoE `first_applicable` after a narrow
candidate was shadowed. Overlap is now exclusive; B300 recommission remains required.

On 2026-09-02 the deployed validator lineage through 2026-09-01 (completed
verdicts preserved across a crown, settlement owned by the standing supervisor,
screen results preserved across restart, retained-PASS reward merging) merged
into the GLM branch. Its separate reduce-first planner patch was superseded by
target-catalog v2 rather than merged; the GLM branch remains the single source
for the next commission. The same day the recoverable qualification dispatcher
began installing or verifying the commissioned incumbent against the durable
evaluation stack before any claim: a commission pinned to a superseded baseline
fails before a lease or GPU request exists, and a new arena such as GLM-5.3
receives its genesis stack row from its first commissioned claim instead of
failing at its first PASS commit.

On 2026-08-30 the exact `0.5.18` CUDA-13 image, full
`incoai/GLM-5.3-NVFP4` checkpoint, and two physical TP4 lanes exercised all
five registered targets (six callable seams because the atomic DP-attention
target binds both all-gather and reduce-scatter). Component controls retained
captured all-rank receipts and included a routed-MoE negative control that
changed model-visible output, excluding silent stock substitution.

The concluding mixed-cell control consumed three 128-request
8,192-input/1,024-output batches and three 24-request
65,536-input/4,096-output batches. It completed all 18 B/C/B′ response
artifacts, passed the sealed numeric quality judge, and recorded exactly 226
captured `moe.fused_routed_experts` calls on each of ranks 0–3. The faithful
stock-equivalent candidate measured 1.0013560726x against a 1.0102795639x bar,
so its terminal decision was correctly `FAIL`; a faithful bundle is an
execution control, not a fabricated speed winner. The canonical acceptance
summary digest is
`d5ce302b5b2fb85b6c01a5a556b29bb08e3a8b8edb91d25072fd7587392fa40d`.
This establishes exact-runtime off-chain arena execution and fail-closed speed
grading for the exercised target. It does not substitute for a production
commissioned qualification, a miner speed PASS, reproduction, settlement, or
mainnet activation.

The same artifacts fix the arena's measured wall-clock envelope: warm engine
start to first batch took 196 seconds, and the steady sealed windows took 66
seconds (short cell) and 122 seconds (long cell). The screen deployment's
1,800-second initialization and batch ceilings — raised on 2026-08-31 after a
healthy four-rank MiniMax-M3 graph capture and its first production batch each
outran the prior 900/600-second values — cover these measured windows with
better than nine-fold start and ten-fold first-batch margin. On the same day
the deployed validator lineage (settlement on every pass, the champion-floor
projection, and both ceiling raises) merged into the GLM branch, so one source
now carries both the mixed-cell arena and live mainnet behavior.

On 2026-09-02 the recoverable qualification dispatcher began installing or
verifying the commissioned incumbent against the durable evaluation stack
before any claim. On 2026-09-03 this became queue-segment-aware: reservations
persist the exact stack assigned to their FIFO segment, an old resident may
drain its already-bound segment after settlement advances durable lineage, and
a different commissioned stack is requested before the first lease in the next
segment. A new arena still receives its genesis stack row from its first
commissioned claim instead of failing at its first PASS commit.
The gap was reached on mainnet the same day: reservation `69f50573` passed
qualification at 1.0827x on the recommissioned arena, and the standing owner
rejected the result with "evaluation stack is not initialized" because no
remote-path caller had created the arena's row since `e1c77204`.

On 2026-09-04 the sealed direct-artifact lane (`cutlass.cute.cubin.v1`:
`aot_exports` and `artifact_resources` manifest rows, the artifact provider
registry, CUBIN admission/launch/materialization, the compile-profile prebuild
path, and the `artifact_context` seam row) was retired from the tree. No
submitted mainnet manifest had ever declared it (285 stored manifests checked
against the live intake store). A manifest that still declares those rows is
rejected at intake as an unregistered op field. Removing the provider registry
from the catalog snapshot rotated the catalog digest from `dd61300c…` to
`aec370cd…` and therefore the evaluation stack digest; deploying this tree is
a fresh-arena recommission, and its first real screen plus qualification on
the pod is the proof of the deployment. The launch digest of ordinary source
bundles is unchanged; the resident audit binding digest no longer carries a
null compile-profile key.

The same rotation surfaced a latent weights defect on 2026-09-04: the reward
projection required every crowned stack's sealed catalog snapshot to equal the
live catalog byte for byte, so the four crowned arenas sealed under
`dd61300c…` could not be projected on the new tree and the weight-offer
service failed every tick. The fence now compares the reward-relevant catalog
policy only (target identity, structure, contracts, and composition rules);
admission policy and retired sections may differ without re-crowning. See
`docs/validator-guide/settlement-and-weights.md`.

On 2026-09-05 a retained PASS pair on `moe.fused_experts_reduce` was found to
credit +13.25% while its candidate lane ran at 2289–2307 tok/s, inside the
2277–2333 tok/s band every other retained candidate produced that day, and
slower than the fastest of them. Both halves had read the incumbent lane at
2022–2031 tok/s against a 2224–2290 tok/s band for the same incumbent artifact
across the other ten retained halves. The slow baseline state has appeared in 9
of 64 retained baseline-role reads since the champion baseline began (August
19 record: 8.5% of halves); the two-PASS minimum absorbed every case except
that pair, where both halves drew it. Under v7 a clear PASS at the B/C precheck
never reads B′, so the check that could have caught it is skipped exactly when
the baseline is wrong, and no absolute band exists in `cacheon/eval`. The
operator command `chain-reopen-qualification` now returns such a pair to the
screen queue for a fresh pair against the current incumbent, gated on the
retained lane rates (`cacheon/chain/baseline_band.py`); the settlement-side
band gate and always-read B′ remain open work. The first live reopen exposed
the queue trap: the row still carried the service digest of the arena that
screened its old pair, so the queue backfill bound it to that retired stack
before its fresh screen ran, and the evaluator held on
`baseline_commission_required` with newer submissions waiting behind it. A
reopened row now clears that digest and binds to the stack whose service
re-screens it; a second run of the command repairs a row already caught.

### Routing-only resident screen

The abbreviated-serving stage may keep a stock engine resident and hot-swap a
bounded candidate queue. Each swap is generation-bound, triggers graph
recapture, and is checked by shared stock brackets, contamination canaries, and
exact all-rank fired/completed evidence before promotion.
The calibrated screen policy is retained by the arena provider. Graph-enabled
B300 screen engines set SGLang `watchdog_timeout=1800` so the default 300s
scheduler watchdog cannot SIGKILL ranks mid CUDA-graph capture on the live
resident loop (observed on netuid-14 FIFO recommission 2026-08-04 as
`outer_oci_client_returncode=137`). The host `resident-intake` root is mode
`0711` so the non-root OCI runtime user can traverse to a content-addressed
digest under the read-only swap mount; `0700` left digests unreachable and
failed closed as `staged swap bundle is absent or writable` after capture
completed. After that fix, live swap still requires the sealed worker image's
installed `cacheon.manifest` to accept the pre-cutover
`optima-op-abi-v0` spelling (`fe55be1` `_PRE_CUTOVER_ABI_SHA256`); the
standing `13c72417` image (built at `77fae0ec`) does not, and an image-only
rollout cannot pass sealed primary-authority worker/preflight pins. On
2026-08-04 the sealed primary-authority worker/preflight was rebound to
derived image `cfc0c7a3660b…` (harness with `_PRE_CUTOVER_ABI_SHA256`); epoch
`4d3df000…` then completed §10 two durable FIFO screens with
`adapter_start_count=1` (screening only; not green).

Direct AOT artifacts, dependency patches, native rebuilds, and setup hooks are
not safely hot-swappable. They receive a typed screen waiver and proceed to
dedicated qualification. A waiver and a screen promotion are routing products,
not qualification evidence.

### Candidate time budget in the resident screen (2026-09-02)

Each candidate read in the resident screen is bounded by
`max(300 s, 10 × the latest stock read)` on the same engine and prompts
(`ScreenPolicy.candidate_time_multiple`, `candidate_time_floor_s`). A read
that outlives its budget is raised as a candidate failure and graded as a
terminal screen FAIL with a receipt reason starting `candidate_timeout:`.
The session's absolute batch timeout is unchanged and still classifies as
infrastructure when it fires outside a candidate read. Motivation: on
2026-09-02 a mainnet `moe.fused_experts` bundle whose prefill path was a
per-expert Python dequantization loop ran for the whole 1800 s batch timeout
twice, surfaced as `remote_screen_infrastructure`, re-held, and burned the
release cap; four ranks were stack-dumped mid-read to attribute it. Proven on
the mainnet pod the same day under release `3d27775e`: that bundle reached a
terminal FAIL in 10 min 21 s with reason
`candidate_timeout: candidate read exceeded 364s (10x the 36.4s stock read,
floor 300s)`, and the standing owner moved to the next reservation.

### Isolated eager screen mode deleted (2026-09-02)

`B300BuildABIGraphScreenAdapter` no longer carries an `isolated` execution
mode. The per-candidate eager and graph OCI sessions, their slot-audit witness
and graph observation evidence, and the eager half of `B300ScreenExecutionPlan`
were deleted; the ABI and graph rows are always resident carrier deferrals.
No commissioned deployment had selected the isolated mode since the resident
lane was registered. The coordinator identity digest keeps the literal
`execution_mode: resident`, so sealed screen deployments replay unchanged.

### One workload authority (2026-08-21)

Until 2026-08-21 the manifest declared a hand-written two-regime workload
mixture (decode 256-token/concurrency-32 plus long-prefill 8,192-token) that
no execution path consumed, while the scored session actually ran the sealed
corpus's short prompts (40–230 actual input tokens, median 74) at 256 output
tokens. Retained mainnet evidence surfaced the split on 2026-08-21. Current
source consumes one or more exact cells directly from the sealed prompt
authority. `prompt_batch_cells` binds every batch to its engine-observed input
tokens, output budget, concurrency, and timed-read count; the session sends that
request-local geometry, and commissioning rejects missing, extra, or reordered
cell coverage. Engine context length and admission width derive from the full
cell set. Single-cell authorities preserve their historical workload digest;
mixed authorities bind the ordered geometry in v3. Each read's evidence carries
the engine-observed prompt token count per request, validated against its own
cell at the protocol boundary. A mismatch or missing count is infrastructure,
never a candidate verdict.

### Sealed arena target set (updated 2026-08-30)

The commissioning input names a sorted `registered_targets` set. The evaluator
validates it against the cross-arena catalog and seals its complete complement
as `ArenaServiceManifest.closed_targets`; there is no import-time MiniMax-M3
target constant. The same sealed input now supplies the model-profile key and
model/runtime engine settings, while per-cell and per-target launch fields are
derived by the evaluator. Intake parks a proposal for a closed target at the
fingerprint step with reason
`target_unavailable:<target>` and decision `NO_DECISION` — replay never
echoes it — and releases the cited eval-cost payment pointer or admission
credit in the same transaction. Before this, a closed family could only be
marked in documentation while intake charged and evaluated against a
workload that could not resolve it.

### Graded speed-failure reasons (2026-08-21)

Until 2026-08-21 every speed FAIL was published as `speed_regression`,
including in-band misses: retained report 162dee6a recorded a 1.00338×
speedup against a 1.01024× bar — a 0.685-point miss inside the round's noise
band — as a regression. Current source grades the reason with the verdict:
`candidate_slower` requires the speedup below the mirrored bound 1 − u for a
bar of 1 + u, or a directly measured conditioning regression;
`speed_threshold_not_met` covers the band. The reason is produced by the
witness graders and carried through stage exits, reports, and miner feedback;
report composition refuses a speed FAIL that arrives without it. Reports and
stage exits settled earlier revalidate under their retained coarse code.

### Resident adaptive qualification

Since **2026-08-16** resident speed policy **version 7** swaps both arms. Under
version 6 and earlier only the candidate lane took a swap, which handed the
candidate role a measured advantage on identical work: a bundle audited
`aot_invoked:0` read 0.9–2.7% fast in the C role across six runs and both
physical orientations, of which position explained 0.117% and the physical lane
none. Under version 7 the baseline takes a stock-to-stock swap of its own, so
neither role is measured unswapped. The swap also reports the per-rank execution
count for the generation it closes. That count remained diagnostic-only until
2026-08-21, when retained mainnet evidence showed a candidate at 0/4 ranks had
still received a speed verdict. Current source unconditionally holds the leg
unless every rank completed the candidate under exactly the activation
generation. Unobserved or incomplete evidence is an infrastructure
HOLD/non-verdict, never candidate PASS or FAIL. Before version 7 the resident
lane emitted no execution evidence at all — it is launched stock, so the
one-shot driver's `active`-gated receipt directory was never created for it,
and registration was the only thing the crossover could observe.

Since **2026-08-23** (source on this branch, not yet deployed) the closing swap
carries each rank's receipt rows, not only their count: per registered slot,
the call count, whether any call happened inside a CUDA-graph capture, the
exception a raising entry left behind, and the routing reasons for calls that
went to stock. The host reduces the rows itself; a rank counts as executed only
when it loaded, raised nothing, and was captured on every slot SGLang serves
from its graph (a slot whose SlotSpec declares an eager serving seam is exempt),
so a candidate called only during eager warmup can no longer pass the guard.
The rows are published by the worker as the unsealed `qualification.execution`
artifact, travel in the product, and are rendered by `chain-miner-report
--evidence-root` and `explain`. A hold raised by the guard names the faulting
ranks. Retained mainnet evidence for this path is the four lane-A rank lines of
2026-08-23 (msa_block_score, 1,140 calls per rank, captured), read back from
the container log and used as the fixture; no run has yet produced the
artifact on the pod.

Since **2026-08-18** resident speed policy **version 8** serves candidates that
cannot be hot-swapped into a loaded engine. A bundle declaring CUDA, C++ or PTX
sources, AOT artifacts, dependency patches, or engine setup is routed to the
two-process crossover, which launches its own baseline and candidate engines.
That substrate refused every version-6 and later policy outright from
**2026-08-15** (`87944430`) until this change: the conditional-bookend schedule
landed on the pair-native path and the two-process path was left asserting the
older one. Affected candidates screened `promote` and then received no speed
verdict at all — measured on 2026-08-18, bundles declaring CUDA sources took 295
screen attempts for 3 verdicts (1%) against 580 attempts for 110 verdicts (19%)
for the rest, with 30 such bundles cycling through requeue at that point.

Version 8 reads B, C and B′ — always three, never more. B′ is precommitted
rather than earned by a close call because the quality gate harvests its
stock-drift control from the second baseline read
(`reference_quality.stock_drift_upper_bound` is its only consumer; the
candidate-versus-baseline comparison discards it), so a conditional bookend
would leave a clear PASS with no control. An unconditional read also preserves
the anti-reroll property versions 6 and 7 enforce. C′ and B″ do not exist under
version 8, and versions 6 and 7 are refused on this substrate rather than
silently producing evidence the quality stage cannot use.

The version is selected per candidate when the qualification plan is built, from
the same swappability predicate the worker routes execution on. The commissioned
provider policy is unchanged and continues to serve every swappable candidate;
only the read order differs, and every calibrated threshold is the sealed one.

Mixed-cell arenas use resident speed policy **version 9** on the same
two-process B/C/B′ substrate. A median of heterogeneous per-batch rates would
erase the minority cell, so v9 grades total timed output tokens over the full
host-observed mixture makespan while retaining the individual windows as raw
evidence. Hidden quality samples the full sealed prompt pool, including long
requests. Each selected prompt retains its complete B/C/B′ output length; the
quality profile bounds the largest declared output budget. Mixed-length pristine
reference requests use ORQ2/ORE2 frames with per-prompt lengths. Uniform requests
retain their ORQ1/ORE1 bytes, and single-cell M3 authorities retain their prior
policy and digest shape. Selection remains a random sample, so a small sample
does not guarantee that every workload cell appears in each attempt.

Since **2026-08-10** measurement-reuse identity is controller-blind: the
calibration context binds `ReferenceManifest.measured_digest` and no longer
carries a controller distribution digest, and raw quality bindings match the
same measured reference identity (`EngineLaunchSpec` exposes the analogous
`measured_digest`). Full manifest and launch digests remain the provenance
record — pristine T-session witnesses and per-run receipts still pin the
exact controller — but sealed calibration and durable measurement
authorities now survive controller-code revisions instead of being
invalidated by every commit. Calibration packages sealed under the earlier
eleven-field context shape do not parse under this contract and are resealed
from their durable inputs, not re-measured.

Since **2026-08-21**, commissioning refuses a resident-speed `min_windows`
floor above the workload's total timed reads: an unsatisfiable evidence floor
is a commissioning error, not a runtime refusal discovered after a measured
run. A proposed policy-only calibration relabel was removed before deployment:
the nested reference identity still changed with the arena, and downstream
quality binding required exact context equality. Fast recommissioning must
make the fresh durable seal inexpensive instead of weakening that identity.

Providers commissioned on **2026-08-10** sealed resident speed policy
**version 4**: every timed read is graded (the version-3 mid-run
window-scatter refusal is retired as verdict control flow), window scatter is
carried as recorded fitness evidence under an advisory bound, and the speed
verdict is decided by bookend invariance — a candidate fails when it loses
against its most favorable bookend, passes when it clears the requirement
against its least favorable one, and an undecidable spread terminates as
`FAIL` (`valid_not_faster`) because a crown requires demonstrated
improvement. Retained version-1..3 evidence regrades under its own sealed
arithmetic; cross-version splicing remains refused. Later policies retain the
same versioned witness family but change the permitted read schedule.

Since **2026-08-10** the sealable range also includes resident speed policy
**version 5**, which adds the bracket-drift ruling: when the flanking
baseline brackets disagree beyond the sealed noise ceiling, the earliest
bracket is the only comparison baseline — the drifted later brackets are
excluded — and the candidate is graded against B alone under version 4's
terminating arithmetic. Bracket drift therefore resolves to a decision at
the initial grade instead of escalating or re-queueing. Sealed version-4
evidence continues to regrade under bookend invariance without the
exclusion. The 2026-08-10 commission selected version 5 prospectively at that
time; current swappable work selects v7, single-cell non-swappable work selects
v8, and mixed-cell work selects v9. Already sealed providers and evidence keep
their original policy identity.

Production providers previously selected qualification policy version 3:

1. two isolated resident TP lanes are assigned incumbent and candidate roles;
2. speed begins with B/C/B′;
3. borderline evidence adds C′/B″ under the frozen escalation rule;
4. a registered sampled slot audit runs in a separate eager, untimed candidate
   role and is regraded by the host;
5. candidate lifetimes are destroyed before candidate-free pristine T grades
   the sealed trajectory; and
6. an eligible reproduction exchanges the physical incumbent and candidate
   lane roles.

Version 1 and version 2 evidence remain readable for historical compatibility.
Screen measurements do not enter this authority. Candidate-attributable
failure can produce `FAIL`; infrastructure, drift, missing evidence, or broken
authority produces `NO_DECISION`.

From 2026-08-29 the qualification commission constructs its measured baseline
from the durable incumbent stack the qualification-capabilities factory
declares (`incumbent_entries`; empty at genesis, where it reproduces the
stock tree exactly), while pristine T remains the empty stock stack and the
resident screen keeps its stock baseline. The change followed the first
production crown (2026-08-28), which advanced the durable evaluation stack
while the worker could only construct a stock baseline. The former
empty-incumbent resolver mode is removed; an empty closed resolver is the
genesis authority and fails closed on any lookup. Commission registrations
sealed before this revision do not replay against it; a fresh commission is
required.

Also from 2026-08-29, the declared baseline is realized per schedule rather
than by booting champion trees into the pair lanes: v7 pair-native engines
boot plain stock and the baseline read injects the sealed incumbent bundle
through the swap path (identity sealed from the stack entry and its
resolver-verified manifest, never from a swap acknowledgement; execution must
be proven per rank or the read holds), while the version-8 two-process
schedule boots the materialized incumbent tree for its baseline process. v7
serves hot-swappable candidates replacing the incumbent's registered target
(or genesis); everything else routes to v8, and the worker routes on the
sealed plan version. The retired pre-v6 pair-native grading branches were
deleted in the same change; the pair-native substrate now refuses speed
policy versions below 6 at run and regrade.

From 2026-07-25 the resident speed policy of record was version 2: the scored
rate is the
steady-state timed window (`timed_tokens / timed_seconds`), conditioning stays
bounded by the sealed operational timing budget rather than entering the scored
rate, and the calibrated maximum baseline disagreement must be at most 2%.
Version-1 witnesses graded the charged rate and regrade only under their own
sealed arithmetic; the policy version is digest-bound and cross-version
splicing is refused. The motivating evidence is the 2026-07-24 stage-exit
described under empirical evidence, retained verbatim as the repository test
fixture `tests/fixtures/speed_stage_exit_45cbcc04.json`.

Policy version 3 is implemented, tested, and as of 2026-07-25 has produced its
first settled production program (described under empirical evidence): the
scored rate becomes the
median over per-batch timed windows retained in the witness rows, each read
carries a sealed window-scatter bound that refuses grading of an unfit
measurement everywhere (live and on reopen), version-3 timed reads
request no log-probability collection so evaluation work never shares the
clock with a speed measurement (`top_logprobs_num` 0 is now expressible
through the session protocol, worker, and binary evidence codec), and a
sealed conditioning slowdown bound fails a candidate whose unscored
conditioning span (the host-visible prefill surface) regresses past the
baseline's at equal warmth position — graded from spans already sealed in
every read, adding no measurement time. The companion quality mode
(decided 2026-07-25) is teacher-NLL-only: a zero top-k width in the
qualification profile and raw quality binding selects it end to end —
empty support rows through the reference protocol and worker, explicit
null distribution/KL evidence with uniformity enforced at every layer,
and typed refusal of any threshold policy that names a distribution
metric against it. Distribution-level numerics coverage remains with the
in-engine slot audit stage. The earlier standalone instrument authority
(`box_certificate`, sealed per-session stock-vs-stock null floors with
double-bounded expiry) was retired 2026-08-08 without ever gaining a
production caller; instrument validity is owned by the version-3
resident-pair speed policy and calibration path. Overnight 2026-07-24/25
measurement context: two version-2 joined primaries passed clearly while
the timed noise floor of the box deteriorated 0.72% to 3.09% across the
night and the final calibration honestly refused; version 3 is the
structural response.

The audit gate is Torch-free, checks exact slot × TP-rank/process coverage, and
canonicalizes floating-point facts before durable receipt identity. Audit is
authoritative only when the frozen plan registers the matching requirement.

### Settlement

Settlement requires two complete PASS attempts over the same economic identity
with distinct authority/evidence commitments and, for version-3 resident-family
evidence including current v7/v8 witnesses, the required physical-lane role
swap. It uses the lower accepted speedup, reopens exact
evidence, and commits the candidate disposition, hash-chained events, claims,
and optional evaluation-stack transition transactionally.

Held reservations require a typed evidence-preserving disposition. Lease expiry
is not arena retirement, and the repository does not implement a generic typed
arena-retirement transition.

### Legacy V1 weights

`set-weights` is a separate signer control plane with intent-before-submit,
readback, pending, held, released, and confirmed states. It supports:

- signer-free dry-run and reconciliation;
- an all-uncrowned bootstrap projection to a registered `--burn-hotkey`;
- a chain-resolved `--burn-to-subnet-owner` bootstrap that resolves one
  metagraph UID owned by the subnet owner coldkey (prefer `SubnetOwnerHotkey`,
  else lowest UID) and publishes `{owner: 1.0}` through the durable
  intent/pending/confirmed journal with `require_current_crown=False`,
  refusing a foreign in-flight journal head (2026-07-30: restored journal
  routing and settlement-state refusal after review of the interim
  journal-free shape);
- stable-UID finality catch-up when authority and weighted-recipient mappings
  remain unchanged; and
- continuous `--watch` operation with bounded retry for retryable transport /
  submit / readback faults.

The standing CPU supervisor can now enable a wallet-free `weights` stage from a
second sealed config. At each refresh it reads finalized authority and pushes
either the live V1 offer or the explicitly configured crownless burn offer to
the gateway; it never signs. Configuration must name `enable_weights` and
`weights_stage_config` consistently or startup fails closed.

The current `cacheon.emissions.v1.5` projection rewards every distinct retained
two-PASS contribution, including settlement holds. Credit uses logarithmic
speedup, submission-time stall bonus, and exponential decay; a later crown does
not erase or rerun an earlier PASS. Matching v1.1/v1.3/v1.4 policy bindings
advance, while any numeric policy change remains refused. This restores the
two-PASS rule after an interim CROWN-only deployment on 2026-09-03.
The follower also resumes its retained in-flight projection before adopting a
fresh offer, using the same existing recovery helper as `set-weights`.

A stack transition does not invalidate, archive, or requeue completed
qualification evidence. The old resident drains the contiguous FIFO segment
already assigned to it; screening may continue because it is
baseline-independent. Qualification holds for manual recommissioning only when
the queue cursor reaches a segment assigned to a different stack. A retained
remote qualification product that differs from the live commissioned stack is
released through a digest-bound recovery transition instead of being imported.

When a valid active claimant is absent from the current metagraph, that
family's allocated ppm is sent to the validator hotkey for the tick rather than
holding unrelated families. The claim remains active and resumes at its
then-current decayed share if the hotkey returns. Stale, incompatible, missing,
or unreopenable evidence still holds the complete projection.

Both burn bootstraps become invalid when a claim, crown, or active V2
composition exists: the projection builders refuse, the refusal is a
nonretryable publication fault, and a `--watch` loop stops with a typed error
rather than overwriting settlement vectors. Settlement confirmation still
requires exact recipient/value/`last_update` readback (SDK success alone is
not enough).

### Shared current-weight distribution

The implementation separates evaluator, gateway, and chain signer:

- `push-weight-offer` builds a legacy or debt/composition
  `CurrentWeightOffer` without opening a chain-signing wallet and pushes it
  with a rotatable timestamped HMAC credential;
- push-enabled `serve-weights` retains a second HMAC envelope over the
  credential id and exact offer digest, verifies it before signing a response,
  and serves only a caller with a live validator permit; and
- `follow-weights` pins the response authority, rebinds the follower signer,
  bounds initial projection age by the refresh cadence, verifies stable UIDs,
  and publishes through the normal readback reconciler using a dedicated
  signer-only journal that accepts both V1 and V2 offers.

The signer resolves the subnet's live `WeightsVersionKey` and passes it to the
chain call; it does not rely on the SDK's default. Chain connections may use a
primary WebSocket endpoint plus explicit archive/fallback endpoints, with
reconnect between watch passes while one healthy client is retained instead of
recreated on every tick.

Current-offer persistence rejects effective-block rollback, same-block
equivocation, and V2-to-V1 regression. The combined legacy `set-weights`
object-store path advances only after a live `pending` or `confirmed`
reconciliation and completes synchronously under the exclusive publication
journal lock. Dry-run, reconcile-only, and held paths do not externalize a new
offer.

These are implemented protocol and durability controls. Push-disabled gateway
mode deliberately trusts configured raw storage. A valid old authenticated
envelope can still be replayed within the bounded follower freshness window,
and storage, gateway, push-secret, response-hotkey, signer-wallet, and host-root
availability/custody remain deployment responsibilities.

### Retired the MiniMax-M3 model-side MoE seam (2026-09-05)

The `moe_reduce` seam row bound `moe.fused_experts_reduce` to
`MiniMaxM3MoE.forward_normal`, and its two-sided completion handshake (the
model-side wrapper in `integrations/sglang_moe.py` and the reduce-owner marker
protocol in `dispatch.py`) was the only model-specific code in the seam table.
With the M3 arena formally retired, both sides were deleted together, along
with the `requires` skip field on `SeamAdapter` that existed only for that
row, the importerless `integrations/_by_value_function.py` helper, and the
M3 availability notices in the miner guide. The `moe.fused_experts_reduce`
contract stays registered in the catalog, since its bytes are part of every
recorded catalog digest, but no arena binds it: a reduce-owning MoE kernel
needs the serving model's outer reduction suppressed, and that hook belongs
to the arena that registers the target. The change is inert on the GLM-5.3
arena, whose registered set never included the target and whose seam-binding
derivation is identical without the row; the deployed M3 lane runs from its
own pinned source tree and is untouched.

### Removed reviewed-release manifest lane (2026-09-05)

On **2026-09-05** the reviewed-release manifest lane in `stack_manifest.py` —
the engine release manifest, the release context, the integration review record
and its artifact references, and the integrated contribution reference — was
removed together with the integrated-source branches in the engine tree
materializer, the stack planner, the marginal runtime, the sealed commission,
and the closed source resolver. Nothing in the tree constructed a review record,
no retained manifest ever carried an integrated entry, and the signed release
product those types fed had already been removed on 2026-08-19. The closed
source resolver keeps its two-argument call shape and its recorded digest bytes
because the sealed qualification packets construct it; the second argument must
now be empty and goes when the packet template stops passing it.

### Removed discovery lane (2026-08-19)

On **2026-08-19** the fenced discovery lane — the separate discovery proposal
ABI, overlay build/activation, discovery arm qualification, and every
discovery branch in intake, settlement, OCI session, and qualification code
(`discovery.py`, `discovery_overlay.py`, and the lane's tests and guide) — was
removed. It never admitted a production proposal: the live store holds zero
discovery reservations or claims. Work that does not fit a registered target
is now refused at resolution. Durable shapes are preserved unchanged: the
settlement `lane` field (value set narrowed to `registered`), the submitted-delta
`product_kind` (narrowed to `component`), the always-`None` session-plan key
`expected_discovery_overlay_identity_digest` in its digest domains, the
`DISCOVERY_BOUNTY` settlement event vocabulary, and the legacy V1
discovery-bounty economics and tables, which stay fenced V1 schema. Complete
implementation remains in Git history at the parent of the removing commit.

### Inactive V2 finite debt

On **2026-08-09** the V2 finite-debt economics implementation — finite
registered-CROWN debt, the reviewed-discovery bounty class, campaign and
composition policies, the wallet-free activation command, and
`set-debt-weights` publication — was extracted from the tree without ever
being activated. No live V2 activation or publication receipt ever existed.
The design intent is retained (a bounded post-activation claim paid down over
confirmed epochs) and the complete implementation is recoverable from Git
history at [`dc158fb4`](https://github.com/latent-to/cacheon/commit/dc158fb4).

One durable-compatibility artifact remains in the tree: the shared-weight
offer wire schema keeps its `lane`/`debt_binding` fields with `lane`
restricted to `legacy_v1` and any debt-lane payload rejected, so historical
stored offers reopen byte-identically. The reserved schema-4/5/6 migrations
and V2 table DDL were retired on **2026-09-05**: no reader of those tables
existed anywhere in the tree. Intake keeps accepting metadata stamps 3 through
6, so a database created before that date opens unchanged with its 108 V2
objects untouched (16 tables, 32 triggers, 5 named indexes and 55 automatic
primary-key indexes; never dropped), while a fresh database now holds 68
schema objects at stamp 3 instead of 176 at stamp 5. A validator archive
records whichever shape it captured: archives of existing databases are
byte-identical to before, and an archive of a fresh database now carries stamp
3 and 16 fewer table counts. A stamp-3 database later opened by a controller
built before this date gains the V2 objects and stamp 5; both shapes open under
either build.

Reintroducing V2 is a new reviewed change, not a revert switch.

### Engine release

Evaluation and serving remain separate products. The release model includes
reviewed integration records, sealed model/native identities, deterministic
source/wheel products, SBOM/provenance, Ed25519 signatures, OCI context, host
policy, registry types, and serving receipts.

Current release authority is incomplete:

- the serving wheel does not close every manifest runtime import;
- builder output, effective runtime arguments, management-route policy, and
  complete release/session receipt binding still require end-to-end closure;
  and
- no final deterministic registry pair, authorized image, or complete all-rank
  serving receipt set is claimed for this revision.

Loading sealed native artifacts inside evaluation OCI proves evaluation
runtime support. It does not close the serving release.

## Empirical evidence

### Unattended mainnet loop and PR #95 merge boundary (2026-08-15–19)

By 2026-08-18 the deployed, frozen source cohort on netuid 14 had exercised the
ordinary finalized path through intake, remote screening, commissioned remote
qualification, independent reproduction, and transactional settlement without a
human manually importing each evaluator result. The retained database contained
two mechanically crowned rows and four settlement holds. The two crowned targets
and sealed pair values were:

| Target | Primary / reproduction speedups | Stored settlement speedup |
|---|---:|---:|
| `activation.silu_and_mul` | 1.0224 / 1.1465 | 1.0224 |
| `collective.moe_finalize_ar_rmsnorm` | 1.0178 / 1.0075 | 1.0075 |

This establishes that the live chain-ordered control plane could carry real work
through the terminal settlement transaction and retain both attempt artifacts. It
does **not** upgrade those rows into an uncontested performance claim. A post-hoc
review of the sealed reads identified boot-state/lane anomalies in the small-kernel
measurements, including a same-bytes near-null reading for the first target;
subsequent analyses disagreed on whether the large pair spread was a physical-lane
bias or a transient engine state. The pair-native per-generation execution guard
was also record-only. The stored settlement events remain historical facts, while
their performance interpretation remains explicitly unresolved and the expensive
artifacts must be preserved rather than rerun or silently regraded.

The same 2026-08-18 snapshot did not show a complete weight-publication outcome:
the chain still carried the crownless burn vector and the follower lane was not
healthy. Later PR #95 commits added the standing wallet-free weight-offer stage,
live `WeightsVersionKey` stamping, endpoint failover, and follower reconnect
behavior; those code changes are implementation evidence, not retroactive proof
that the earlier chain vector changed.

PR #95 then merged 180 commits as `d94713a7` on 2026-08-19 after all required
checks were non-failing at head `53aaff30`. At merge time the live VM and pod still ran explicit
frozen source directories; merging `main` did not perform a deployment cutover.
Accordingly, this ledger makes no claim that merge commit `d94713a7` itself has
executed a mainnet GPU qualification or weight publication.

### Retained B200 qualification and settlement

The strongest retained production-shaped crown evidence predating the resident
version-3 path is a TP4 joined block-score qualification on an 8×B200 host:

| Field | Primary | Reproduction |
|---|---:|---:|
| Charged-basis speedup | 1.0561× | 1.0487× |
| Timed-section diagnostic | 1.0932× | 1.0866× |
| Decision | PASS | PASS |

Settlement used the lower value, 1.0487×, committed a generation-1 crown, and
reopened successfully after restart. The attempts had distinct required
authority/evidence digests. They do not prove distinct operators or failure
domains.

This evidence used an earlier SM100 worker image and SGLang source build
`0.0.0.dev1+g56e290315`, not the repository pin `0.5.13.post1`. It predates
the resident version-3 schedule and current audit transport. It therefore
demonstrates the earlier bound qualification/settlement mechanism, not a
current-revision production canary.

### MiniMax-M3 fused-epilogue evidence

Earlier 4×B300 runs measured shallow and deep fused-epilogue submissions through
the historical referee and later through the testnet intake loop. Those runs
remain mechanism and performance evidence for their exact runtime, policy, and
hardware. They are not retroactive current-schema crowns and do not qualify the
resident version-3 path. See the
[MiniMax-M3 evidence note](../results/minimax-m3.md).

### Audit-path canary status on 2026-07-19

A bounded 4×B300 run on 2026-07-19 did not satisfy the current launch gate. The
sabotage control was rejected. The honest primary produced no verdict after
concurrent legs shared an executor label and invalidated quiescence authority.
The honest reproduction passed graph and pristine-T quality, but its deep slot
had only four audited calls per rank against the required 32 and its speed gate
failed at 1.005507×. Zero observed audit comparison violations did not repair
insufficient coverage.

This is retained failure evidence. It is not an activation canary, PASS, or
performance authority. At that point the subsequent resident screen and
two-lane adaptive qualification implementation were test-covered and informed
by GPU calibration, but no retained end-to-end version-3
primary/reproduction canary existed. Later mainnet attempts are recorded in the
2026-08-15–19 section above with their separate unresolved limitations; they do
not rewrite this failed July attempt.

### First production version-3 speed verdict (2026-07-24)

A 4×B300 joined primary on 2026-07-24 executed the resident version-3 path end
to end for the first time: prepare, intake hygiene, graph, and the slot audit
all passed, and the speed stage produced the first production version-3 speed
verdict. That verdict was `FAIL speed_regression` under version-1 charged-basis
arithmetic: the second baseline read ran as a warm continuation of the first
baseline session while the candidate read ran cold, so the conditioning-
inclusive scored rate turned a positional split into 6.3% apparent baseline
noise and a 1.126 required bar, while the candidate was faster than both
baseline reads on every timed window. The sealed stage-exit is retained
verbatim as `tests/fixtures/speed_stage_exit_45cbcc04.json`, and a regression
test pins both readings: version-1 arithmetic reproduces the shipped verdict
exactly, and version-2 timed-basis arithmetic grades the same sealed reads as a
clear pass. The reservation's terminal disposition stands; no verdict was
altered after the fact, and any future attempt requires a fresh submission
under the version-2 policy.

### First settled crown under speed policy version 3 (2026-07-25)

A 4×B300 program on 2026-07-25 ran the resident version-3 policy through every
production phase for the first time and settled the first crown: intake, graph
verification, controller snapshot, two per-lane calibrations, a joined
primary, a joined lane-swapped reproduction, a restart proof, and a signer-free
weight projection all passed in one continuous program against testnet
finalized intake (reservation
`c7713892…`, target `collective.ar_residual_rmsnorm`, candidate content
`747405b4…`, source revision `07c032ed`). The calibrations sealed per-lane
timed noise floors of 0.30% (primary lane) and 0.125% (reproduction lane)
against the 2% ceiling, with negative, positive, and stock controls sealed
before any timed arm was observed. The joined primary graded a timed speedup
of 1.0212 and the reproduction, with candidate and baseline physically
swapped across lanes, graded 1.0278; baseline bracket disagreement inside the
timed phases was 0.21% and 0.03% respectively. Settlement bound the pair and
accepted the lower speedup, 1.0212. This was also the first production
execution of the teacher-NLL-only quality mode (zero top-k width) and of the
joined reproduction, restart, and weights orchestration. The evidence was
relocated to validator-owned storage with every retained artifact re-verified
against its bound content digest, and a signer-free weight projection built
from the settled state (crown count 1, full pool to the crowned hotkey).
This establishes measurement and settlement under the version-3 policy. It
does not establish on-chain weight publication (the signed submission is a
separate, operator-reviewed act), incentive activation, integration review, or
serving readiness.

### Public object-storage intake canary (2026-07-27)

The exact fused-epilogue proposal from the version-3 program (content hash
`747405b41845506800939507a93b6011d38f5a94e69a5ec303a3d39a48e77709`)
was packaged and uploaded to a miner-side Hippius S3-compatible bucket under
the content-addressed key
`cacheon/miner-bundles/sha256/747405b41845506800939507a93b6011d38f5a94e69a5ec303a3d39a48e77709.tar.gz`.
An anonymous download from the resulting public HTTPS URL was byte-identical
to the 24,012-byte stored archive (archive SHA-256
`d86162982a72b66bed39751686cfdced15a2e25518a39ead61f8eb57f8533d7f`).
The production hardened fetcher then downloaded it without credentials,
enforced the archive policy, extracted it into a private mode-0700 cache, and
re-derived the exact committed tree hash.

A subsequent synthetic-finalized intake pass used the same public URL through
the unmodified validator loop. It durably reserved one arrival in a fresh
`FinalizedIntakeStore` SQLite database, advanced it to `published`, and created
a mode-0555 worker publication with digest
`765778bef17a1d6a6c5b3f93bcb46b1db48c2530306e60c4ff76980398131673`.

This establishes the external miner-origin → anonymous validator-fetch
transport and URL → SQLite → worker-publication intake plumbing for those exact
bytes. The finalized snapshot in the intake canary was synthetic; this is not a
new chain submission, crown, qualification, object immutability guarantee, or
validator database service. S3 objects remain mutable at the hosting layer; the
finalized content hash, not the URL or object metadata, authenticates proposal
identity.

### Private validator recovery canaries (2026-07-27)

A provider-neutral `chain-snapshot` invocation used generic S3 configuration
with the Hippius endpoint, never a provider-specific archive implementation.
The input was a fresh synthetic-finalized intake of the public fused-epilogue
bundle above plus one retained qualification artifact, one redacted chain-audit
record, and one harmless explicitly sealed policy file. The first canary stored
11 new blobs representing 591,709 source bytes below
`cacheon/validator-archive/v1/canary/20260727-provider-neutral`. Its manifest
digest was
`4148e6a3815f557345fd01004b1a88313c840512182c8d495a131c78983d62fa`
and its online SQLite backup digest was
`f59e65f1647144d8ba1a3050d414fd8c482e50650f9caedfb4c38230833f1424`.
Every object reopened after upload. A fresh retained restore then passed SQLite
integrity/foreign-key checks, reopened the worker publication and qualification
artifact, validated the journal, and emitted the closed restore map. Anonymous
HTTPS access to the manifest returned 403.

The current worktree was then deployed through the validator's rsync setup path
to an idle eight-B200 pod. The host deployment tree and container `/cacheon` both
matched local runtime-source aggregate SHA-256
`dd11ab2d8f40f586a7f9661871c68ce6480cec6b63e7ea0190eca6a7ac1c59f8`;
the sync excluded `.env` and the private worklog, and the spawned-interpreter
bootstrap resolved Cacheon from `/cacheon`. Without starting a GPU process, the
pod independently repeated anonymous HTTPS intake, uploaded another 11-blob
snapshot under `cacheon/validator-archive/v1/canary/pod-b200-20260727`, and
semantically restored it from both the isolated source copy and the final
`/cacheon` deployment. That manifest and database digests were respectively
`0e856820f37c1031407afda701591a443c1f0866ae4c0663c461118d2e0bba74`
and
`2beaafbd28fdc337f9e1cd28a3bffa183fb017e0f4a0f9790dcfaaa1aa91589e`;
its anonymous manifest request also returned 403. The temporary pod credential
file was removed after verification.

These canaries establish consistent snapshot, private object transport,
bounded authenticated download, and fresh-root semantic restore for those
exact synthetic records. They do not establish bucket encryption,
versioning/object lock, long-term retention, production evidence completeness,
chain finality for the synthetic reveal, a production restore cutover, GPU
qualification, or a new B/C/B′ verdict.

### Transported crown and first signed crown publication (2026-07-30/31)

A 4×B300 program on 2026-07-30/31 ran the complete production loop for the
first time with the proposal delivered through the public object-storage
transport under a real finalized chain commitment: miner-side `chain-publish`
of the fused-epilogue proposal (content `747405b4…`), a fresh on-chain
commit-reveal carrying the public HTTPS URL, anonymous hardened fetch,
finalized intake into a from-genesis database (reservation `1089ffc0…`),
graph verification, controller snapshot, two per-lane calibrations, a joined
primary (timed speedup 1.0327), a joined lane-swapped reproduction (1.0237),
restart and signer-free weights phases, and settlement at the lower accepted
speedup 1.0237 for `collective.ar_residual_rmsnorm` (crown reason
`qualified_win`). Measured post-intake wall time was approximately 2 h 47 min:
graph verification 9.5 min, calibrations ≈38 min per lane, joined primary
39.2 min, joined reproduction 39.5 min, prepare/restart/weights under one
minute each.

While the program ran, the journaled `--burn-to-subnet-owner` watch operated
against the crownless mirror with real signed publications. The initial
publication and two refresh-boundary re-publications all landed on chain; both
refreshes were left `pending` by SDK result timeouts and were finished through
signer-free `--reconcile-only` authoritative readback. Each pending head also
stopped the watch loop at the next tick because the block-bound projection
digest no longer matched; this change set fixes that defect (the watch now
resumes its own in-flight head and still refuses a foreign one). After the
crowned database replaced the mirror, the watch refused with the typed
settlement-state error and exited — the designed crown yield.

The real crown weight publication was then signed and submitted for the first
time: commit-reveal submission at block 7675230, commit inclusion advancing
the validator's last-update row at 7675234, timelocked reveal applying the
crown vector, and exact authoritative readback confirming at block 7675528
(projection `7e869ecd…`, full pool to the crowned hotkey). A production
private recovery snapshot of the crowned mission was then published and
independently re-downloaded and semantically reopened (manifest
`1d0d9ebb…`); anonymous access to the manifest returned 403.

This establishes the end-to-end transported crown loop, the first real signed
crown weight publication with finalized readback, refresh-cycle burn operation
with typed recovery, and production recovery archiving of a crowned mission.
It does not establish mainnet economics, incentive activation, integration
review, serving readiness, object immutability at the hosting layer, or
unattended validator operation.

### Live shared-weight follower drill (2026-07-31)

A scoped testnet-307 drill exercised `push-weight-offer` → `serve-weights` →
`follow-weights` as three separate processes on one Mac, with the real private
object store between them. The evaluator-side process built the standing crown
offer without a wallet or bucket credentials. The gateway verified the HMAC
push, retained the signed offer at
`weights-service/testnet307/current_weights.json`, and served it through the
permit-gated signed response path. The follower pinned that gateway authority,
rebound the offer to its own signer, and advanced its dedicated journal through
intent and pending to confirmed at block 7680827; chain `LastUpdate` read back
the same block.

An idempotent second pass confirmed without another submission, and a
deliberately wrong expected authority produced the typed refusal. Because the
offer matched the standing vector, commit inclusion advanced `LastUpdate` and
the first readback already matched; the unchanged-vector case did not require a
second timelocked reveal. The drill also exposed two operator requirements:
select `--object-store-provider s3` explicitly for `serve-weights`, and create
the follower journal parent with mode 0700.

This establishes the live test-chain shared-weight path and its journal/readback
controls for those exact identities. It does not establish independent hosts or
failure domains, mainnet economics, V2 activation, or unattended operation.

### Hardware suite validation and containment tiers (2026-08-09)

The complete test suite ran twice on a two-B200 staging validator host
(driver 580.126.20, Python 3.12.3, torch 2.12.0+cu130, triton 3.7.0): once
beside a resident TP2 serving container holding roughly 168 GiB per device,
and once on empty devices. Both runs reported identical results — 2,496
passed, one failed, five environment skips, in roughly five minutes —
including the families hosted CI cannot execute: the two-device collective
and CUDA-graph contracts and the Linux `renameat2` publication cases. The
single failure was a test defect, not a product defect: an
unwritable-directory probe simulated with permission bits, which a root
euid overrides; that test now skips under root with the reason recorded.

The public CLI was exercised end to end on the same host as subprocesses:
`verify` accepted the pure-torch and Triton example bundles on CUDA with
graph capture and replay engaged, refused both deliberately broken bundles
with the verdict exit code, kept the infrastructure-error exit distinct
from the verdict exit for a Triton bundle on a CPU-only environment, and
verified the collective allreduce example across both devices. The OCI
containment flag vocabulary was exercised against the host's real
container daemon: network egress refused, read-only binds refused writes
with `EROFS`, a representative module-loading syscall refused, the
packaged seccomp profile accepted at container create, and the non-root
user identity enforced.

A seam-currency audit ran this repository's compatibility doctor inside
the launch-lineage staging worker image, against that image's SGLang
source build `0.0.0.dev1+g56e290315`: every registered seam-table
chokepoint bound, and every signature, engine-API, and server-argument
check passed. The single failure was the version pin `0.5.13.post1`
itself, which is the designed refusal for a non-pinned runtime.
Chokepoint presence and signature agreement are the compatibility gate's
own error boundary, not behavioral equivalence.

A live activation demonstration then ran three serving boots of a small
instruct model inside that image on one device, beside the resident TP2
server, all with this tree on the interpreter path: a null-armed
baseline, an armed run carrying the exact-math example bundle for
`activation.silu_and_mul`, and an armed run carrying its deliberately
broken variant. The image preloads `cacheon.bootstrap` in every
interpreter through its installed `.pth`, and the arming chain engaged
end to end: the exact-math bundle produced byte-identical temperature-0
output against the baseline, while the broken bundle visibly corrupted
the generated text — direct evidence that the armed kernel executes
inside the scheduler's serving path under CUDA graph capture. A
tensor-parallel variant repeated all three boots across both devices
(`--tp-size 2`) with the same verdicts: both scheduler ranks armed, the
exact-math bundle byte-identical, the broken bundle corrupting. Seam
activation emits no server log lines in this image, so behavioral
probes, not log inspection, are the working detector; and in this
image's SGLang build the plugin-framework loader has no call site in
the serving path, so activation rides the `.pth` bootstrap alone.

The chain-commitment content hash was then pinned as a cross-platform
golden: `content_hash` over every committed example bundle and stack
fixture produced byte-identical digests on macOS arm64 / CPython 3.11
and on the validator host's Linux x86_64 / CPython 3.12, and the matched
vectors are committed with a test that fails on any future divergence —
a consensus break or an unreviewed identity epoch, never a routine
refresh.

These behaviors are retained as standing tests
(`tests/test_cli_examples_e2e.py`, `tests/test_oci_live_container.py`,
`tests/test_golden_consensus_vectors.py`) that activate by capability
probe — CUDA device count and a usable container daemon — and skip
cleanly elsewhere, so hosted CI keeps the CPU, containment, and golden
tiers while the GPU tiers re-arm on any future validator host.

The live activation demonstration is additionally codified as a
repeatable, strictly opt-in tier (`tests/test_seam_activation_live.py`):
the same three boots — null-armed baseline, exact-math bundle, broken
bundle — with the byte-identity and corruption verdicts asserted. The
tier runs nowhere by default, including GPU hosts; it arms only under
`CACHEON_LIVE_SERVE_TESTS=1` with an operator-supplied worker image and
model, and once armed it fails loudly on a missing prerequisite rather
than skipping. It was validated armed on the staging host against the
launch-lineage worker image. The miner submission pipeline also gained
offline tests (`tests/test_submit_dry_run.py`): the dry-run path returns
before any chain object is touched, so the commitment round-trip, the
plaintext-HTTP refusal, and the 1024-byte chain cap are exercised with
no wallet and no subtensor.

The native `cutlass.cute.cubin.v1` toolchain gained its first live
proof (`tests/test_native_toolchain_live.py`, opt-in via
`CACHEON_LIVE_NATIVE_TESTS=1`): a deviceless container compiled a
minimal `@cute.jit` kernel under the validator compiler recipe on the
staging image's cutlass-dsl 4.5.2, and a GPU container accepted the
produced bytes through the production ELF gate and Driver-API
admission, resolving exactly the declared kernel. The lane's CPU tests
exercise its policy code against forged headers and a monkeypatched
compiler; this tier is the only place genuine compiler output meets the
gate and a device.

This establishes suite health, the tested verify/containment behaviors,
the seam-table currency of this tree against that staging image, the
live seam activation path on it at one and two devices — now
re-provable on demand — the native toolchain's compile and admission
path on genuine compiler output, the offline submission wire policy,
and the platform stability of the chain-commitment content hash. It
does not establish serving performance, the sealed prebuild protocol,
slot numeric contracts, qualification or settlement evidence, or any
crown claim.

### Incentive evidence

Historical V2 evidence — the deterministic one-campaign load study (semantic
report digest `b4de2350328a1bb8665cbcdf33f1256723023db662bf429cf80ed3343fb2b4b9`)
and the signer-free shadow projections against exact testnet membership —
was accounting-sensitivity and fixture evidence only. It never authorized
activation, and no live V2 activation or debit-confirming publication receipt
ever existed. The study, its fixtures, and the shadow tooling were extracted
with the V2 implementation on 2026-08-09 and remain in Git history at
[`dc158fb4`](https://github.com/latent-to/cacheon/commit/dc158fb4).

## Public CLI

The live command inventory is:

```text
slots  compat  chain-compat  scan  verify
chain-package  chain-publish  chain-eval-cost  chain-eval-cost-credit
chain-submit  chain-status  chain-register
chain-reservation-status  chain-miner-report  chain-validate
chain-snapshot  chain-snapshot-verify
chain-archive-schema3-hold  chain-evaluation-lease
model-provision  release-verify  release-context
set-weights
mint-push-credentials  mint-weight-gateway  push-weight-offer  serve-weights
follow-weights
```

The local miner loop is `scan` plus `verify`. Complete-engine performance and
quality authority begins with a deployment-injected arena provider.

### MiniMax-M3 plain-MoE NVFP4 contract smoke (2026-08-22)

Source `70bffb8f` on one B300, in sealed image `a7cbc41a` (SGLang
`g56e290315`, FlashInfer `0.6.12`, Torch `2.11.0+cu130`), passed BF16 eager
verification, CUDA capture, and three dynamic replays for the validator-owned
`nvfp4_layer` view (`cosine=1.00000`, maximum absolute error `5.969e-05`).
This is contract/graph evidence only: no finalized M3 layer, TP4 live seam,
full model, quality authority, or performance claim was exercised.

## Deployment boundary

A live netuid-14 deployment exists and has produced the dated mainnet control-plane
evidence above. Deployment-owned inputs remain outside this repository: endpoints,
registrations, permit/stake, validator/miner/burn identities, wallet/key custody,
immutable hosted bundles, GPU capacity, sealed capability bytes, backups,
monitoring, and frozen source-directory selection.

Repository merge, deployment cutover, and runtime qualification are separate
events. PR #95 put the live dependency set on `main`, but the 2026-08-19 merge did
not switch the running VM or pod away from their frozen revisions. A later cutover
must revalidate every runtime path and identity and produce fresh acceptance
artifacts; CI, this ledger, and the existence of earlier mainnet rows cannot stand
in for that proof.

## Source anchors

- [Slot catalog](https://github.com/latent-to/cacheon/blob/main/cacheon/slots.py)
- [Target catalog](https://github.com/latent-to/cacheon/blob/main/cacheon/target_catalog.py)
- [Hardened fetch](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/fetch.py)
- [Miner object-store publication](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/publish.py)
- [Eval-cost quote and payment](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/eval_cost.py)
- [Eval-cost make-good credits](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/eval_cost_credit.py)
- [Private validator archive](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/archive.py)
- [Standing CPU supervisor](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/standing_cpu_supervisor.py)
- [Persistent B300 remote adapter](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/b300_remote_worker_adapter.py)
- [Resident screening](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/resident_screen_lane.py)
- [Resident-pair crossover](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/resident_pair_crossover.py)
- [Adaptive resident runtime](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/crossover_runtime.py)
- [Qualification](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification_runner.py)
- [Audit gate](https://github.com/latent-to/cacheon/blob/main/cacheon/audit_gate.py)
- [Settlement](https://github.com/latent-to/cacheon/blob/main/cacheon/settlement.py)
- [Legacy publication](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/weights.py)
