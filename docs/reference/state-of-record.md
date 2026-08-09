# State of record

This page is the dated capability and evidence ledger for Cacheon. Evergreen
pages define contracts and procedures; this page identifies the implementation
revision, evidence class, and unresolved limits behind readiness claims.

Passing tests, completing an empirical qualification, activating an incentive
policy, publishing weights, and producing a deployable engine release are
different events. Evidence for one does not authorize another.

## Source snapshot

Snapshot date: **2026-07-31**

| Item | Value |
|---|---|
| Repository | [`latent-to/cacheon`](https://github.com/latent-to/cacheon) |
| Implementation parent | [`a04b824f`](https://github.com/latent-to/cacheon/commit/a04b824f); this page accompanies the Optima→Cacheon identifier rename |
| Production Python | 129 files and 106,756 lines under `cacheon/` |
| Tests | 127 Python files and 65,135 lines under `tests/` |
| Complete local suite | 2,420 passed, 17 skipped, 0 failed at the 2026-08-03 protocol rename head; the 2026-07-31 rename head recorded 2,417 passed and 19 skipped |
| Test command | `python3 -m pytest -q tests` with Python 3.10.4 in an unrestricted local environment |
| SGLang pin | `0.5.13.post1` in `cacheon/compat.py` |
| Bittensor raw-reveal storage ABI | `10.3.2` in `cacheon/chain_canary.py` |
| Public CLI | 25 commands after the 2026-08-09 V2 economics extraction |

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
`HOW_CACHEON_WORKS.md` redirects to the canonical
architecture documentation. The rename does not alter kernels, timed
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
- Replayed discovery proposals are terminally disposed or deduplicated before
  screening. Legacy schema-3 single-PASS migration holds are non-crownable and
  have an evidence-preserving archive command.

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

### Slots, targets, and direct artifacts

The executable catalog contains 11 slots and one registered atomic target:

| Kind | Registered identifiers |
|---|---|
| Op | `activation.silu_and_mul`, `norm.rmsnorm` |
| Block | `attention.sdpa`, `attention.decode`, `attention.msa_block_score`, `attention.msa_prefill_block_score`, `moe.fused_experts` |
| Collective | `moe.fused_experts_reduce`, `collective.all_reduce`, `collective.ar_residual_rmsnorm`, `collective.moe_finalize_ar_rmsnorm` |
| Atomic target | `collective.moe_epilogue.v1` over the two MoE epilogue collective members |

The closed direct-artifact registry has one crownable provider,
`cutlass.cute.cubin.v1`. Candidate compiler-factory code runs in a GPU-hidden,
no-network child and may publish one sealed CUBIN. Validator code owns ABI
admission, ordinal binding, pointer/scalar/TMA materialization, launch, storage,
cleanup, and evidence. The schema exposes collective vocabulary, but the
standard provider does not supply arbitrary group/peer resolvers; unsupported
plans fail closed.

### Routing-only resident screen

The abbreviated-serving stage may keep a stock engine resident and hot-swap a
bounded candidate queue. Each swap is generation-bound, triggers graph
recapture, and is checked by shared stock brackets and contamination canaries.
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

### Resident adaptive qualification

Production providers select qualification policy version 3:

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
with distinct authority/evidence commitments and, for version 3, the required
physical-lane role swap. It uses the lower accepted speedup, reopens exact
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

### Inactive V2 finite debt

On **2026-08-09** the V2 finite-debt economics implementation — finite
registered-CROWN debt, the reviewed-discovery bounty class, campaign and
composition policies, the wallet-free activation command, and
`set-debt-weights` publication — was extracted from the tree without ever
being activated. No live V2 activation or publication receipt ever existed.
The design intent is retained (a bounded post-activation claim paid down over
confirmed epochs) and the complete implementation is recoverable from Git
history at [`dc158fb4`](https://github.com/latent-to/cacheon/commit/dc158fb4).

Two durable-compatibility artifacts remain in the tree:

- [`chain/reserved_schema.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/reserved_schema.py)
  preserves the schema-4/5/6 migrations and V2 table DDL verbatim, so every
  existing intake database keeps validating and fresh databases keep
  producing byte-identical schemas; and
- the shared-weight offer wire schema keeps its `lane`/`debt_binding` fields
  with `lane` restricted to `legacy_v1` and any debt-lane payload rejected,
  so historical stored offers reopen byte-identically.

Reintroducing V2 is a new reviewed change, not a revert switch.

### Engine release

Evaluation and serving remain separate products. The release model includes
reviewed integration records, sealed model/native identities, deterministic
source/wheel products, SBOM/provenance, Ed25519 signatures, OCI context, host
policy, registry types, and serving receipts.

Current release authority is incomplete:

- the serving wheel does not close every manifest/direct-artifact runtime
  import;
- release preparation does not provider-specifically rebuild and reopen the
  complete CuTe index/compile-profile authority;
- `release_runtime.py` does not propagate the signed CuTe compile-profile
  digest into the engine process;
- builder output, effective runtime arguments, management-route policy, and
  complete release/session receipt binding still require end-to-end closure;
  and
- no final deterministic registry pair, authorized image, or complete all-rank
  serving receipt set is claimed for this revision.

Loading sealed native artifacts inside evaluation OCI proves evaluation
runtime support. It does not close the serving release.

## Empirical evidence

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

### Current audit-path canary status

A bounded 4×B300 run on 2026-07-19 did not satisfy the current launch gate. The
sabotage control was rejected. The honest primary produced no verdict after
concurrent legs shared an executor label and invalidated quiescence authority.
The honest reproduction passed graph and pristine-T quality, but its deep slot
had only four audited calls per rank against the required 32 and its speed gate
failed at 1.005507×. Zero observed audit comparison violations did not repair
insufficient coverage.

This is retained failure evidence. It is not an activation canary, PASS, or
performance authority. The subsequent resident screen and two-lane adaptive
qualification implementation are test-covered and informed by GPU calibration,
but no retained end-to-end current-revision version-3 primary/reproduction
canary is claimed here.

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
chain-package  chain-publish  chain-submit  chain-status  chain-register
chain-reservation-status  chain-validate  chain-snapshot  chain-snapshot-verify
chain-archive-schema3-hold  chain-evaluation-lease
model-provision  release-verify  release-context
set-weights
mint-push-credentials  mint-weight-gateway  push-weight-offer  serve-weights
follow-weights
```

The local miner loop is `scan` plus `verify`. Complete-engine performance and
quality authority begins with a deployment-injected arena provider.

## Deployment boundary

The source implements the mainnet-shaped intake, evaluation, settlement, and
publication control planes. Moving from a completed production-shaped testnet
exercise to a live subnet still requires deployment-owned inputs and evidence:
endpoint/netuid, registrations, permit/stake, validator/miner/burn identities,
wallet/key custody, immutable hosted bundles, GPU capacity, backups,
monitoring, and the required current-version GPU canary. This page does not
claim that a live mainnet deployment or receipt exists.

## Source anchors

- [Slot catalog](https://github.com/latent-to/cacheon/blob/main/cacheon/slots.py)
- [Target catalog](https://github.com/latent-to/cacheon/blob/main/cacheon/target_catalog.py)
- [Hardened fetch](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/fetch.py)
- [Miner object-store publication](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/publish.py)
- [Private validator archive](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/archive.py)
- [Resident screening](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/resident_screen_lane.py)
- [Adaptive resident runtime](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/crossover_runtime.py)
- [Qualification](https://github.com/latent-to/cacheon/blob/main/cacheon/eval/qualification_runner.py)
- [Audit gate](https://github.com/latent-to/cacheon/blob/main/cacheon/audit_gate.py)
- [Settlement](https://github.com/latent-to/cacheon/blob/main/cacheon/settlement.py)
- [Legacy publication](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/weights.py)
- [Reserved V2 schema](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/reserved_schema.py)
- [Release construction](https://github.com/latent-to/cacheon/blob/main/cacheon/release.py)
