# Diagnostics by lifecycle stage

Diagnose a proposal from its last authoritative state. A local PASS cannot
override a later intake, qualification, or settlement result because
each stage has different identity and evidence.

## Decision vocabulary

Cacheon uses three qualification grades:

| Decision | Meaning | Miner response |
|---|---|---|
| `PASS` | the bound evidence passed this stage | a complete audited qualification PASS becomes `qualified`; settlement is separate |
| `FAIL` | the candidate or its declared applicability violated a requirement | change the proposal and submit a new content identity |
| `NO_DECISION` | authority, infrastructure, conditioning, drift, or evidence was insufficient for a safe verdict | preserve the proposal identity and wait/retry under operator policy |

`NO_DECISION` is not a weak pass, and `FAIL` is not converted to a zero-scored
candidate.

## Durable intake states

The production SQLite state machine currently exposes these statuses:

| Status | What it means |
|---|---|
| `reserved` | finalized reveal admitted and waiting for a fetch lease |
| `fetching` | HTTPS fetch/extract/hash/publication work is active |
| `transport_retry` | transient transport failure is eligible for another fetch attempt |
| `published` | immutable worker publication and selected-delta identity exist; waiting in the qualification queue |
| `qualifying` | one authority-bound version-3 attempt—paired replay of B and C, registered eager audit A, then pristine T—is active |
| `reproduction_pending` | historical transition; a retained complete primary PASS is accepted on restart without another GPU run |
| `qualified` | one complete audited PASS is retained; settlement is separate |
| `no_decision` | retryable qualification evidence/failure product was retained |
| `held` | automatic progress stopped under retry, capacity, or safety policy; operator action is required |
| `failed` | terminal invalid/rejected/failed candidate |
| `expired` | terminal finalized-block SLA expiry; wall-clock age is not the authority |

These transitions are implemented in
[intake.py](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/intake.py).
The command `chain-status` shows revealed chain commitments, not this private
state machine. Obtain lifecycle receipts through the operator's published
surface.

## 1. Manifest and target resolution

Start with:

```bash
python -m cacheon.cli scan my_bundle
```

`scan` parses paths and the component manifest and checks the source tree. It
does not reproduce production target resolution or trusted rebuild-feature
observation. For a target error, use the intake receipt and compare the parsed
manifest with the active catalog; `verify` additionally preflights variant
domains but is still not the intake resolver.

Common failures and fixes:

| Symptom | Likely cause | Fix |
|---|---|---|
| unsupported ABI | `abi_version` is not a supported identifier | author new bundles with the published `cacheon-op-abi-v0` component ABI |
| unsafe/missing path | absolute path, traversal, symlink, or undeclared file | make every declaration bundle-relative and source-only |
| competition mode mismatch | the mode is not `slot` | use `mode = "slot"` with `forward_pass` or `prefix_cache` |
| `node addresses must sit under ...` | a node-address row names a module outside the roots, or is malformed | name modules under `model` or `logits_processor`, or the prefix cache `tree_cache`; `*` stands for one whole segment |
| `tree_cache entry ... must be a function accepting the runtime cache` or `... must be callable with only the runtime cache` | the entry is not a synchronous function in the declared source, or needs other arguments | define `entry(cache)` and return `type(cache)` or a subclass of it; do not supply a constructor or `prepare` |
| `tree_cache: the cache served a page that does not hold the KV the engine computed for its prefix` | a served prefix's slots held other bytes: a stale or reused slot, a page kept across a flush, a host copy not brought back, or a page from another namespace | serve only slots whose bytes the engine computed for that exact prefix, and drop every page at a flush |
| `tree_cache: it replaced match_prefix on the instance ...` or `... on its class, which skips the check` | the cache assigned or deleted a protected handoff method after binding | override methods in the subclass returned by the factory |
| `tree_cache: state validation is unavailable for this hybrid pool` | the engine's sliding-window or recurrent pool has a layout the validator does not recognize | this is missing validator coverage, not a failed miner contribution |
| `tree_cache: this hybrid runtime disables prefix caching` | the arena's engine configuration turns radix caching off, as the Qwen development inputs do | the prefix cache is not a target on that arena; use an arena that serves with prefix caching |
| `... overlap; claim the wider node alone` | two rows of one bundle name a node and a node inside it | keep the wider address and drop the narrower one |
| feature not allowed | `setup` or an unknown extra is outside policy; `override points are retired` names a row that sets `base_kernel` or `override_point` | remove it; both targets admit declared CUDA sources and the registered rebuild |
| incomplete feature evidence | intake could not independently observe the rebuild feature set | use only registered rebuild declarations and complete source inventory |
| duplicate slot requires variants | repeated rows omit explicit unique `variant` | name every variant |
| overlapping domains | two variants can route the same live call | make capability domains provably disjoint |

A missing `[competition]` resolves by node roots, but declare the target
explicitly for new submissions.

### Do not collapse every local error into “scan failed”

The point at which output stops identifies the layer:

| Last output | What has actually happened | Next check |
|---|---|---|
| TOML/path exception before the bundle summary | manifest parsing or declared-path validation stopped | syntax, required fields, identifier spelling, file existence, relative path containment |
| bundle summary plus `[VIOLATIONS]` | the manifest loaded, but one declared or recursive-tree source policy failed | every printed file/line and any undeclared executable material |
| clean `scan`, then a variant registration error in `verify` | static source is clean, but the rows cannot register unambiguously | JSON types, manifest/metadata intersection, duplicate or overlapping variants, token-count constraints |
| `[INTERFACE OK]` | the entry imports and its signature accepts the module, prepared state or runtime cache | run `check` in the published image for the audit and captured execution |

`check` reports `Early stop: gross numerical audit violation` as soon as a valid
rolling receipt proves a large numerical mismatch. It stops the owned engine
and ranks, retains the receipts and logs, and skips graph execution. Near misses,
comparison/reference errors and incomplete receipts do not trigger this early
stop; they retain their normal terminal handling.

Run `scan` separately even though `verify` repeats recursive policy checks. The separate
command gives the cheapest no-import result; `verify` then registers the variants and
imports the entries. Neither command performs production target feature resolution or
runs candidate math, so a local clean result cannot overrule a later catalog rejection.

## 2. Capability routing

The node binder describes each live call by four fields only: dtype, last
dimension, GPU architecture and graph mode. A row that constrains any other field,
such as `model`, `phase`, `head_dim` or `tp_size`, never matches, so stock serves
every call. Neither `scan` nor `verify` evaluates routing. In `check`, a claimed
address with no audit windows or no captured execution is the symptom.

If a claimed address never runs, check:

- canonical architecture spelling (`sm103`, not an informal GPU name);
- dtype intersection between manifest and metadata;
- `last_dim` and `graph_mode` values, if declared;
- any other declared field, which must be removed.

Qualification requires every claimed address to complete on every rank, so a row
that routes to stock cannot pass. The prefix cache is selected once at engine start with an
empty descriptor; a `tree_cache` row declares no dtypes, architectures or metadata.

## 3. Interface and numerical checks

Run the cheapest relevant check first:

```bash
python -m cacheon.cli verify my_bundle
```

`verify` scans the bundle and imports each entry in a spawned child. It catches a
wrong entry name, a signature that cannot take the node's stock arguments, and
import errors. It runs no forward math: numerical truth is the stock module in the
running engine.

Then run `cacheon check` in the published arena image with the arena's development
inputs (see [Your first bundle](your-first-kernel.md)). Its table has one row per
claimed node address and rank: audit windows, violations, the worst passing
fraction, and captured execution. Typical failures:

- **wrong positional signature:** compare with [Kernel ABI](kernel-abi.md);
- **violations on every audited call:** the result differs from stock beyond the
  honest-twin bound; confirm formula, scale, mask, dtype and returned structure;
- **violations on some calls only:** compare those calls' shapes and phases with
  your tiling, ragged-tail and stride assumptions;
- **engine-state rows differ:** a node that writes KV or recurrent state must
  write the same rows stock writes for that batch;
- **no audited calls (exit 3):** the claimed address bound nothing the requests
  exercise, or the inputs are too short; use the arena's real development inputs;
- **one rank hangs/fails:** ensure all ranks issue collectives in the same order
  and do not hide an exception before a peer collective;
- **prepared representation wrong:** keep `prepare` deterministic, input-pure,
  and consistent with the live dtype/quantization binding.

A CPU `verify` pass is never a numerical, distributed, graph, or throughput pass.

## 4. Graph evidence

Qualification has no separate graph stage; a graph problem shows up in one of
three places:

- the candidate raises while the engine captures its graphs: the timed run
  stops with the candidate's original error;
- `never invoked inside a CUDA-graph capture`: every claimed slot completed,
  but at least one only ever ran eagerly, usually because the declared domain
  excludes the captured shapes. The candidate would have been timed as stock,
  so this is a `FAIL`;
- a captured kernel that replays a stale answer passes execution and fails the
  pristine quality gate.

A local `cacheon check` starts a graphs-on engine after its audit passes and
reports captured execution per address and rank. The prefix cache runs outside
CUDA graphs, so `tree_cache` needs no captured execution.

Common causes are host synchronization, data-dependent Python branching,
capture-time compilation/allocation, stale pointers, partial replay writes, or
collective ordering changes.

Do not “fix” graph failure by disabling CUDA graphs in a local profile; that changes the
serving regime. See [Graph evidence](graph-safety.md).

## 5. Chain payload, fetch, and publication

If the commitment is not accepted locally, use `chain-submit --dry-run` and
check the exact HTTPS URL and 64-character lowercase hash. If `--pay` transferred
TAO but the reveal commit failed, retry without `--pay` using the unused pointer
in [Submitting](submitting.md#step-by-step-commands).

After reveal, transport failures divide into two classes:

- transient DNS/timeout/selected server failures may enter `transport_retry`;
- canonical URL, public-route, TLS, size, archive-shape, or content-hash failures
  are terminal candidate failures.

Check that:

- the hosted object is still the archive produced by `chain-package`;
- the bundle directory was not edited between package and submit;
- redirects also resolve to valid public HTTPS destinations;
- the URL has no credentials or fragment;
- the server returns the full object within current limits;
- the archive is gzip-compressed tar with one valid wrapper/root and only
  permitted identity-bearing files;
- no regular file exceeds 16 MiB, no inspectable source/configuration file
  exceeds 8 MiB, and all inspectable files remain within the 32 MiB aggregate
  budget.

Publication/storage faults after a valid fetch are validator-side
`NO_DECISION`, not candidate failure. The transport boundary is implemented in
[fetch.py](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/fetch.py)
and immutable publication in
[publication.py](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/publication.py).

## 6. Full qualification

Qualification aggregates mandatory graph, marginal speed, registered eager
audit A when required by the plan, and pristine T quality evidence. The current
speed order is the paired replay of B and C, followed by A and T. Any FAIL makes
the attempt fail; any `NO_DECISION` prevents PASS.

Speed problems:

- the paired gain has not cleared the sealed statistical boundary by the last
  sealed window: candidate FAIL (`speed_threshold_not_met`, or `candidate_slower`
  when the estimate falls below the mirrored bound). The last window always
  decides PASS or FAIL;
- the first lane orientation reads slower than the arena's sealed futility
  margin: candidate FAIL without the swapped orientation;
- replay windows or physical-lane identities fail their bound consistency
  checks: authority/infrastructure uncertainty, never a candidate pass;
- C clears the speed bound but misses service attainment: candidate FAIL
  (`service_contract_not_met`);
- arm identities/resources differ: authority mismatch, never a valid speedup;
- candidate falls back for material calls: no positive marginal effect.

Quality problems:

- required fidelity/task metric regresses: candidate FAIL;
- stock/reference drift overlaps the calibrated boundary: `NO_DECISION`;
- referenced pristine-T identity or raw quality artifact cannot reopen:
  authority/infrastructure failure;
- a contributor-controlled quality check differs: local flags and prompts are not the
  bound reference policy.

Contributor-controlled matched A/B profiling can reproduce a mechanism, but it cannot
contest a retained validator grade or create qualification authority.

## 7. Qualification acceptance

One complete audited PASS becomes `qualified`. The same attempt's speed, graph,
audit, and pristine-quality evidence backs settlement; no mandatory repeat runs.
Historical paired results retain their original evidence and lower accepted score.
A crown still requires the separate transactional settlement step.

## 8. Settlement and reward

`qualified` means one complete audited PASS is retained; settlement can still wait
for older overlapping arrivals, a cohort lease, and retained-evidence reopening.
Transactional settlement may:

- crown a registered-target candidate;
- neutralize a non-winning/overlapped candidate;
- hold a candidate when authority cannot safely advance.

Weight publication is a separate reconciled action. Under V1 a PASS earns
decaying credit when it beats the best earlier rewarded PASS in its reward group
by the required margin; a crown is not required. V1 promises no fixed per-slot or
token payout. See [Incentives](incentives.md).

A crown is not integration: adopting crowned source into maintained code is a
maintainer decision outside this repository.

When reporting a problem, include the content hash, target ID, arena/evaluation
stack digest, last durable status, decision/reason, and evidence/receipt digest.
Do not include wallet secrets, private URLs, or validator filesystem paths.

### Baseline admission closed

`baseline_closed_at_submission` is a historical admission outcome. The former rule
returned commitments made after the first crown as `NO_DECISION`, before evaluation,
preserving the submission credit or cited evaluation payment. Those records retain
their original message: “This baseline closed before your submission. Your submission
credit has been preserved.” They are not automatically reopened; submit again using
the preserved credit or payment.

New commitments remain admissible after a crown. They run against the commissioned
baseline and must still clear the applicable settlement and reward hurdles.

`lost_potential` means evaluation passed, but the completed comparison did not clear
the required margin. The detail notice distinguishes a reward comparison against the
best earlier rewarded PASS in the reward group from a champion comparison. The PASS and measured throughput
remain visible; this is not an execution or correctness failure.
