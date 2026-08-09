# Cacheon contributor and agent guide

This file is the operational entry point for automated contributors. It is not
a second product manual. The canonical engineering documentation lives under
`docs/` and is built from this repository.

## Read first

Choose the smallest relevant path:

1. `docs/get-started/concepts.md` — system vocabulary and trust boundaries.
2. `docs/architecture/product-model.md` — normative proposal, crown,
   integration, and release contract.
3. `docs/architecture/slot-contract.md` — normative contribution boundary.
4. `docs/reference/state-of-record.md` — dated implementation and evidence
   status.
5. `docs/miner-guide/overview.md` — contribution workflow.
6. `docs/validator-guide/overview.md` — intake, qualification, settlement, and
   publication.
7. `docs/engine/overview.md` — chain-independent serving releases.
8. `docs/security/threat-model.md` — implemented controls and residual risk.

If a task continues earlier Codex or Claude work, follow the cross-harness
continuity instructions supplied by the environment. Historical logs route an
investigation; current code, tests, Git state, and external state remain
authoritative.

`WORKLOG.md` and `docs/WORKLOG.md`, when present, are private local working
records. They are ignored and must not be committed, linked from public docs,
or treated as production authority.

## Product invariants

- A miner proposal is hostile input, not production source.
- The validator owns the model, workload, timing, outputs, references, target
  policy, and verdict.
- A contribution changes one registered singleton/atomic target or enters the
  fenced discovery lane.
- Candidate build and execution remain outside the trusted controller in
  validator-owned, no-egress OCI lifetimes.
- CUDA graphs are part of the scored contract.
- A first PASS is `reproduction_pending`. Settlement requires an independently
  bound PASS pair and uses the lower accepted speedup.
- The resident hot-swap screen is routing-only. Its measurements cannot crown,
  settle, or authorize rewards.
- Production version-3 qualification uses two resident TP lanes, adaptive
  B/C/B′ then optional C′/B″ speed evidence, a separate eager/untimed audit
  role when registered, pristine T, and a physical-lane role swap across
  reproduction.
- Evaluation-stack settlement, incentive activation, weight publication,
  integration review, release signing, and serving are distinct authorities.
- Legacy V1 weights are a fenced state machine. The V2 finite-debt economics
  were extracted from the tree on 2026-08-09; only their reserved durable
  schema remains, and reintroduction requires a new reviewed change. Do not
  infer registered discovery promotion from implemented arithmetic.

If a change weakens one of these statements, it requires an explicit design and
security review—not a local implementation shortcut.

## Repository map

```text
cacheon/                    runtime and control-plane package
  chain/                   finalized intake, durable state, activation, weights
  eval/                    screening, qualification, OCI, evidence, scoring
  integrations/            version-pinned SGLang adapters
cacheon_kernels/            validator-owned reference kernel library
examples/                  miner bundles and adversarial controls
tests/                     executable contracts and regressions
docs/                      canonical documentation site
scripts/                   repository validation and reproducible studies
```

Use `docs/reference/codebase-map.md` for authority-oriented entry points.

## Development workflow

Start from a clean understanding of the worktree:

```bash
git status --short --branch
git diff --stat
```

Do not overwrite, stash, or discard unrelated user changes. Keep changes
scoped. Runtime changes should be accompanied by focused tests before the full
suite.

### Execution control gates

- Keep at most three non-root agents active across the entire descendant tree,
  not merely three direct children. A child may not spawn a descendant unless
  the user explicitly authorizes descendant delegation. Check the live agent
  tree immediately before and after every launch. Give each agent one bounded
  outcome, exact file ownership, explicit forbidden actions, applicable
  product invariants, and exact acceptance tests. Review its output or diff
  before accepting it; delegation is not approval.
- A writing agent may own multiple explicitly listed files when they form one
  bounded implementation unit. It must state its implementation hypotheses
  before editing and name the evidence that would falsify each one. Its handoff
  must enumerate every file and every behavior or configuration choice it
  introduced, then include a brittleness audit covering hardcoded identities
  or paths, behavior under a second target profile, configuration precedence,
  stale state, partial failure, restart/idempotency, concurrency, time bounds,
  and untested assumptions. The primary agent must inspect the diff and
  independently rerun the acceptance tests before accepting it.
- Never use destructive synchronization such as `rsync --delete` for recovery,
  handoff, snapshot, or deployment work. Use a new destination and verify the
  copied manifest before considering any separately authorized cleanup.
- Do not hardcode a target, reservation, model, hotkey, or mission identity in
  a generic evaluator or scheduler. Resolve identities from registered,
  sealed, validated inputs and test at least two distinct target profiles.
- Predeclare device allocation, request concurrency, shard count, and a wall
  bound for multi-GPU evaluation work. A concurrency-one loop across otherwise
  idle devices is a stopped-plan condition unless the sealed workload requires
  it and evidence justifies it.
- Before an expensive or stateful runtime launch, test every writable path as
  the exact runtime UID/GID with create, fsync, reopen, and unlink; reopen and
  hash every read-only input through its exact runtime path; and validate
  socket parents, stale-result collisions, container names, authorization
  expiry, and rollback coordinates. A guessed permission or mount contract is
  a stopped-plan condition, not an infrastructure experiment.
- Do not hand-transcribe derived digests into commands or configuration. Hash
  only genuine trust-boundary artifacts, compute the digest from the consumed
  bytes in the same process whenever possible, and carry it through a
  machine-produced receipt or file reference. Use ordinary generated IDs or
  timestamps for non-authority working names; do not add checksum ceremony to
  incidental files.
- Do not create or grow a code file beyond 1,000 physical lines. When new
  behavior touches an existing oversized file, put the behavior in a focused
  sub-1,000-line module and limit the oversized-file change to the smallest
  necessary wiring or extraction.

CPU setup and baseline validation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cpu,dev,release]"
python -m pytest -q tests
```

Contributor bundle checks:

```bash
python -m cacheon.cli scan examples/miner_silu_torch
python -m cacheon.cli verify examples/miner_silu_torch \
  --device cpu \
  --dtype float32
```

Use `python -m cacheon.cli` for GPU work; SGLang uses spawned processes and the
module entry point preserves the required guard.

## SGLang and GPU rules

- `PINNED_SGLANG` in `cacheon/compat.py` is consensus-critical. An exact version
  mismatch or failed chokepoint is an error.
- The spawn-safe seam is installed in every interpreter through
  `import cacheon.bootstrap` in a `.pth` file.
- `cacheon/seams.py` is the only adapter registry. Bootstrap, activation, binding
  vocabulary, and compatibility checks derive from it.
- Adding a slot starts in `cacheon/slots.py`. A new SGLang chokepoint adds one
  adapter implementation and one `SeamAdapter` row; do not create a parallel
  registry.
- Block and collective contributions must satisfy graph capture/replay and
  declare the required graph metadata.
- Collective verification binds each process to its CUDA device before process
  group initialization.
- Do not mix measurements across runtime, model, image, topology, workload, or
  policy identities.

See `docs/dev/gpu-setup.md` and `docs/dev/sglang-tracking.md`.

## Trust and evidence rules

- Never import candidate Python or native extensions into the trusted
  controller.
- Treat static scanning as defense in depth, not containment.
- Do not convert infrastructure, baseline, reference, teardown, or incomplete
  evidence failures into candidate `FAIL`.
- Do not rerun only a favorable arm, splice authorities, change a threshold
  after observing a result, or replace missing evidence with logs.
- Settlement reopening and full causal regrade are different claims.
- A crown is measurement/attribution evidence. It does not approve provenance,
  maintainability, licensing, integration, or serving.
- Loading a native artifact during evaluation does not prove serving-wheel or
  release-provider closure.

See `docs/security/evidence.md` and `docs/security/isolation.md`.

## Documentation contract

Documentation ships with the code. Every pull request must make a
documentation-impact decision. Update docs in the same PR when changing:

- commands or flags;
- manifests, slots, targets, ABIs, stacks, arenas, receipts, or durable schemas;
- miner or validator workflows;
- settlement, incentives, or weight semantics;
- trust boundaries or failure behavior;
- dependencies, installation, compatibility, or releases.

Pure internal refactors, tests, and formatting may state why no docs update is
required.

Validate documentation with:

```bash
python -m pip install -r docs/requirements.txt
python scripts/check_docs.py
mkdocs build --strict
```

The checker enforces navigation coverage, internal links/anchors, repository
source links, CLI inventory, private-path exclusion, and retired-repository
removal. `site/` is generated output and is not committed.

See `CONTRIBUTING.md` and `docs/contributing/documentation.md`.

## Code-volume discipline

Introduced after a measured audit found production modules with no importer
anywhere, files packed to just under the line cap, and tests split into
numbered files. Deletion is a first-class outcome. These rules bind every
contributor and subagent:

- Never land a production module without a same-change consumer reachable
  from a real entrypoint, registered seam, packaging metadata, or the declared
  capability manifest (`cacheon/capability_manifest.py`). "A future private
  caller will import this" is not a consumer; a module read only by its own
  tests is not integrated.
- `python scripts/check_islands.py` enforces reachability against
  `scripts/island_baseline.txt`. Shrinking the baseline is cleanup; growing it
  is a reviewed decision that must be justified in the pull request.
- Target files under roughly 600 physical lines. The 1,000-line cap is a
  ceiling, not a budget; packing files to just under it is a design smell.
  Never split tests into `_partN` files — split by behavior with named scopes.
- State net line impact in every pull request and handoff summary. Prefer
  diffs that delete. A single reviewable unit above roughly +1,500 net
  production lines must be split or explicitly justified.
- Subagent output is a proposal, not an increment: delete what the task did
  not need before integrating it. Delegation is not accretion.
- Configuration flags a loader rejects, wrappers that cannot be enabled, and
  fences that are not wired are dead on arrival — wire them or drop them.
- Write the doc or kill the surface: an undocumented new command, flag, or
  schema is a deletion candidate, not a TODO.

## Persistence

Committed code, tests, this file, and `docs/` are the portable context. Keep
dated empirical claims in `docs/reference/state-of-record.md` or `docs/results/`;
keep evergreen pages neutral and present-tense. Detailed chronology belongs in
Git history or `docs/history/`, not in operator and architecture pages.
