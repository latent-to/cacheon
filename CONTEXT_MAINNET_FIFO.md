# Mainnet FIFO screen — continuation context

**Status (2026-08-04):** §10 two durable FIFO screens **proven** on netuid-14.
Screening-only — **not** a green subnet. Qualification / crown / settlement (§11)
remain out of scope until explicitly armed.

This file is operator continuity for the next session. It is **not** production
authority. Canonical ops docs live on the CPU VM under
`/root/cacheon-ops/chainops/` (see local `handoff.md` pointer). Dated evidence
claims belong in `docs/reference/state-of-record.md` when committing.

---

## What was achieved

1. **SGLang watchdog** — B300 engine kwargs set `watchdog_timeout=1800` when
   CUDA graphs are enabled (default 300s SIGKILL mid capture).
2. **Resident intake perms** — `resident-intake` created `0o711` so OCI
   non-root (`65534`) can traverse swap root (was `0o700` → EACCES on digest).
3. **Clearer swap error** — inaccessible staged swap surfaces distinctly from
   “absent or writable”.
4. **ABI cutover in worker image** — published FIFO bundles use
   `optima-op-abi-v0`; host accepts via `_PRE_CUTOVER_ABI_SHA256`; sealed image
   rebuilt so in-container swap accepts the alias.
5. **Launcher image rotation** — `start_mainnet_remote_worker.sh` rotates READY
   when `worker_image` changes even if source revision is unchanged.
6. **Authority rebind** — primary-authority + nested shallow receipt rebound to
   cutover image; backup under `/data/cacheon-b300/authority-rebind-abi-cutover-*`.
7. **§10 proof** — epoch `4d3df000acc41525f79a22cdc2a05a9c`,
   `adapter_start_count=1`, multiple durable screens including promotions
   `6e1b3773…`, `b5aa1359…`. Outcome recorded on CPU ledger + outcome JSON.

---

## Hosts and paths

| Role | Access | Notes |
|------|--------|--------|
| CPU VM | `root@5.161.203.13` | `/root/cacheon-ops/` |
| B300 pod | `root@204.9.206.240` port `40050` | known-hosts: `/root/cacheon-ops/state/lium-worker-known-hosts` |
| Intake DB | CPU | `/data/mainnet14-cacheon-h3-m4i-pre-crown/state/intake.sqlite3` |
| Working source | CPU | `/root/cacheon-ops/source-c066360b-git` (ops commits on top of `c066360b`) |
| Launcher | | `chainops/start_mainnet_remote_worker.sh` |
| Authority | pod | `/data/cacheon-b300/launch-b300-v3-m4l/primary-authority-v3-m4l/` |
| Nested receipt | pod | `/data/cacheon-b300/proofs/shallow-b300-graph-v3-m4l/receipt.json` |

---

## Identity pins (proof epoch)

| Item | Value |
|------|--------|
| Worker epoch | `4d3df000acc41525f79a22cdc2a05a9c` |
| Worker image (cutover) | `localhost:5000/cacheon-b300-worker@sha256:cfc0c7a3660bafe3f0ac63ede72c101d42dde50e5cc413e5fb92eb6eae096af9` |
| Base source | `c066360b…` |
| VM ops commits (approx order) | `c49d7fd` (watchdog), `33db68a` (intake 0711), `5970bc6` (image rotate) |
| Local laptop HEAD | still ~`c066360`; uncommitted mirrors of the three fixes + tests/docs |

---

## Hard constraints (do not violate)

- No second screen dispatcher on the same epoch.
- Hard-stop investigation on `epoch_failed`; do not chain-walk or raw-SQL lease recovery.
- Do not call the subnet green after screening-only success.
- Screening ≠ qualification ≠ crown ≠ settlement ≠ weight publication.
- Commit local changes only when explicitly asked.
- SSH/prod writes often need elevated approval; prefer CPU-native ops docs for handoff.

---

## Local workspace vs VM

**Verified 2026-08-04:** all five local diffs reviewed line-by-line against
this file's description and confirmed to match VM ops intent. No missing
pieces found; one cosmetic doc line-wrap fixed. Local `HEAD` is still
`c066360b9c875a999676cf995c8624956869e66d` — nothing has been committed.

- `cacheon/eval/b300_screen_deployment.py` — **present**: `watchdog_timeout=1800`
  set in `_engine_config` when `disable_cuda_graph` is false; `resident-intake`
  now `mkdir(..., mode=0o711)` + explicit `os.chmod(0o711)` (belt-and-suspenders
  against umask on `mkdir`). Matches VM `c49d7fd`.
- `cacheon/eval/oci_session_worker.py` — **present**: `_apply_resident_swap`
  now does `staged.lstat()` first and raises
  `SessionProtocolError("staged swap bundle is inaccessible: {exc}")`,
  distinct from the pre-existing "absent or writable" branch. Matches VM intent
  for a clearer inaccessible-swap error.
- `chainops/start_mainnet_remote_worker.sh` — **present**: the rotate-before-bind
  remote script now takes `POD_WORKER_IMAGE` as a third positional arg, reads
  `old_image` from the prior READY receipt (inline + retired-commission
  fallback paths both updated), and the early-exit guard is
  `old_revision == new_revision && old_image == new_image` — so an image-only
  cutover (same source revision, new sealed image) still rotates. Matches VM
  `5970bc6`. `bash -n` syntax-checks clean.
- `docs/reference/state-of-record.md` — **present**: "Routing-only resident
  screen" section gained watchdog/intake-perms rationale, the ABI cutover
  image mismatch (`13c72417`/`77fae0ec` vs `_PRE_CUTOVER_ABI_SHA256`), the
  2026-08-04 primary-authority rebind to `cfc0c7a3660b…`, and the §10 proof
  outcome (`4d3df000…`, `adapter_start_count=1`, screening-only/not-green).
  Fixed one run-on line-wrap for consistency with surrounding prose (no
  content change).
- `tests/test_b300_screen_deployment.py` — **present**: added
  `test_graph_engine_config_extends_sglang_watchdog_past_cuda_graph_capture`
  (asserts `watchdog_timeout` absent when eager, `1800` when graphs enabled)
  and `test_resident_intake_is_traversable_by_non_owner` (asserts intake mode
  `0o711` on first create and after a stale `0o700` root is left behind by a
  prior lifetime, i.e. `exist_ok` path also re-chmods). Both new tests pass;
  full file is 6/6 passing on CPU-only `.venv` (`python -m pytest -q
  tests/test_b300_screen_deployment.py`).
- `handoff.md` — pointer only (not authority); unchanged, not part of the
  mirror set.

**Sync readiness:** local tree is a faithful mirror of the three VM ops
commits (`c49d7fd`, `33db68a`, `5970bc6`) plus matching docs/tests. Nothing
further to implement here. Commit remains withheld per explicit
instruction — do not commit until the operator asks.

---

## Remaining work (TODO)

### P0 — keep screening healthy

Snapshot: [`STATUS_MAINNET_FIFO.md`](STATUS_MAINNET_FIFO.md) (2026-08-04 22:26 UTC) — **PASS**.

- [x] Confirm screen dispatcher still live for epoch `4d3df000…` (unchanged; no successor).
- [x] Spot-check heartbeat: `adapter_alive=true`, `adapter_start_count==1`, 0 consecutive
      failures, no `epoch_failed` (40/40 recent screens `completed`).
- [x] Cutover image digest still matches READY receipt (`cfc0c7a3660b…`).
- [ ] Ongoing: watch benign `_TransientCoordinatorOwnership` dispatcher restarts; leave
      intake alone (no chain-walk / lease surgery). Published ABI-v0 rows still screening.

### P1 — close the laptop ↔ VM gap

- [ ] Review and commit (when asked) the five local file mirrors so Git matches VM ops commits.
- [ ] Optionally refresh handoff artifact / SHA256SUMS on CPU if a new sealed handoff is required.
- [ ] Ensure `docs/reference/state-of-record.md` §10 note stays accurate if further epochs rotate.

### P2 — authority / image ceremony (ops hygiene)

- [x] Document the authority-rebind procedure as a short runbook step (backup path, nested
      `worker.preflight` capture, `device_execution.sha256`, READY rotate). See
      [`AUTHORITY_REBIND_RUNBOOK.md`](AUTHORITY_REBIND_RUNBOOK.md).
- [x] Live-validated 2026-08-04: backup
      `authority-rebind-abi-cutover-20260804T215516Z`, sealed
      `authority-config.json` / `measurement-config.json`, nested receipt
      `cacheon-runtime-preflight-v2`, worker digests pin cutover `cfc0c7a3660b…`.
- [x] Treat sealed primary-authority mutation as ceremony, not ad-hoc edit
      (runbook §2–§5).
- [ ] Record derived-image build recipe (wheel with `_PRE_CUTOVER_ABI_SHA256` → local registry tag).

### P3 — §11 and “green” (explicitly unarmed; do not start unless ordered)

- [ ] Remote qualification under bound authority (adaptive B/C/B′ … version-3).
- [ ] Evidence mirroring / independent reproduction PASS pair.
- [ ] Crown / settlement (lower accepted speedup).
- [ ] Weight publication / incentive activation (distinct authorities).
- [ ] Only then consider subnet “green” language.

### Out of scope unless re-opened

- Second dispatcher, qualification shortcuts from screen promotions, splicing authorities,
  treating abbreviated_serving promotion as crown evidence.

---

## Quick recovery pointers

- CPU entry: `/root/cacheon-ops/chainops/START_HERE.md`
- Ledger: `/root/cacheon-ops/chainops/00_MAINNET_LAUNCH_LEDGER.md`
- Screen log pattern: `/root/cacheon-ops/logs/mainnet-screen-dispatcher-<epoch>.log`
- Transport log: `/root/cacheon-ops/logs/remote-dispatch-<epoch>.log`
- Pod heartbeat: `/data/cacheon-b300/remote-worker/heartbeat.json`
- READY receipt: `/data/cacheon-b300/worker-bootstrap/ready-receipt.json`

---

## Success criteria already met (§10)

- Two (actually more) durable screens completed under one resident lifetime.
- Stable resident identity: `adapter_start_count == 1` through the proof window.
- Outcome on CPU launch ledger + outcome JSON.
- Explicit non-claim: not green; §11 unarmed.
