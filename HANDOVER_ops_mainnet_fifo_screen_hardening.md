# Handover — `ops/mainnet-fifo-screen-hardening`

**Date:** 2026-08-04 (evening, America/Sao_Paulo)  
**Branch:** `ops/mainnet-fifo-screen-hardening` → `origin/ops/mainnet-fifo-screen-hardening`  
**Tip commit:** `a6fa7ac` — *Harden B300 FIFO screening after §10 proof failures*  
**PR:** https://github.com/latent-to/cacheon/pull/75 (base: `codex/mainnet-autonomous-ops-cacheon`)  
**Parent continuity:** CPU-native handoff at `/root/cacheon-ops/chainops/` (see local [`handoff.md`](handoff.md)); this file covers **what changed after that handoff** on the screening/FE path.

This document is operator continuity, **not** production authority. Canonical ops ledger remains `/root/cacheon-ops/chainops/00_MAINNET_LAUNCH_LEDGER.md`. Evergreen product rules: `docs/reference/state-of-record.md`, `AGENTS.md`.

---

## 1. Where we started (earlier handoff)

From the 2026-08-04 CPU handoff (`START_HERE.md` / `MAINNET_HANDOFF_2026-08-04_2100_IST.md` / emergency handoff):

- Netuid **14** mainnet; CPU `root@5.161.203.13`; B300 pod `root@204.9.206.240:40050`.
- Source base **`c066360b`** (resident worker across FIFO screens). Materialized handoff tree verified at that commit.
- Live path was **FIFO screening only**. Qualification/crown/settlement/weights intentionally closed on the remote screen stack (`B300ScreenDeploymentAuthorities`; `run_qualification` fail-closed until evidence mirroring).
- Morning **m4l** one-shot (graph→cal→joined→qual) already ran against reminted FE content **`314cf17c…`** / reservation **`a9a258…`** and terminal-**FAIL**ed `speed_regression`. That program receipt forbade a second m4l retry on that failure.
- Testnet-proven FE v9 content is **`747405b4…`** (optima object on Hippius). Remint **`314cf17c…`** is the cacheon-vocabulary twin used in m4l — **not** the same row as `747405b4…`.

---

## 2. What this branch is for

Land the **ops fixes that made §10 resident FIFO screening work**, plus operator continuity docs, on a reviewable Git branch:

| Area | Change |
|------|--------|
| `cacheon/eval/b300_screen_deployment.py` | `watchdog_timeout=1800` when CUDA graphs on; `resident-intake` `0o711` |
| `cacheon/eval/oci_session_worker.py` | Distinct error when staged swap is inaccessible (EACCES) |
| `chainops/start_mainnet_remote_worker.sh` | Rotate READY when **worker image** changes even if source revision unchanged |
| `tests/test_b300_screen_deployment.py` | Watchdog + intake-traversability tests |
| `docs/reference/state-of-record.md` | FIFO / ABI cutover / §10 screening-only notes |
| Ops notes (repo root) | `CONTEXT_MAINNET_FIFO.md`, `TODO_MAINNET_FIFO.md`, `STATUS_MAINNET_FIFO.md`, `AUTHORITY_REBIND_RUNBOOK.md`, `handoff.md` |

VM already had equivalent ops commits (`c49d7fd`, `33db68a`, `5970bc6`) on `/root/cacheon-ops/source-c066360b-git`. This branch mirrors that intent onto GitHub.

**Tests already run for the PR:** `python -m pytest -q tests/test_b300_screen_deployment.py` → **6/6** on CPU `.venv`.

---

## 3. What we did after the earlier handoff (ops narrative)

### A. §10 resident FIFO screen proof (done)

Failure chain fixed in order, then recommissioned:

1. SGLang default `watchdog_timeout=300` SIGKILL mid CUDA-graph capture → extend to 1800.
2. `resident-intake` `0o700` → OCI uid 65534 could not `lstat` digest → `0o711`.
3. Sealed worker image lacked `_PRE_CUTOVER_ABI_SHA256` → rebuild cutover image `…@sha256:cfc0c7a3660b…`.
4. Launcher skipped READY rotate on image-only change → rotate on image too.
5. Primary sealed authority rebound to cutover (backup `authority-rebind-abi-cutover-20260804T215516Z`).

**Proof:** epoch `4d3df000acc41525f79a22cdc2a05a9c`, `adapter_start_count=1`, multiple durable screens (incl. promotions). Recorded on CPU ledger. **Screening-only — not green.**

### B. Checklist / hygiene (done)

- Live health snapshot: [`STATUS_MAINNET_FIFO.md`](STATUS_MAINNET_FIFO.md) (PASS at 22:26 UTC while screening was still up).
- Authority-rebind runbook: [`AUTHORITY_REBIND_RUNBOOK.md`](AUTHORITY_REBIND_RUNBOOK.md) (paths live-validated).
- Continuity: [`CONTEXT_MAINNET_FIFO.md`](CONTEXT_MAINNET_FIFO.md), [`TODO_MAINNET_FIFO.md`](TODO_MAINNET_FIFO.md).

### C. FE v9 full-eval attempt (partial — interrupted)

Owner asked to pause screening, fix repro image skew, put FE v9 at front of queue, start full eval.

| Step | Status |
|------|--------|
| Stop screen dispatcher + remote-dispatch for epoch `4d3df000…` | **Done** — both stopped |
| Stop resident `cacheon-runtime-*` container | **Done** — all 8 GPUs **0 MiB** |
| Rebind **repro** authority worker pins to cutover `cfc0c7a…` (match primary) | **Done** — backup `authority-rebind-repro-cutover-20260804T233345Z` |
| Update `/data/cacheon-b300/worker-image-b300-v3-m4.txt` to cutover | **Done** (verify still cutover before launch) |
| `chain-submit` FE v9 `747405b4…` (Hippius optima URL, wallet `main`/`vali`) | **Done** — `submitted=True` |
| Intake publish FE into pre-crown | **Done** — see pins below |
| Stage pod mission `m4n` (rsync intake + publication; rewrite `publication_root`) | **Incomplete** — empty dirs only; rsync/approval interrupted |
| New one-shot program root (cannot reuse spent m4l OUT/GRAPH/STAGE) | **Not started** |
| Launch graph→cal→joined→qual for `747405b4…` | **Not started** |
| Crown / weights | **Do not** until both PASS |

---

## 4. Live state snapshot (at handover write)

| Item | Value |
|------|--------|
| Screen dispatcher | **stopped** |
| Remote dispatch `4d3df000…` | **stopped** |
| GPU memory | **all 8 × 0 MiB** |
| Cutover image | `…@sha256:cfc0c7a3660bafe3f0ac63ede72c101d42dde50e5cc413e5fb92eb6eae096af9` |
| Primary + repro worker.image | both end in `…cfc0c7a…096af9` (repro rebound) |
| Intake (pre-crown) | `published: 1`, `promoted: 4`, `held: 16`, `failed: 12`, `expired: 32` |
| **FE v9 reservation** | `dd382ab6dafd56af4913e2f664925532a66652ad00d5485896f33a3cc25ab0d3` |
| FE status | **`published`** @ block `8774167` |
| FE content | `747405b41845506800939507a93b6011d38f5a94e69a5ec303a3d39a48e77709` |
| FE URL | `https://s3.hippius.com/cacheon-prod/optima/miner-bundles/sha256/747405b4….tar.gz` |
| FE publication (CPU) | `/data/mainnet14-cacheon-h3-m4i-pre-crown/publications/ef/ef0742588527763f76e021e1f327e793026936f3f9efe463095ddc160b462197` |
| Pod `m4n` tree | dirs exist (`state/`, `publications/`, …) but **no intake DB / pub payload yet** |
| Spent m4l roots (do not reuse) | `/data/cacheon-b300/launch-b300-v3-m4l`, `proofs/shallow-b300-graph-v3-m4l`, `controller-snapshots/77fae0ec-v3-m4l-stage` |

**Two stacks (do not confuse):**

1. **Remote screen worker** — screen-only; intentionally cannot qualify. Currently **off**.
2. **m4l/m4n one-shot** — private graph/cal/joined/qual on the pod (same *kind* as testnet). Morning m4l spent; need **new** `m4n` (or similar) roots + mission DB that contains FE as `published`.

---

## 5. What needs to be done next

### P0 — finish FE v9 one-shot (if still ordered)

1. **Complete pod mission staging for m4n**
   - From CPU: rsync pre-crown `intake.sqlite3` (sqlite `.backup`), `competition-start*.json`, `chain-audit.jsonl`, FE publication tree → `/data/mainnet14-cacheon-h3-m4n/…`
   - On pod: rewrite `publication_root` prefixes `…-pre-crown/` → `…-m4n/`
   - Confirm FE row `dd382ab6…` is `published` and `Path(publication_root).is_dir()`
2. **Author `chain_mainnet14_m4n.sh`** (clone `/data/stage/mainnet-m4/chain_mainnet14_m4l.sh`)
   - `CONTENT=747405b4…`
   - New `OUT`, `GRAPH_ROOT`, `STAGE`, `M=/data/mainnet14-cacheon-h3-m4n`
   - Worker image from cutover file / READY (`cfc0c7a…`)
   - Fresh audit seed; one-attempt rules on new roots
   - **Do not** reopen `a9a258` / `314cf17c`
3. **Launch** under pod tmux with a hard time/$ ceiling; CPU supervisor optional (pattern: `m4l_supervisor.sh`)
4. **Only after both primary + reproduction PASS:** evidence export → crown/weights ceremony (Shiv-fired wallet steps per CPU ledger)

### P1 — PR / branch hygiene

- Review/merge [PR #75](https://github.com/latent-to/cacheon/pull/75) (screening hardening + continuity docs).
- Optionally commit this handover + refreshed `TODO_MAINNET_FIFO.md` on the same branch.
- Keep VM `/root/cacheon-ops/source-c066360b-git` and Git branch aligned when commissioning again.

### P2 — if screening must resume later

- Recommission/restart **one** screen dispatcher + remote-dispatch for a **new** epoch after FE program finishes (or explicitly abort FE).
- Do **not** start a second dispatcher on `4d3df000…`.
- Confirm intake `0o711`, cutover image, and READY identity before claiming §10 again.

### Out of scope unless re-ordered

- Automatic remote qualification drain for all `promoted` rows (needs evidence-mirror seam — see CPU handoff §11).
- Calling the subnet green after screening or after a single PASS.
- Raw SQL lease surgery / chain-walk recovery.

---

## 6. What to test

| Layer | Check |
|-------|--------|
| Unit (already green) | `pytest -q tests/test_b300_screen_deployment.py` |
| Image identity | Pod READY + `worker-image-*.txt` + primary/repro `worker.image` all `cfc0c7a…` |
| FE mission | Pod m4n DB: exactly one FE `published` row; publication path exists; content hash fetchable |
| Program invariants | New roots empty before start; graph receipt `status=passed`; no favorable-arm redraw; no second retry after typed FAIL |
| Resident / GPU | Before qual launch: no leftover `cacheon-runtime-*` holding lanes; gpu_coord leases clean (see ledger compaction notes on zombie leases) |
| After PASS pair | Settlement uses lower speedup; weights only via existing offer/follower path |

---

## 7. Key pins (copy/paste)

```text
FE content:     747405b41845506800939507a93b6011d38f5a94e69a5ec303a3d39a48e77709
FE reservation: dd382ab6dafd56af4913e2f664925532a66652ad00d5485896f33a3cc25ab0d3
FE URL:         https://s3.hippius.com/cacheon-prod/optima/miner-bundles/sha256/747405b41845506800939507a93b6011d38f5a94e69a5ec303a3d39a48e77709.tar.gz
FE pub (CPU):   /data/mainnet14-cacheon-h3-m4i-pre-crown/publications/ef/ef0742588527763f76e021e1f327e793026936f3f9efe463095ddc160b462197

Cutover image:  localhost:5000/cacheon-b300-worker@sha256:cfc0c7a3660bafe3f0ac63ede72c101d42dde50e5cc413e5fb92eb6eae096af9
Screen epoch:   4d3df000acc41525f79a22cdc2a05a9c   (stopped; do not double-dispatch)

Do not retry:   a9a258… / 314cf17c… (m4l FAIL speed_regression)
```

---

## 8. Related local files

| File | Role |
|------|------|
| [`handoff.md`](handoff.md) | Pointer to CPU-native handoff root |
| [`CONTEXT_MAINNET_FIFO.md`](CONTEXT_MAINNET_FIFO.md) | FIFO §10 continuity (may lag FE section — prefer this handover for FE) |
| [`TODO_MAINNET_FIFO.md`](TODO_MAINNET_FIFO.md) | Short checklist |
| [`STATUS_MAINNET_FIFO.md`](STATUS_MAINNET_FIFO.md) | Last screening health snapshot (while screening was live) |
| [`AUTHORITY_REBIND_RUNBOOK.md`](AUTHORITY_REBIND_RUNBOOK.md) | Sealed authority/image rebind ceremony |

CPU entry: `/root/cacheon-ops/chainops/START_HERE.md`  
m4l script template: `/data/stage/mainnet-m4/chain_mainnet14_m4l.sh` (on pod)
