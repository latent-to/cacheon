# Mainnet FIFO screening — status snapshot

**Checked:** 2026-08-04 22:26 UTC (CPU VM `date -u`)
**Scope:** Live screening health only. §10 proof already passed; subnet is **not**
green; §11 qualification/crown/settlement remains unarmed. This is a read-only
observation, no writes/mutations were performed.

---

## Epoch

- Pinned epoch `4d3df000acc41525f79a22cdc2a05a9c` — **still current**, no
  successor rotation observed.

## Dispatcher

- tmux session `cacheon-mainnet-screen-dispatcher` on CPU VM — **live**
  (created Aug 4 21:57:11 UTC, still present in `tmux ls`).
- Only one dispatcher session for this epoch — no second-dispatcher violation.
- Log `/root/cacheon-ops/logs/mainnet-screen-dispatcher-4d3df000acc41525f79a22cdc2a05a9c.log`
  actively growing; last event at `time_unix=1785882311` (22:25:11 UTC), ~1
  minute before the check — consistent with an active dispatch loop.

## Heartbeat (pod, via CPU hop)

| Field | Value |
|---|---|
| `adapter_alive` | `true` |
| `adapter_start_count` | `1` (unchanged since §10 proof — resident identity stable) |
| `consecutive_adapter_failures` | `0` |
| `state` | `evaluating` |
| `worker_epoch` | `4d3df000acc41525f79a22cdc2a05a9c` (matches pin) |
| `time_unix` | `1785882381` (22:26:21 UTC — fresh) |

## Image digest

- Ready-receipt `worker_image`:
  `localhost:5000/cacheon-b300-worker@sha256:cfc0c7a3660bafe3f0ac63ede72c101d42dde50e5cc413e5fb92eb6eae096af9`
- **Match: YES** — exact match to the cutover digest (`cfc0c7a3660b…`) pinned in
  `CONTEXT_MAINNET_FIFO.md`. No image drift.
- Ready-receipt `state`: `READY_FOR_REGISTRATION`; source revision
  `5970bc63e61ea1d026ee2f4f929bb16e0331dc8a` (image-rotate commit lineage).

## Resident intake (pod)

- `commissioned/resident-intake` is a directory (per-reservation swap staging),
  perms `0711` (`drwx--x--x`) — matches the intake-permission fix in
  `CONTEXT_MAINNET_FIFO.md` (was `0700`/EACCES before the fix).
- 18 active per-reservation subdirectories, most recently modified 22:22 UTC —
  consistent with ongoing intake activity. No explicit `mode` marker file found
  (not expected — directory listing is the mode signal here).

## Recent screen outcomes (pinned-epoch log)

- 40 `screen_completed` events observed in the tail window, **all**
  `disposition:"completed"` — zero `failed`/other dispositions in this window.
- `epoch_failed`: **0 occurrences** in the full log (hard-stop condition not
  triggered).
- 6 `dispatcher_restart` events, all `error_type:"_TransientCoordinatorOwnership"`
  (`"evaluation store lock/cursor did not stabilize within retry bounds"`),
  each followed within 1-2s by `dispatcher_ready` — self-recovering, did not
  affect `adapter_start_count` (still `1`). Benign transient pattern, not an
  escalation.

## Intake reservation status counts (read-only SQLite query)

```
expired    24
failed     12
held       15
promoted    6
published   7
```

(Total 64 reservations tracked. No chain-walk or raw-SQL lease recovery
performed — counts only.)

## Verdict: **PASS**

Screening is healthy for the pinned epoch: dispatcher live, single instance,
resident identity stable (`adapter_start_count=1`, `adapter_alive=true`, zero
consecutive adapter failures), worker image matches the cutover digest exactly,
and the log shows continuous successful screen completions with no
`epoch_failed` events.

## Recommended follow-up

- Keep an eye on the periodic `_TransientCoordinatorOwnership` dispatcher
  restarts — currently self-healing and not affecting resident identity, but
  worth a trend check if frequency increases.
- No action needed on the `failed`/`expired` intake counts under current
  constraints (no chain-walk/lease surgery); revisit only if `held` count grows
  unusually or `epoch_failed` appears.
- Continue P1 (sync local↔Git file mirrors) and P2 (authority-rebind runbook)
  from `CONTEXT_MAINNET_FIFO.md` when explicitly requested — no §11 action.
