# Mainnet FIFO — remaining TODO

Companion to [`CONTEXT_MAINNET_FIFO.md`](CONTEXT_MAINNET_FIFO.md).
§10 proof is done (epoch `4d3df000…`). Screening-only — not green.

## Done

- [x] Watchdog / intake perms / swap error / ABI image / launcher image rotate / authority rebind
- [x] §10 two durable screens + ledger outcome (`adapter_start_count=1`)

## Next

1. **Monitor** live screen dispatcher / heartbeat on current epoch (keep `adapter_start_count==1`).
   Checked 2026-08-04 22:26 UTC: dispatcher live, epoch unchanged (`4d3df000…`),
   heartbeat healthy (`adapter_start_count=1`, `adapter_alive=true`, 0 consecutive
   failures), worker image matches cutover digest, log shows only `completed`
   screens (0 `epoch_failed`). PASS — see `STATUS_MAINNET_FIFO.md`.
2. **Sync local → Git** (when asked): mirror VM commits `c49d7fd` / `33db68a` / `5970bc6` + tests/docs.
   Verified 2026-08-04: all five local mirrors present and match intent (watchdog
   `1800` + intake `0711`, inaccessible-swap error, launcher image-change rotate,
   §10/FIFO/ABI docs note, watchdog+intake tests). `tests/test_b300_screen_deployment.py`
   passes 6/6 on CPU-only venv. Local mirrors are commit-ready; **still awaiting
   explicit commit ask** — HEAD unchanged at `c066360`.
3. **Authority-rebind runbook** — [`AUTHORITY_REBIND_RUNBOOK.md`](AUTHORITY_REBIND_RUNBOOK.md)
   drafted from source + live-validated 2026-08-04 (backup
   `authority-rebind-abi-cutover-20260804T215516Z`, sealed authority/measurement/
   nested receipt paths and worker field names match; image `cfc0c7a3660b…`).
   - [x] Validate paths/fields against live pod backup dirs and sealed authority files.
4. **§11 only if ordered** — qualification → reproduction → crown/settlement → weights → green.

## Do not

- Second dispatcher on same epoch · chain-walk / raw SQL lease recovery · call subnet green after screen-only · treat screen promotion as crown.
