Done. The handoff is now CPU-native and does not require access to this computer, `/tmp`, GitHub, this Codex task, or any local file.

The canonical operator entry point is:

/root/cacheon-ops/chainops/START_HERE.md

A second top-level pointer is installed at:

/root/cacheon-ops/HANDOFF_START_HERE.md

The complete CPU-side documentation directory is:

/root/cacheon-ops/chainops

It contains:

00_MAINNET_LAUNCH_LEDGER.md
START_HERE.md
CPU_VM_START_HERE.md
HANDOFF_LAUNCH_NIGHT.md
MAINNET_HANDOFF_2026-08-04_2100_IST.md
MAINNET_EMERGENCY_HANDOFF_2026-08-04.md
MAINNET_OPERATOR_RUNBOOK.md
LIUM_WORKER_BOOTSTRAP.md
RESIDENT_REVIEW_FINDINGS_0329aaaa.md
mainnet_calibration_pairing_fix_20260804T071246Z.json
mainnet_m4k_corrected_program_receipt_20260804T073231Z.json
mainnet_m4l_preflight_corrected_program_receipt_20260804T074845Z.json
mainnet_recovery_receipt_20260804T040944Z.json
mainnet_resident_worker_deploy_receipt_20260804T145019Z.json
verify_cpu_handoff.sh
SHA256SUMS

The exact source repository is already materialized on the CPU VM at:

/root/cacheon-ops/handoff/20260804T1530Z/cacheon-c066360b

It is detached at and verified against:

c066360b9c875a999676cf995c8624956869e66d

The self-contained artifact/evidence root is:

/root/cacheon-ops/handoff/20260804T1530Z

It includes the full Git bundle, source archive, patch, consistent SQLite snapshot, live VM/pod logs, process snapshots, worker configuration, runbooks, and checksums.

A replacement operator can verify everything directly on the CPU VM with:

/root/cacheon-ops/handoff/20260804T1530Z/verify_cpu_handoff.sh

That verifier passed completely:

snapshot_reservations=[('failed', 2), ('held', 3), ('published', 32)]
HANDOFF-VERIFY-OK
commit=c066360b9c875a999676cf995c8624956869e66d

The deployment source remains clean and separate from the private chainops documents, so reading or updating the operator documentation cannot contaminate the exact committed source used for pod commissioning.