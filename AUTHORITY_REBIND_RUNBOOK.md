# B300 mainnet sealed authority-rebind runbook

Operator ceremony for rebinding the sealed FIFO screen `primary-authority`
(and its nested device-execution receipt) to a new worker image. This is
**not** production authority itself — it documents a procedure. The sealed
files on the pod, the CPU launch ledger, and
`docs/reference/state-of-record.md` remain the authoritative record of any
given rebind. See [`CONTEXT_MAINNET_FIFO.md`](CONTEXT_MAINNET_FIFO.md) and
[`TODO_MAINNET_FIFO.md`](TODO_MAINNET_FIFO.md) for current operator
continuity state.

Scope: screening-only (§10). This runbook does not authorize qualification,
crown, settlement, or weight publication (§11), and does not itself make any
claim about subnet "green" status.

---

## 1. When a rebind is required

The sealed `primary-authority` directory pins **exact** worker identity
fields (image, local image id, runtime/base-engine/validator-overlay
digests, worker distribution digest) inside `authority-config.json` and
`measurement-config.json`, and pins a device-execution SHA-256 that includes
a nested runtime-preflight receipt
(`cacheon/eval/b300_screen_deployment.py::_derive_inputs` /
`_same_authority_identity` / `_find_preflight`). Deployment fails closed
("sealed worker authority differs from runtime preflight or READY") if any
pinned field does not exactly match a freshly captured preflight against the
image actually running.

| Situation | Action |
|---|---|
| Same source revision, same worker image digest | No rebind. Ordinary `--commission-current-pod` recommission (or plain restart) is enough. |
| Same source revision, **different** worker image digest (e.g. ABI-cutover rebuild with `_PRE_CUTOVER_ABI_SHA256`) | **Full rebind required.** `commission-current-pod` refuses to silently overwrite the existing commission receipt for the same revision when the image differs (`chainops/remote_worker_service.py` commission path: `"current pod identities differ from its existing commission receipt"`), and even if it did, the *sealed* primary-authority worker block would still pin the old image/digests and fail deployment with "identities differ" style errors at swap/launch time. |
| Different source revision | Full rebind (worker + possibly topology/model sections), same procedure below, plus normal source-revision recommissioning. |
| Only the resident hot-swap candidate bundle is changing (miner ABI payload) | **No** authority rebind. That is ordinary FIFO screening traffic through the routing-only resident swap, unrelated to sealed worker identity. |

Do not attempt to patch only `worker_image` in a running registration or
launcher invocation and skip the sealed authority files — the authority
cross-checks (`_same_authority_identity`, worker-image/local-image-id/digest
equality in `_derive_inputs`) will fail closed on the next deployment
attempt, and mid-ceremony half-states are exactly what backups exist to
recover from.

---

## 2. Backup steps (before touching anything sealed)

Perform on the pod (`root@204.9.206.240:40050`), read-write, but strictly
additive — never edit sealed files in place without a backup copy existing
first.

1. Choose one timestamped backup directory following the established
   pattern:

   ```text
   /data/cacheon-b300/authority-rebind-abi-cutover-<UTC timestamp or short label>/
   ```

2. Copy, don't move, the full sealed authority directory and the nested
   receipt into the backup directory, preserving relative structure:
   - `/data/cacheon-b300/launch-b300-v3-m4l/primary-authority-v3-m4l/authority-config.json`
   - `/data/cacheon-b300/launch-b300-v3-m4l/primary-authority-v3-m4l/measurement-config.json`
   - `/data/cacheon-b300/proofs/shallow-b300-graph-v3-m4l/receipt.json`
3. Record, in a small backup manifest alongside the copies (or in the ops
   ledger), at minimum:
   - source worker image digest (the one being replaced),
   - target worker image digest (the cutover image),
   - reason for rebind (one line — e.g. "ABI cutover, image-only recommission
     fails: identities differ"),
   - SHA-256 of each file backed up.
4. Do not delete or rotate any prior backup directory. Backups are
   append-only evidence, not scratch space.
5. Confirm (read-only `stat`/`sha256sum`, no writes) that the backup files
   are byte-identical to the live sealed files before proceeding to step 3.

---

## 3. Capture a fresh nested `worker.preflight` against the target image

The device-execution receipt
(`/data/cacheon-b300/proofs/shallow-b300-graph-v3-m4l/receipt.json`) must
contain exactly one embedded runtime-preflight row with
`schema == "cacheon-runtime-preflight-v2"` (`HOST_RECEIPT_SCHEMA` in
`cacheon/eval/runtime_preflight.py`), found by recursive search
(`_find_preflight`) — conventionally nested at `$.worker.preflight` per
existing ops layout, though the loader does not hard-code that path; it
requires the row to be the *sole* match anywhere in the receipt.

Steps:

1. **Before** editing any sealed file, run a fresh preflight probe against
   the **target** (cutover) image only —
   `localhost:5000/cacheon-b300-worker@sha256:cfc0c7a3...096af9` — using the
   validator-owned probe (`cacheon.eval.runtime_preflight.run_runtime_preflight`
   / `RuntimePreflightConfig`). This launches a bounded, no-GPU, read-only
   container inspection; it does not run an evaluator, profiler, or kernel.
2. Confirm the resulting receipt's `requested_image` equals the exact
   cutover digest reference (not a mutable tag), and that
   `local_image_id`, `worker_distribution_digest`, `runtime_digest` /
   `base_engine_digest` / `validator_overlay_digest` all reflect the new
   image — not stale values copied from the old receipt.
3. Only after this fresh capture succeeds should you rewrite the sealed
   receipt/authority files (step 4). Never hand-edit only the `image` string
   while leaving digests from the old preflight capture in place — that
   produces an internally-inconsistent authority that will fail
   `_same_authority_identity` or the worker-identity check in
   `_derive_inputs`, or (worse) silently pins a preflight that does not
   correspond to what is actually running.

---

## 4. Fields to update

All of the following must move together, atomically, to the **same** new
values (no partial rebind):

**`authority-config.json`** (`$.worker`):

- `image` — new cutover `repo@sha256:...` reference
- `local_image_id` — from fresh preflight
- `runtime_digest` — from fresh preflight (`runtime_identity_from_preflight`)
- `base_engine_digest` — from fresh preflight
- `validator_overlay_digest` — from fresh preflight
- `worker_distribution_digest` — from fresh preflight

**`authority-config.json`** (`$.device_execution`):

- `sha256` — SHA-256 of the rewritten nested receipt file (must match the
  file on disk exactly; `_derive_inputs` re-hashes and compares)
- `path` — only if the receipt file itself is relocated (normally unchanged)

**`measurement-config.json`**:

- Same `worker.*` fields as above, and the same `device_execution.sha256` —
  `_same_authority_identity` requires `authority-config.json` and
  `measurement-config.json` to agree exactly on `topology`, `model`,
  `worker`, `prompt`, and `device_execution` sections. A rebind that updates
  one file and not the other will fail closed immediately on next
  deployment.

**Nested receipt** (`.../shallow-b300-graph-v3-m4l/receipt.json`):

- Replace the embedded preflight row (`$.worker.preflight` by convention)
  with the exact fresh capture from §3. Do not hand-splice individual
  fields inside it.

**READY receipt** (`/data/cacheon-b300/worker-bootstrap/ready-receipt.json`,
produced by commissioning, not hand-edited):

- `worker_image` must equal the same new digest. `_derive_inputs` requires
  `authority.worker.image == ready.worker_image == preflight.requested_image`
  — all three must agree.

Do not update only a subset of these fields "for now." Any mismatch between
`authority-config.json`, `measurement-config.json`, the nested receipt, and
the READY receipt fails closed at the next deployment attempt (by design —
this is the fail-closed check that made the image-only recommission
impossible in the first place).

---

## 5. READY / commission rotation expectation

`chainops/start_mainnet_remote_worker.sh` (`--commission-current-pod` path)
rotates the prior immutable commission when **either** the source revision
**or** the worker image changes — not source revision alone. The relevant
guard (pod-side, inside the script) treats same-revision-same-image as a
no-op and everything else as requiring rotation:

```bash
# Same source revision with a different sealed worker image must rotate too:
# commission-current-pod binds image identity into READY and refuses in-place
# overwrite (image-only ABI/cutover rollouts hit this path).
if [[ "$old_revision" == "$new_revision" && "$old_image" == "$new_image" ]]; then
  exit 0
fi
```

Expect, for an image-only cutover:

- the prior commission (`commissioned/`, `commission-worker-epoch`,
  `ready-receipt.json`) archived under
  `/data/cacheon-b300/worker-bootstrap/retired-commissions/<old_epoch>/`
  (not deleted),
- a new READY receipt bound to the new image digest and a new
  `worker_epoch`,
- the launcher then rotating the pod tmux worker session, the CPU transfer
  dispatcher, and the standing FIFO screen dispatcher onto the new epoch (it
  refuses to run two dispatchers on the same epoch — see §7).

If you run `start_mainnet_remote_worker.sh --commission-current-pod` and it
completes **without** archiving a prior commission when you expected a
rotation, stop — that means the script did not see the image change (check
`--pod-worker-image` was actually passed as the new digest, and that the
sealed authority you rebuilt in §4 matches it) and you likely have a stale
launch in progress, not a successful rebind.

---

## 6. Verification checklist (post-rebind, before declaring done)

All read-only. Do not proceed to any write if a check fails — resolve
before continuing:

- [ ] New `worker_epoch` present in `current-registration.json` on the CPU
      VM and in the pod's READY receipt; both agree.
- [ ] Pod heartbeat (`/data/cacheon-b300/remote-worker/heartbeat.json`)
      reports the new `worker_epoch` and `adapter_alive: true`.
- [ ] `adapter_start_count == 1` for the new epoch (resident identity has
      not silently restarted).
- [ ] No `epoch_failed` in the screen dispatcher log
      (`/root/cacheon-ops/logs/mainnet-screen-dispatcher-<epoch>.log`) or
      the CPU transfer log
      (`/root/cacheon-ops/logs/remote-dispatch-<epoch>.log`).
- [ ] At least one FIFO row screens end-to-end under the new epoch without
      `ManifestError` (screen smoke test) — a promotion or a clean
      non-promotion verdict both count as a healthy smoke result; a hard
      infra/ManifestError does not.
- [ ] Confirm via the outcome/heartbeat evidence that this is still
      screening-only: no qualification, crown, settlement, or weight
      publication artifact was produced as a side effect.
- [ ] Update the CPU launch ledger
      (`/root/cacheon-ops/chainops/00_MAINNET_LAUNCH_LEDGER.md`) with the
      new epoch, image digest, and backup directory path.
- [ ] Update `docs/reference/state-of-record.md` if this rebind changes the
      dated evidence claims already recorded there (only when committing;
      not required to complete this runbook).

---

## 7. Hard don'ts

- Do not run a second screen dispatcher against the same or a different
  epoch concurrently. Stop the old one cleanly before starting the new one
  (the launcher does this automatically during rotation — do not race it by
  hand).
- Do not call the subnet "green" because a rebind + screen smoke test
  succeeded. Screening ≠ qualification ≠ crown ≠ settlement ≠ weight
  publication.
- Do not splice authorities — never take `worker.*` fields from one
  authority/measurement/receipt triple and other sections (topology, model,
  prompt, device_execution) from a different, unrelated capture. The whole
  set in §4 must originate from one coherent rebind against one target
  image.
- Do not treat a resident-swap screen promotion (abbreviated-serving,
  routing-only) as crown or qualification evidence. It cannot authorize
  rewards.
- Do not hand-edit the nested preflight row's individual fields to "fix up"
  a mismatch. If it doesn't match, recapture it (§3) against the actual
  running image.
- Do not delete or overwrite a backup directory to "clean up." Backups are
  append-only ceremony evidence.
- Do not skip capturing a fresh preflight and instead reuse an old
  preflight capture with only the `image` string edited — this is exactly
  the failure mode ("identities differ") this ceremony exists to avoid.
- Do not start §11 (qualification, crown, settlement, weight publication)
  as part of, or immediately following, this ceremony unless explicitly
  ordered separately.

---

## Verification performed while writing this runbook

This runbook was produced by reading the committed source of truth in this
repository (read-only, no pod writes):

- `chainops/start_mainnet_remote_worker.sh` — commission/rotation guard
  (source revision **or** image change triggers rotation; archived, not
  overwritten).
- `cacheon/eval/b300_screen_deployment.py` — `_same_authority_identity`,
  `_derive_inputs`, `_find_preflight` — exact authority/measurement field
  names and the fail-closed cross-checks between `authority-config.json`,
  `measurement-config.json`, the nested device-execution receipt, and the
  READY receipt.
- `cacheon/eval/runtime_preflight.py` — `HOST_RECEIPT_SCHEMA`,
  `RuntimePreflightConfig`, `RuntimePreflightReceipt`, `run_runtime_preflight`
  — nested preflight schema and capture inputs.
- `chainops/remote_worker_service.py` — commissioning receipt identity
  check ("current pod identities differ from its existing commission
  receipt").
- `CONTEXT_MAINNET_FIFO.md`, `TODO_MAINNET_FIFO.md`,
  `docs/reference/state-of-record.md` — prior operator record of the
  2026-08-04 ABI-cutover rebind this runbook generalizes.

**Live path confirmation (read-only, 2026-08-04 via CPU→pod SSH):**

- Backup dir present:
  `/data/cacheon-b300/authority-rebind-abi-cutover-20260804T215516Z/`
  containing `authority-config.json`, `measurement-config.json`, `receipt.json`.
- Sealed primary-authority files present and mode `0400`:
  `authority-config.json`, `measurement-config.json` under
  `/data/cacheon-b300/launch-b300-v3-m4l/primary-authority-v3-m4l/`.
- Nested receipt
  `/data/cacheon-b300/proofs/shallow-b300-graph-v3-m4l/receipt.json`
  has `worker.preflight.schema = cacheon-runtime-preflight-v2`.
- Live `worker.image` on authority, measurement-config, and nested receipt
  all match cutover digest `cfc0c7a3660b…`; worker keys match the field list
  in §4 (`image`, `local_image_id`, `runtime_digest`, `base_engine_digest`,
  `validator_overlay_digest`, `worker_distribution_digest`).
