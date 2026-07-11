"""Optima validator CLI — drives the submission pipeline end to end.

    python -m optima.cli slots
    python -m optima.cli scan      <bundle>
    python -m optima.cli verify    <bundle> [--dtype bfloat16] [--device cuda]
    python -m optima.cli evaluate  <bundle> --model <path> [--max-new-tokens 128]

Pipeline (mirrors the validator flow):

    manifest -> static scan -> (isolated) load -> op-correctness -> register
             -> build engine -> baseline vs candidate -> throughput + KL -> score

SECURITY NOTE: ``verify`` and ``evaluate`` import the miner module, which runs
its code in THIS process. That is only acceptable because the whole validator
host is expected to be the sandbox (no network, per-eval GPU context, watchdog).
Do not run this on a machine you care about without that isolation. See
``optima/sandbox.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from optima.manifest import (
    all_declared_cuda_sources,
    all_declared_dep_patches,
    all_declared_system_patches,
    load_manifest,
    resolve_source,
)
from optima.sandbox import scan_path


def _json_obj(raw: str | None) -> dict:
    if not raw:
        return {}
    out = json.loads(raw)
    if not isinstance(out, dict):
        raise argparse.ArgumentTypeError("JSON value must be an object")
    return out


def _parse_gpu_devices(raw: str | None, *, expected: int) -> tuple[int, ...]:
    if not raw:
        raise ValueError(
            f"production OCI evaluation requires --oci-gpus with {expected} IDs"
        )
    pieces = raw.split(",")
    if any(not piece.strip().isdigit() for piece in pieces):
        raise ValueError("--oci-gpus must be comma-separated non-negative integers")
    devices = tuple(int(piece) for piece in pieces)
    if len(devices) != expected or len(set(devices)) != len(devices):
        raise ValueError(
            f"--oci-gpus must contain exactly {expected} distinct device IDs"
        )
    return devices


def _registered_referee_source(args, arena) -> Path:
    """Build/verify the immutable source release used by crownable OCI paths."""

    from optima.referee_runtime import resolve_referee_runtime

    checkout_or_release = Path(
        args.oci_source_dir or Path(__file__).resolve().parents[1]
    )
    publication = Path(args.oci_release_root).expanduser()
    return resolve_referee_runtime(
        checkout_or_release,
        publication,
        expected_tree_digest=arena.referee_tree_digest,
        expected_referee_source_digest=arena.referee_source_digest,
    ).root


def _retained_host_attestation_verifier(publication_root):
    """Return the standalone store verifier used by every emission path."""

    from optima.eval.host_attestation import verify_host_attestation

    root = Path(publication_root).expanduser().resolve()

    def verify(reference, expected_context):
        return verify_host_attestation(
            root, reference, expected_context=expected_context
        )

    return verify


def _registered_oci_launcher(args, arena, bundle, competition):
    """Create and prebuild the only crownable direct-evaluate launch backend."""
    from optima.bundle_hash import content_hash
    from optima.eval.oci_backend import OCILauncher, profile_for_arena

    devices = _parse_gpu_devices(args.oci_gpus, expected=arena.tp_size)
    source = _registered_referee_source(args, arena)
    model = Path(args.oci_model_dir or arena.model_path)
    artifact_root = Path(args.oci_artifact_root).expanduser().resolve()
    scratch_root = Path(args.oci_scratch_root).expanduser().resolve()
    artifact = artifact_root / arena.fingerprint / content_hash(bundle)
    artifact.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    profile = profile_for_arena(
        arena,
        source_dir=source,
        model_dir=model,
        artifact_dir=artifact,
        scratch_root=scratch_root,
        gpu_devices=devices,
        bundle_dir=bundle,
        competition_target=competition.target,
    )
    launcher = OCILauncher(profile)
    launcher.prebuild_candidate_artifacts()
    return launcher


def _publish_direct_host_attestation(args, launcher, prepared_qualification):
    """Publish the retained host evidence for a settlement-facing direct eval."""

    from optima.eval.host_attestation import publish_host_attestation
    from optima.eval.oci_backend import OCIBackendError

    runtime_receipt = getattr(launcher, "runtime_preflight_receipt", None)
    if runtime_receipt is None:
        raise OCIBackendError(
            "crownable direct launcher lost its stock runtime preflight receipt"
        )
    return publish_host_attestation(
        Path(args.oci_artifact_root).expanduser().resolve(),
        context=prepared_qualification.attestation_context(),
        runtime_preflight=runtime_receipt.canonical_payload(),
        device_receipts=launcher.attestation_receipts,
        qualification_evidence=prepared_qualification.evidence_dict(),
    )


def _framework_mode_for_manifest(args: argparse.Namespace, manifest) -> bool:
    """Select the external fidelity lane from bundle capabilities, not operator memory."""
    setup_slots = tuple(op.slot for op in manifest.ops if op.setup)
    framework_mode = bool(args.framework_mode or setup_slots)
    if setup_slots and not args.framework_mode:
        print("  [policy] setup() declared for " + ", ".join(setup_slots)
              + "; forcing framework token fidelity + candidate isolation")
    return framework_mode


def _strictest_kl_threshold(
    member_slots: tuple[str, ...],
    *,
    advisory: bool,
    fallback: float,
) -> float | None:
    """Use the strictest effective KL policy across a competition's members."""
    if advisory:
        return None
    from optima.slots import get_slot

    thresholds = []
    for member in member_slots:
        calibrated = get_slot(member).kl_threshold
        thresholds.append(calibrated if calibrated is not None else fallback)
    if not thresholds:
        raise ValueError("competition target has no member slots")
    return min(thresholds)


# Registered arenas own every score-affecting knob. ``--prompt-seed`` is the one
# dynamic input: the chain loop derives it from post-commit block entropy and the
# qualification report stamps it for independent cross-checking.
_ARENA_CONTROLLED_EVALUATE_OPTIONS = {
    "--model", "--dtype", "--max-new-tokens", "--num-prompts",
    "--timed-iters", "--warmup-iters", "--conditioning-iters",
    "--speedup-margin", "--input-len",
    "--top-logprobs", "--ignore-eos", "--no-ignore-eos", "--kl-threshold",
    "--argmax-disagree-rate", "--p99-kl-threshold", "--kl-advisory",
    "--fidelity-mode", "--audit-rate", "--mem-fraction", "--no-deterministic",
    "--attention-backend", "--candidate-attention-backend",
    "--disable-cuda-graph", "--tp-size", "--max-running-requests",
    "--moe-runner-backend", "--candidate-moe-runner-backend",
    "--disable-custom-all-reduce", "--candidate-disable-custom-all-reduce",
    "--no-candidate-disable-custom-all-reduce",
    "--engine-kwargs-json", "--candidate-engine-kwargs-json", "--framework-mode",
    "--token-match-threshold", "--isolate", "--no-isolate",
    "--allow-unsafe-no-isolation",
}


def _dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def cmd_slots(_: argparse.Namespace) -> int:
    from optima.slots import SLOTS, list_slots

    print("Registered op-slots (the submission ABI):")
    for name in list_slots():
        spec = SLOTS[name]
        print(f"  {name}  [{spec.kind}]")
        print(f"      {spec.summary}")
    return 0


def cmd_compat(_: argparse.Namespace) -> int:
    from optima.compat import format_checks, run_checks

    checks = run_checks()
    print("sglang compatibility canary (run after any sglang bump):")
    print(format_checks(checks))
    return 0 if all(c.ok for c in checks) else 2


def cmd_chain_compat(_: argparse.Namespace) -> int:
    from optima.chain_canary import format_checks, run_checks

    checks = run_checks()
    print("bittensor chain-SDK canary (introspects the installed SDK; no network):")
    print(format_checks(checks))
    return 0 if all(c.ok for c in checks) else 2


def cmd_set_weights(args: argparse.Namespace) -> int:
    from optima import chain
    from optima.arenas import ARENAS, ArenaPolicyError, get_arena
    from optima.chain.validator_loop import (
        LedgerLockError,
        WeightSafetyError,
        _archive_released_weight_hold,
        _exclusive_ledger_pass,
        _global_arena_set_sha256,
        _load_weights_state,
        _validate_weight_publication_state,
        _weight_state_path,
    )
    from optima.commit_reveal import Ledger, make_chain_scope

    try:
        arena = get_arena(args.arena)
    except ArenaPolicyError as exc:
        print(f"REFUSED: {exc}")
        return 2
    subtensor = chain.connect(args.network)
    import bittensor as bt

    validator_wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    validator_hotkey = validator_wallet.hotkey.ss58_address
    try:
        with _exclusive_ledger_pass(args.ledger):
            led = Ledger.load(args.ledger)
            expected_scope = make_chain_scope(
                genesis_hash=str(subtensor.get_block_hash(0)),
                netuid=int(args.netuid),
                scheme=arena.settlement.chain_scope_scheme,
            )
            led.bind_chain_scope(expected_scope)
            led.bind_validator_hotkey(validator_hotkey)
            registered_arenas = tuple(ARENAS[name] for name in sorted(ARENAS))
            arena_set_sha256 = _global_arena_set_sha256(registered_arenas)
            if args.release_publication_hold:
                if args.dry_run:
                    print(
                        "REFUSED: publication-hold release and --dry-run are "
                        "mutually exclusive"
                    )
                    return 2
                state_path = _weight_state_path(args.ledger, expected_scope)
                state = _load_weights_state(state_path)
                _validate_weight_publication_state(
                    state,
                    chain_scope=expected_scope,
                    arena_set_sha256=arena_set_sha256,
                    emission_policy=arena.settlement.emission_policy,
                )
                archive = _archive_released_weight_hold(
                    state_path,
                    release_block=int(subtensor.get_current_block()),
                    reason=args.release_reason or "",
                )
                print(
                    "released held weight publication after explicit operator "
                    f"audit; archived={archive}"
                )
                return 0
            if args.release_reason:
                print(
                    "REFUSED: --release-reason requires "
                    "--release-publication-hold"
                )
                return 2
            active_weights = chain.read_validator_weights(
                subtensor, args.netuid, validator_hotkey
            )
            # This operator path must project the same complete registered-arena
            # authority as ``run_pass``. ``--arena`` validates the chain-scope
            # policy for compatibility; it never narrows the emitted vector.
            weights = led.current_weights_across_arenas(
                registered_arenas,
                host_attestation_verifier=_retained_host_attestation_verifier(
                    args.oci_artifact_root
                ),
                validator_hotkey=validator_hotkey,
            )
            if not weights:
                if active_weights:
                    print(
                        "REFUSED: no retained valid global champion authority, but "
                        "this validator still has an active on-chain weight vector; "
                        "requalify or use an explicitly registered neutralization policy"
                    )
                    return 2
                print(f"no champion(s) in {args.ledger}; nothing to weight")
                return 1
            if not args.dry_run:
                print(
                    "REFUSED: production weight publication is journaled by "
                    "chain-validate; standalone set-weights is inspection-only"
                )
                return 2
            res = chain.set_weights(
                subtensor, None, args.netuid, weights, dry_run=True
            )
            print(f"DRY RUN (network={args.network}, netuid={args.netuid}): "
                  f"would set uids={res['uids']} weights={res['weights']}")
            return 0
    except (LedgerLockError, WeightSafetyError) as exc:
        print(f"REFUSED: {exc}")
        return 2


def cmd_chain_package(args: argparse.Namespace) -> int:
    from optima.chain.fetch import package_bundle

    out, ch = package_bundle(args.bundle, args.out)
    print(f"archive:      {out}")
    print(f"content_hash: {ch}")
    print("host the archive at a stable URL, then commit it: optima chain-submit "
          f"{args.bundle} --url <URL> --netuid <N> --network <WSS>")
    return 0


def cmd_chain_submit(args: argparse.Namespace) -> int:
    from optima.chain.payload import PayloadError
    from optima.chain.submit import submit_bundle

    from optima import chain

    subtensor = wallet = None
    if not args.dry_run:
        import bittensor as bt

        subtensor = chain.connect(args.network)
        wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    try:
        res = submit_bundle(subtensor, wallet, args.netuid, args.bundle, args.url,
                            blocks_until_reveal=args.blocks_until_reveal,
                            dry_run=args.dry_run)
    except PayloadError as e:
        print(f"REFUSED before signing: {e}")
        return 2
    print(f"content_hash: {res['content_hash']}")
    print(f"payload:      {res['payload']}")
    if args.dry_run:
        print("DRY RUN — nothing sent. The payload above is what would be committed "
              f"(timelock, reveals after {args.blocks_until_reveal} blocks).")
        return 0
    ok = bool(res.get("submitted"))
    print(f"set_reveal_commitment submitted={ok} "
          f"(reveals after {args.blocks_until_reveal} blocks; the validator picks it "
          "up on its next pass after the reveal)")
    return 0 if ok else 1


def cmd_chain_status(args: argparse.Namespace) -> int:
    from optima import chain
    from optima.chain.payload import decode_payload

    subtensor = chain.connect(args.network)
    block = int(subtensor.get_current_block())
    print(f"network: {args.network}  netuid: {args.netuid}  block: {block}")
    mg = chain.fetch_metagraph(subtensor, args.netuid)
    print(f"neurons: {len(mg.uids)}  permits: {sum(mg.validator_permit)}")
    if args.wallet:
        import bittensor as bt

        wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
        hk = wallet.hotkey.ss58_address
        uid = mg.uid_of(hk)
        permit = bool(uid is not None and uid < len(mg.validator_permit)
                      and mg.validator_permit[uid])
        print(f"our hotkey {hk}: uid={uid} permit={permit}")
    revealed = chain.read_revealed_commitments(subtensor, args.netuid)
    print(f"revealed commitments: {len(revealed)}")
    for hk, rc in sorted(revealed.items(), key=lambda kv: kv[1].block):
        ref = decode_payload(hk, rc.block, rc.data)
        if ref is None:
            print(f"  block {rc.block}  {hk}  (unparseable payload)")
        else:
            print(f"  block {rc.block}  {hk}  {ref.content_hash[:16]}…  {ref.url}")
    return 0


def cmd_chain_validate(args: argparse.Namespace) -> int:
    import logging

    from optima import chain
    from optima.arenas import ArenaPolicyError, get_arena
    from optima.chain.validator_loop import (
        command_evaluator,
        oci_evaluator,
        run_validator,
        verify_evaluator,
    )
    from optima.eval.host_attestation import verify_host_attestation

    try:
        arena = get_arena(args.arena)
    except ArenaPolicyError as exc:
        print(f"REFUSED: {exc}")
        return 2
    provided = set(getattr(args, "_provided_options", ()))
    if "--margin" in provided:
        print("REFUSED: --margin is owned by the immutable arena settlement policy; "
              f"this arena uses {arena.settlement.dethrone_margin:g}")
        return 2
    try:
        gpu_devices = _parse_gpu_devices(args.oci_gpus, expected=arena.tp_size)
    except ValueError as exc:
        gpu_devices = ()
        if not args.eval_cmd and not args.verify_only:
            print(f"REFUSED: {exc}")
            return 2
    if args.eval_cmd and args.oci_gpus:
        print("REFUSED: choose validator-owned OCI evaluation or development --eval-cmd, not both")
        return 2
    if args.verify_only and (args.eval_cmd or args.oci_gpus):
        print("REFUSED: --verify-only cannot be combined with a GPU evaluator")
        return 2
    retained_host_verifier = _retained_host_attestation_verifier(
        args.oci_artifact_root
    )

    margin = arena.settlement.dethrone_margin if not args.verify_only else 0.0
    if gpu_devices:
        try:
            referee_source = _registered_referee_source(args, arena)
            evaluator = oci_evaluator(
                arena=arena,
                source_dir=referee_source,
                model_dir=args.oci_model_dir or arena.model_path,
                artifact_root=args.oci_artifact_root,
                scratch_root=args.oci_scratch_root,
                gpu_devices=gpu_devices,
                timeout_s=(
                    args.eval_timeout if "--eval-timeout" in provided else None
                ),
            )
        except Exception as exc:
            print(f"REFUSED: OCI evaluator preflight failed: {exc}")
            return 2
    elif args.eval_cmd:
        evaluator = command_evaluator(
            args.eval_cmd, arena=arena, timeout_s=args.eval_timeout
        )
        print("NOTE: --eval-cmd is development-only and cannot mint a crown; use "
              "--oci-gpus for validator-owned authenticated qualification")
    else:
        evaluator = verify_evaluator(device=args.eval_device, timeout_s=args.eval_timeout)
        print("NOTE: verify-mode evaluator (pass/fail plumbing score of 1.0) — a 1.0 "
              "never clears the dethrone margin, so crown plumbing runs with "
              "--margin 0; use --oci-gpus for the full GPU gate chain")
    subtensor = chain.connect(args.network, retry_forever=not args.once)
    # Daemon-mode observability: between passes the loop reports only through the
    # "optima.chain.*" loggers (--once prints its own summary below). This must run
    # AFTER connect(): the bittensor import reconfigures global logging — it sets
    # every pre-existing third-party logger's level to CRITICAL (measured in the
    # 2026-07-10 soak: the ledger advanced every pass while the log stayed empty;
    # optima.chain.validator read level=50). Own the subtree outright: reset levels
    # to inherit, dedicated handler, no propagation upward.
    for _name, _lg in list(logging.root.manager.loggerDict.items()):
        if _name.startswith("optima.") and isinstance(_lg, logging.Logger):
            _lg.disabled = False
            _lg.setLevel(logging.NOTSET)
    _chain_lg = logging.getLogger("optima.chain")
    _chain_lg.setLevel(logging.INFO)
    _chain_lg.propagate = False
    if not _chain_lg.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        _chain_lg.addHandler(_handler)
    import bittensor as bt

    validator_wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    validator_hotkey = validator_wallet.hotkey.ss58_address
    wallet = None if args.dry_run_weights else validator_wallet
    res = run_validator(subtensor, wallet, args.netuid, ledger_path=args.ledger,
                        bundles_dir=args.bundles_dir, evaluator=evaluator,
                        arena=arena,
                        margin=margin, interval_s=args.interval, once=args.once,
                        dry_run_weights=args.dry_run_weights,
                        host_attestation_verifier=retained_host_verifier,
                        validator_hotkey=validator_hotkey)
    if args.once and res is not None:
        print(f"pass @block {res.block} (round {res.round_id}): seen={res.seen} "
              f"new={len(res.new)} copies={len(res.copies)} rejected={len(res.rejected)}")
        for ch_, ok in res.evaluated.items():
            print(f"  evaluated {ch_[:16]}… passed={ok}")
        for ch_, why in res.rejected.items():
            print(f"  rejected  {ch_[:16]}… {why}")
        print(
            f"weights: {res.weights}  submitted={res.weights_submitted} "
            f"pending={res.weights_pending} held={res.weights_held} "
            f"confirmed={res.weights_confirmed}"
        )
    return 0


def cmd_chain_register(args: argparse.Namespace) -> int:
    import bittensor as bt

    from optima import chain

    subtensor = chain.connect(args.network)
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    hk = wallet.hotkey.ss58_address
    if subtensor.is_hotkey_registered(hotkey_ss58=hk, netuid=args.netuid):
        print(f"already registered: {hk}")
    else:
        cost = subtensor.recycle(args.netuid)
        print(f"registering {hk} on netuid {args.netuid} (burn ≈ {cost}) …")
        resp = subtensor.burned_register(wallet, args.netuid)
        ok = bool(getattr(resp, "success", resp))
        print(f"burned_register success={ok} {getattr(resp, 'message', '')}")
        if not ok:
            return 1
    for check in chain.preflight(subtensor, wallet, args.netuid):
        print(f"  [{'ok' if check.ok else 'MISSING'}] {check.name}: {check.detail}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from optima.sandbox import scan_tree

    m = load_manifest(args.bundle)
    print(f"bundle: {m.bundle_id}  abi: {m.abi_version}  ops: {len(m.ops)}")
    rc = 0
    for op in m.ops:
        src = resolve_source(args.bundle, op)
        result = scan_path(src)
        status = "clean" if result.ok else "VIOLATIONS"
        print(f"  [{status}] {op.slot} <- {op.source}")
        for v in result.violations:
            print(f"      {v}")
            rc = 2
    # Recursive guard: catch a vendored/extra .py the per-op (entry-only) scan misses, and
    # (fail-closed, manifest now loaded) any file that's neither a scanned .py, a declared
    # cuda_source, nor benign metadata — e.g. an undeclared .cu or a stray .so.
    # Exact file match, not startswith: a prefix filter would also drop violations in
    # e.g. "kernels/silu.py_evil.py" because it string-prefixes "kernels/silu.py".
    op_sources = {op.source for op in m.ops}
    declared_cuda = all_declared_cuda_sources(args.bundle, m)
    declared_patches = all_declared_dep_patches(args.bundle, m)
    declared_system_patches = all_declared_system_patches(args.bundle, m)
    extra = [v for v in scan_tree(args.bundle, declared_cuda_sources=declared_cuda,
                                  declared_dep_patches=declared_patches,
                                  declared_system_patches=(
                                      declared_system_patches
                                  )).violations
             if v.split(":", 1)[0] not in op_sources]
    if extra:
        print("  [VIOLATIONS] vendored/extra/undeclared files (recursive scan):")
        for v in extra:
            print(f"      {v}")
        rc = 2
    return rc


def _recursive_scan_ok(bundle: str, manifest=None) -> bool:
    """Fail-closed vendored-tree guard for the eval paths: scan every bundle .py, not just the
    declared entries (a vendored library .py using open/importlib/subprocess must not slip in
    unscanned). Prints violations; returns False if any.

    ``manifest`` (already loaded by the caller) supplies the declared ``cuda_sources``
    allowlist, so scan_tree runs in its fail-closed mode: any file that's neither a
    scanned ``.py``, a declared cuda_source, nor benign metadata is rejected. Passing
    ``None`` falls back to the old (looser) behavior — kept only for callers that scan
    without a manifest; every call site in this file now has one available.
    """
    from optima.sandbox import scan_tree

    declared_cuda = all_declared_cuda_sources(bundle, manifest) if manifest is not None else None
    declared_patches = (all_declared_dep_patches(bundle, manifest)
                        if manifest is not None else None)
    declared_system_patches = (
        all_declared_system_patches(bundle, manifest)
        if manifest is not None else None
    )
    tree = scan_tree(bundle, declared_cuda_sources=declared_cuda,
                     declared_dep_patches=declared_patches,
                     declared_system_patches=declared_system_patches)
    if not tree.ok:
        print("  [FAIL] recursive policy scan (vendored-tree guard):")
        for v in tree.violations:
            print(f"      {v}")
    return tree.ok


def _declared_model(bundle: str, op) -> str | None:
    """Dev convenience: read the model an op's metadata JSON declares, to pick the
    validator's per-model slot profile when --model isn't given. Never reads thresholds
    from metadata (those are validator-owned in slots.MODEL_PROFILES) — only the model id,
    which selects WHICH validator profile applies. Best-effort; returns None on any issue."""
    if not getattr(op, "metadata", None):
        return None
    try:
        import json
        from pathlib import Path

        meta = json.loads((Path(bundle) / op.metadata).read_text())
        return meta.get("model") or meta.get("model_profile")
    except Exception:
        return None


def _declared_metadata(bundle: str, op) -> dict:
    """Best-effort manifest metadata read for validator-consumed capability flags."""
    if not getattr(op, "metadata", None):
        return {}
    try:
        from pathlib import Path

        value = json.loads((Path(bundle) / op.metadata).read_text())
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def cmd_verify(args: argparse.Namespace) -> int:
    from optima.manifest import VALIDATOR_DEVICE_EXECUTION
    from optima.slots import SLOTS, get_slot, model_profile, slot_for_model
    from optima.verify import format_verify, verify_entry

    m = load_manifest(args.bundle)
    if not _recursive_scan_ok(args.bundle, manifest=m):  # vendored-tree guard (every .py, not just entries)
        return 2
    import torch
    # Mirror the ACTUAL device resolution, including verify_collective's fallback:
    # a collective needs world_size GPUs, so a 1-GPU box silently runs gloo/CPU.
    ws = getattr(args, "world_size", None) or 2
    has_collective = any(op.slot in SLOTS and get_slot(op.slot).kind == "collective"
                         for op in m.ops)
    cuda_ok = torch.cuda.is_available() and (
        not has_collective or torch.cuda.device_count() >= ws)
    resolved_device = args.device or ("cuda" if cuda_ok else "cpu")
    if resolved_device == "cpu":
        print("[note] some or all of this verify runs on CPU: it checks op-correctness "
              "only — it does not predict GPU throughput, CUDA-graph capture, or the "
              "fidelity gates (see docs/GPU_SETUP.md).")
    rc = 0
    for op in m.ops:
        if op.slot not in SLOTS:
            print(f"  [SKIP] {op.slot}: not a known slot on this validator")
            continue
        model_key = args.model or _declared_model(args.bundle, op)
        if model_profile(model_key, op.slot) is not None:
            via = "via --model" if args.model else "declared in metadata"
            print(f"  [profile] {op.slot}: model {model_key!r} ({via}) -> validator slot profile "
                  "(activation + low-bit metric)")
        slot = slot_for_model(op.slot, model_key)
        src = resolve_source(args.bundle, op)
        metadata = _declared_metadata(args.bundle, op)
        graph_safe = None if slot.kind == "op" else bool(metadata.get("graph_safe", False))

        # A validator_device source is CUDA, not an importable Python module.  The
        # recursive declared-source scan above already covered it; route build,
        # trusted ABI loading, and verification through a fresh development child.
        # This remains non-crownable because raw CUDA pointers are not a memory
        # isolation boundary (see optima.device_component).
        if op.execution_class == VALIDATOR_DEVICE_EXECUTION:
            if not str(resolved_device).startswith("cuda"):
                print(
                    f"  [FAIL] {op.slot}:{op.variant}: validator_device "
                    "verification requires CUDA"
                )
                rc = 2
                continue
            from optima.device_component import verify_device_entry_from_bundle
            from optima.eval._launch import call_in_subprocess

            result = call_in_subprocess(
                verify_device_entry_from_bundle,
                str(args.bundle),
                op.slot,
                op.variant,
                dtype_name=args.dtype,
                device=resolved_device,
                seed=args.seed,
                jitter_seed=args.seed,
                model_key=model_key,
            )
            print(format_verify(result))
            if not result.passed:
                rc = 2
            continue

        scan = scan_path(src)
        if not scan.ok:
            print(f"  [FAIL] {op.slot}: failed policy scan")
            for v in scan.violations:
                print(f"      {v}")
            rc = 2
            continue

        if slot.kind == "collective":
            # Collectives span ranks -> distributed verify (spawns world_size ranks;
            # gloo/CPU if device=cpu, nccl/GPU if cuda). No per-op single-process path.
            from optima.verify_collective import verify_collective

            ws = getattr(args, "world_size", None) or 2
            result = verify_collective(slot, str(src), op.entry, prepare_name=op.prepare,
                                       world_size=ws, device=args.device, seed=args.seed,
                                       jitter_seed=args.seed,  # anti shape-branch, like per-op
                                       model_key=model_key,
                                       graph_safe=bool(graph_safe),
                                       # rebuild plan (declared cuda_sources) must apply
                                       # in the ranks that load the kernel
                                       bundle_path=str(args.bundle))
            print(format_verify(result))
            if not result.passed:
                rc = 2
            continue

        # Load + run the miner kernel in a FRESH spawned process, so THIS trusted CLI
        # process never imports miner code (no in-process RCE sink). Production must also
        # namespace/no-egress that child; this removes the trusted-process execution.
        from optima.eval._launch import call_in_subprocess
        from optima.verify import verify_entry_from_source

        result = call_in_subprocess(
            verify_entry_from_source, op.slot, str(src), op.entry,
            prepare_name=op.prepare, dtype_name=args.dtype, device=args.device, seed=args.seed,
            jitter_seed=args.seed,  # count-dim jitter so shapes vary per run (anti shape-branch)
            model_key=model_key,  # validator per-model slot profile (activation + metric)
            override_point=op.override_point,  # compose a miner epilogue into the base kernel
            graph_safe=graph_safe,
            # Normative capability domains filter offline shapes before miner
            # invocation.  Descriptive regime prose and legacy-only metadata keep
            # their historical unfiltered verify behavior.
            eligibility_metadata=(metadata if "capabilities" in metadata else None),
            manifest_dtypes=op.dtypes,
            manifest_architectures=op.architectures,
        )
        print(format_verify(result))
        if not result.passed:
            rc = 2
    return rc


def cmd_evaluate(args: argparse.Namespace) -> int:
    from optima.slots import SLOTS
    from optima.eval.throughput_kl import EvalConfig, evaluate

    settlement_output = bool(getattr(args, "report", None) or getattr(args, "ledger", None))
    arena = None
    if getattr(args, "arena", None):
        from optima.arenas import ArenaPolicyError, get_arena
        try:
            arena = get_arena(args.arena)
        except ArenaPolicyError as exc:
            print(f"REFUSED: {exc}")
            return 2
        provided = set(getattr(args, "_provided_options", ()))
        overrides = sorted(provided & _ARENA_CONTROLLED_EVALUATE_OPTIONS)
        if overrides:
            print("REFUSED: --arena is profile-authoritative; remove score-affecting "
                  f"override(s): {', '.join(overrides)}")
            return 2
        if settlement_output and (
            "--prompt-seed" not in provided or type(args.prompt_seed) is not int
            or args.prompt_seed <= 0
        ):
            print("REFUSED: crown/report output requires an explicit positive "
                  "post-commit --prompt-seed; chain-validate derives and cross-checks it")
            return 2
        if getattr(args, "report", None) and (
            type(args.seed_round_id) is not int or args.seed_round_id < 0
            or type(args.seed_block) is not int or args.seed_block < 0
            or not args.seed_block_hash
        ):
            print("REFUSED: qualification reports require --seed-round-id, "
                  "--seed-block, and --seed-block-hash from the finalized chain receipt")
            return 2
        if getattr(args, "report", None) and (
            not isinstance(getattr(args, "miner_hotkey_address", None), str)
            or not getattr(args, "miner_hotkey_address", "")
            or len(getattr(args, "miner_hotkey_address", "")) > 256
            or any(
                char in getattr(args, "miner_hotkey_address", "")
                for char in "\x00\r\n"
            )
            or type(getattr(args, "settlement_round_id", None)) is not int
            or getattr(args, "settlement_round_id", -1) < 0
            or type(getattr(args, "evaluation_block", None)) is not int
            or getattr(args, "evaluation_block", -1) < 0
            or getattr(args, "settlement_round_id", -1)
            != getattr(args, "evaluation_block", -1)
            // arena.settlement.round_blocks
            or getattr(args, "evaluation_block", -1) < args.seed_block
        ):
            print(
                "REFUSED: qualification reports require controller-owned "
                "--miner-hotkey-address, --settlement-round-id, and "
                "--evaluation-block with arena-consistent round/block provenance"
            )
            return 2
        if getattr(args, "report", None) and (
            not isinstance(getattr(args, "chain_scope", None), str)
            or re.fullmatch(
                r"[A-Za-z0-9_.-]{1,128}:sha256:[0-9a-f]{64}",
                getattr(args, "chain_scope", "") or "",
            ) is None
            or not isinstance(getattr(args, "validator_hotkey_address", None), str)
            or not getattr(args, "validator_hotkey_address", "")
            or len(getattr(args, "validator_hotkey_address", "")) > 256
            or re.fullmatch(
                r"[0-9a-f]{64}", getattr(args, "evaluation_id", "") or ""
            ) is None
        ):
            print(
                "REFUSED: qualification reports require the controller-owned "
                "--chain-scope, --validator-hotkey-address, and 64-hex "
                "--evaluation-id for the exact persisted evaluation lease"
            )
            return 2
        if getattr(args, "ledger", None):
            print("REFUSED: registered-arena scores may only be recorded by the chain "
                  "controller after recomputing prompt entropy; use --report here")
            return 2
        try:
            arena.verify_referee_source()
            arena.verify_model_receipt(args.oci_model_dir or arena.model_path)
        except ArenaPolicyError as exc:
            print(f"REFUSED: arena runtime identity preflight failed: {exc}")
            return 2
    else:
        if settlement_output:
            print("REFUSED: --report/--ledger scoring requires an explicit registered --arena")
            return 2
        if not args.model:
            print("REFUSED: ad-hoc development evaluate requires --model (or select --arena)")
            return 2
        if not math.isfinite(args.speedup_margin) or args.speedup_margin <= 0:
            print("REFUSED: evaluate requires --speedup-margin > 0")
            return 2

    # Trusted parent: validate + scan only. It never imports miner code — the
    # kernel is loaded inside the (to-be-isolated) model process by the plugin.
    m = load_manifest(args.bundle)
    from optima.competition import CompetitionError, resolve_competition

    try:
        competition = resolve_competition(
            m, for_settlement=settlement_output
        )
    except CompetitionError as exc:
        print(f"REFUSED: bundle has no valid settlement target: {exc}")
        return 2
    if not competition.crownable:
        print(
            "  [development-only] bundle has no component settlement authority: "
            f"{competition.reason or 'unregistered competition target'}"
        )
    if (arena is not None and competition.mode != "system"
            and any(op.setup for op in m.ops)):
        print("REFUSED: registered component arenas do not admit setup()/framework "
              "mutation; submit it as the whole-serving isolated system product")
        return 2
    framework_mode = _framework_mode_for_manifest(args, m) if arena is None else False
    if not _recursive_scan_ok(args.bundle, manifest=m):  # vendored-tree guard (every .py, not just entries)
        return 2
    known = 1 if m.system is not None else 0
    if m.system is not None:
        print(f"  [ok]   bounded system patch product -> {competition.target}")
    for op in m.ops:
        if op.slot not in SLOTS:
            print(f"  [skip] {op.slot}: unknown slot")
            continue
        src = resolve_source(args.bundle, op)
        if op.execution_class == "untrusted_host":
            scan = scan_path(src)
            if not scan.ok:
                print(f"  [FAIL] {op.slot}: failed policy scan; aborting")
                for v in scan.violations:
                    print(f"      {v}")
                return 2
        known += 1
        lane = "device source" if op.execution_class == "validator_device" else "scan clean"
        print(f"  [ok]   {op.slot} <- {op.source} ({op.entry}) [{lane}]")

    if known == 0:
        print("no known slots in this bundle; nothing to evaluate")
        return 1

    # An atomic target must clear every member's fidelity policy, so the lowest
    # effective threshold wins. The CLI value remains the fallback for members
    # without a calibrated slot-specific threshold.
    if arena is not None:
        cfg_kwargs = arena.eval_config_kwargs()
        cfg_kwargs["prompt_seed"] = int(args.prompt_seed)
        if competition.mode == "system":
            cfg_kwargs["framework_mode"] = True
        cfg = EvalConfig(**cfg_kwargs)
        _kl_threshold = cfg.kl_threshold
    else:
        _kl_threshold = _strictest_kl_threshold(
            competition.members,
            advisory=args.kl_advisory,
            fallback=args.kl_threshold,
        )
        cfg = EvalConfig(
            model_path=args.model,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            num_prompts=args.num_prompts,
            framework_mode=framework_mode,
            token_match_threshold=args.token_match_threshold,
            isolate=args.isolate,
            allow_unsafe_no_isolation=args.allow_unsafe_no_isolation,
            timed_iters=args.timed_iters,
            warmup_iters=args.warmup_iters,
            conditioning_iters=args.conditioning_iters,
            speedup_margin=args.speedup_margin,
            prompt_seed=args.prompt_seed,
            input_len=args.input_len,
            top_logprobs_num=args.top_logprobs,
            ignore_eos=args.ignore_eos,
            kl_threshold=_kl_threshold,
            argmax_disagree_rate_threshold=args.argmax_disagree_rate,
            p99_kl_threshold=args.p99_kl_threshold,
            deterministic=not args.no_deterministic,
            attention_backend=args.attention_backend,
            disable_cuda_graph=args.disable_cuda_graph,
            mem_fraction_static=args.mem_fraction,
            tp_size=args.tp_size,
            max_running_requests=args.max_running_requests,
            moe_runner_backend=args.moe_runner_backend,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            candidate_attention_backend=args.candidate_attention_backend,
            candidate_moe_runner_backend=args.candidate_moe_runner_backend,
            candidate_disable_custom_all_reduce=args.candidate_disable_custom_all_reduce,
            extra_engine_kwargs=_json_obj(args.engine_kwargs_json),
            candidate_extra_engine_kwargs=_json_obj(args.candidate_engine_kwargs_json),
            fidelity_mode=args.fidelity_mode,
            audit_rate=args.audit_rate,
        )
    if _kl_threshold is not None and cfg.fidelity_mode == "kl":
        print(f"  (target={competition.target} mode={competition.mode}; strictest "
              f"member KL threshold {_kl_threshold:g})")
    if cfg.fidelity_mode == "audit":
        print(f"  (fidelity=audit: external stock-control gate + extra untimed "
              f"diagnostic audit at rate {cfg.audit_rate:g}; KL advisory)")
    if arena is not None:
        print(f"  (arena={arena.name} regime={arena.workload.regime} "
              f"fingerprint={arena.fingerprint[:16]}… prompt_seed={cfg.prompt_seed})")
    print(f"\nrunning launches of {cfg.model_path} (dtype={cfg.dtype}, "
          f"deterministic={cfg.deterministic}, cuda_graph={not cfg.disable_cuda_graph}, "
          f"attn_backend={cfg.attention_backend or 'auto'}, "
          f"framework_mode={cfg.framework_mode}, isolate_candidate={cfg.isolate}, "
          f"unsafe_no_isolation={cfg.allow_unsafe_no_isolation}): "
          f"baseline then candidate ...")
    oci_launcher = None
    if arena is not None:
        try:
            oci_launcher = _registered_oci_launcher(
                args, arena, Path(args.bundle), competition
            )
        except Exception as exc:
            print(f"REFUSED: registered-arena OCI preflight/prebuild failed: {exc}")
            return 2
        from optima.arenas import arena_environment

        with arena_environment(arena):
            report = evaluate(cfg, str(args.bundle), oci_launcher=oci_launcher)
    else:
        report = evaluate(cfg, str(args.bundle))

    qualification = None
    if arena is not None and settlement_output:
        from optima.bundle_hash import content_hash
        from optima.eval.qualification import QualificationReport

        try:
            qualification = QualificationReport.prepare_evidence(
                report,
                competition=competition,
                arena=arena,
                bundle_hash=content_hash(args.bundle),
                prompt_seed=cfg.prompt_seed,
                seed_round_id=args.seed_round_id,
                seed_block=args.seed_block,
                seed_block_hash=args.seed_block_hash,
                chain_scope=getattr(args, "chain_scope"),
                validator_hotkey=getattr(args, "validator_hotkey_address"),
                evaluation_id=getattr(args, "evaluation_id"),
                miner_hotkey=getattr(args, "miner_hotkey_address"),
                settlement_round_id=getattr(args, "settlement_round_id"),
                evaluation_block=getattr(args, "evaluation_block"),
            )
            host_reference = _publish_direct_host_attestation(
                args,
                oci_launcher,
                qualification,
            )
            qualification = qualification.bind_host_attestation(
                host_reference.sha256
            )
        except Exception as exc:
            if bool(getattr(exc, "validator_fault", False)):
                print(f"REFUSED: trusted-host evidence publication failed: {exc}")
                return 2
            raise

    b, c = report.baseline, report.candidate
    bmin, bmax, bsd = b.spread
    cmin, cmax, csd = c.spread
    print("\n=== Optima end-to-end report ===")
    print(f"bundle: {m.bundle_id}")
    print(f"baseline   {b.tok_per_s:8.1f} tok/s  (min of timed median and charged tail "
          f"{b.conditioning_tok_per_s:.1f}; {len(b.tok_per_s_samples)} timed samples; "
          f"range {bmin:.0f}-{bmax:.0f}, sd {bsd:.1f})")
    print(f"candidate  {c.tok_per_s:8.1f} tok/s  (min of timed median and charged tail "
          f"{c.conditioning_tok_per_s:.1f}; {len(c.tok_per_s_samples)} timed samples; "
          f"range {cmin:.0f}-{cmax:.0f}, sd {csd:.1f})")
    if report.baseline2 is not None:
        b2 = report.baseline2
        print(f"baseline'  {b2.tok_per_s:8.1f} tok/s  (trailing bookend; baseline noise {report.noise:.1%})")
    if not report.confident:
        verdict = "NO-DECISION (box too noisy / un-bracketed; re-queue, never crown)"
    elif report.passed_speedup:
        verdict = "PASS (noise-confident real win)"
    else:
        verdict = "below the noise-derived bar"
    print(f"speedup    {report.speedup:8.3f}x  (needs >= {report.required_speedup:.3f} = "
          f"1 + max({cfg.speedup_margin:g}, {cfg.score_k:g}*noise) -> {verdict})")
    if report.fidelity_mode == "audit":
        print(f"quality    external stock-control gate -> "
              f"{'PASS' if report.passed_quality else 'FAIL'}")
        print(f"           {report.external_quality_desc}")
        print(f"diagnostic in-engine audit (non-authoritative): {report.audit_desc}")
        print(f"           KL (ADVISORY, not gated — launch-nondeterminism confounded): "
              f"mean_kl={report.kl.mean_kl:.3e} max_kl={report.kl.max_kl:.3e} "
              f"argmax_disagree={report.kl.argmax_disagreements}/{report.kl.num_positions} "
              f"token_match={report.token_match:.4f}")
    else:
        print(f"quality    mean_kl={report.kl.mean_kl:.3e} max_kl={report.kl.max_kl:.3e} "
              f"argmax_disagree={report.kl.argmax_disagreements}/{report.kl.num_positions}  "
              f"token_match={report.token_match:.4f}{' (GATE)' if cfg.framework_mode else ''} -> "
              f"{'PASS' if report.passed_quality else 'FAIL'}")
        if getattr(report, "audit_desc", ""):
            print(f"           controller evidence: {report.audit_desc}")
    raw_crownable = bool(
        report.passed_quality and report.passed_speedup and report.confident
    )
    settlement_context = bool(
        arena is not None
        and cfg.prompt_seed > 0
        and settlement_output
        and competition.crownable
    )
    crownable = raw_crownable and settlement_context
    score_label = (
        "crownable speedup" if settlement_context
        else "DEVELOPMENT ONLY; no registered arena + post-commit seed receipt"
    )
    print(f"SCORE      {report.score:.3f}  ({score_label})")
    print(f"QUALIFY    quality={'PASS' if report.passed_quality else 'FAIL'}  "
          f"crownable={'YES' if crownable else 'NO'}")

    if getattr(args, "report", None):
        if qualification is None:  # defensive: settlement_output built it above
            raise RuntimeError("qualification report was not controller-bound")
        qualification.write(args.report)
        print(f"qualification report -> {args.report}")

    if getattr(args, "ledger", None) and getattr(args, "hotkey", None):
        from optima.bundle_hash import content_hash
        from optima.commit_reveal import Ledger
        from optima.compat import PINNED_SGLANG

        ch = content_hash(args.bundle)
        led = Ledger.load(args.ledger)
        led.record_score(args.hotkey, ch, args.round, report.score, report.kl.mean_kl,
                         crownable, sglang_version=PINNED_SGLANG,
                         slot=competition.target if competition.mode == "slot" else "",
                         target=competition.target or "", mode=competition.mode or "",
                         member_slots=competition.members, arena=arena,
                         prompt_seed=cfg.prompt_seed,
                         prompt_engine_version=arena.workload.prompt_engine_version,
                         quality_evidence=str(
                             getattr(report, "external_quality_desc", "")))
        led.save(args.ledger)
        print(f"recorded -> {args.ledger} (hotkey={args.hotkey}, round={args.round}, "
              f"target={competition.target}, mode={competition.mode}, "
              f"members={competition.members}, sglang={PINNED_SGLANG})")
    return 0 if report.passed_quality else 3


def cmd_bench(args: argparse.Namespace) -> int:
    if getattr(args, "ledger", None) is not None or getattr(args, "hotkey", None) is not None:
        print("REFUSED: bench throughput uses natural-length generations and is diagnostic "
              "noise; it cannot write a crownable ledger score. Use evaluate for scoring.")
        return 2
    from optima.slots import SLOTS
    from optima.eval.capability import evaluate_capability
    from optima.eval.throughput_kl import EvalConfig

    m = load_manifest(args.bundle)
    framework_mode = _framework_mode_for_manifest(args, m)
    if not _recursive_scan_ok(args.bundle, manifest=m):  # vendored-tree guard (every .py, not just entries)
        return 2
    known = 0
    for op in m.ops:
        if op.slot not in SLOTS:
            continue
        src = resolve_source(args.bundle, op)
        scan = scan_path(src)
        if not scan.ok:
            print(f"  [FAIL] {op.slot}: failed policy scan; aborting")
            for v in scan.violations:
                print(f"      {v}")
            return 2
        known += 1
    if known == 0:
        print("no known slots in this bundle; nothing to evaluate")
        return 1

    _kl_threshold = _strictest_kl_threshold(
        tuple(op.slot for op in m.ops),
        advisory=args.kl_advisory,
        fallback=args.kl_threshold,
    )
    if args.samples < 100 and not args.kl_advisory:
        print(f"  [note] --samples {args.samples} is small for the accuracy gate "
              "(~12% std at n=12); KL is the primary gate, use ~100-200 for a real accuracy floor.")

    cfg = EvalConfig(
        model_path=args.model,
        dtype=args.dtype,
        timed_iters=args.timed_iters,
        prompt_seed=args.prompt_seed,
        top_logprobs_num=args.top_logprobs,
        ignore_eos=args.ignore_eos,
        kl_threshold=_kl_threshold,
        argmax_disagree_rate_threshold=args.argmax_disagree_rate,
        p99_kl_threshold=args.p99_kl_threshold,
        framework_mode=framework_mode,
        token_match_threshold=args.token_match_threshold,
        isolate=args.isolate,
        allow_unsafe_no_isolation=args.allow_unsafe_no_isolation,
        deterministic=not args.no_deterministic,
        attention_backend=args.attention_backend,
        disable_cuda_graph=args.disable_cuda_graph,
        mem_fraction_static=args.mem_fraction,
        tp_size=args.tp_size,
        max_running_requests=args.max_running_requests,
        moe_runner_backend=args.moe_runner_backend,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        candidate_attention_backend=args.candidate_attention_backend,
        candidate_moe_runner_backend=args.candidate_moe_runner_backend,
        candidate_disable_custom_all_reduce=args.candidate_disable_custom_all_reduce,
        extra_engine_kwargs=_json_obj(args.engine_kwargs_json),
        candidate_extra_engine_kwargs=_json_obj(args.candidate_engine_kwargs_json),
    )
    names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    # (bench keeps the EvalConfig default warmup; its tok/s is documented noise)
    print(f"\nbenchmark eval of {args.model} on {names} "
          f"({args.samples}/bench; framework_mode={cfg.framework_mode}, "
          f"isolate_candidate={cfg.isolate}, "
          f"unsafe_no_isolation={cfg.allow_unsafe_no_isolation}): baseline then candidate ...")
    report = evaluate_capability(
        cfg, str(args.bundle), names,
        samples_per_benchmark=args.samples, acc_tolerance=args.acc_tolerance,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== Optima capability report ===")
    print(f"bundle: {m.bundle_id}")
    for bs in report.benchmarks:
        flag = "" if bs.delta >= -args.acc_tolerance else "  <-- REGRESSION"
        print(f"  {bs.name:10s} baseline {bs.baseline_acc:6.1%} ({bs.baseline_correct}/{bs.n})  "
              f"candidate {bs.candidate_acc:6.1%} ({bs.candidate_correct}/{bs.n})  "
              f"Δ{bs.delta:+.1%}{flag}")
    b2 = f"  baseline' {report.baseline2_tok_s:8.1f}" if report.baseline2_tok_s > 0 else ""
    print(f"throughput baseline {report.baseline_tok_s:8.1f} tok/s  candidate {report.candidate_tok_s:8.1f} tok/s{b2}")
    if not report.confident:
        sp_verdict = "NO-DECISION (box too noisy; re-queue)"
    elif report.passed_speedup:
        sp_verdict = "PASS (noise-confident)"
    else:
        sp_verdict = "below the noise-derived bar"
    print(f"speedup    {report.speedup:8.3f}x  (needs >= {report.required_speedup:.3f}, "
          f"baseline noise {report.noise:.1%}) -> {sp_verdict}")
    kl = report.kl
    if args.kl_advisory:
        kl_note = "advisory (not gated)"
    elif kl.num_positions == 0:
        kl_note = "n/a (no logprobs)"
    else:
        kl_note = f"<= {args.kl_threshold:.1e}"
    rate_note = "" if args.kl_advisory else f" (<= {args.argmax_disagree_rate:.1%})"
    print(f"quality    no-accuracy-regression + KL mean_kl={kl.mean_kl:.3e} ({kl_note}), "
          f"argmax_disagree={kl.argmax_disagreements}/{kl.num_positions} "
          f"({kl.argmax_disagree_rate:.2%}{rate_note}), "
          f"token_match={report.token_match:.4f}{' (GATE)' if cfg.framework_mode else ''} -> "
          f"{'PASS' if report.passed_quality else 'FAIL'}")
    print("CROWNABLE no (bench throughput is diagnostic; use evaluate for scoring)")
    return 0 if report.passed_quality else 3


def cmd_hash(args: argparse.Namespace) -> int:
    from optima.bundle_hash import content_hash

    print(content_hash(args.bundle))
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    from optima.bundle_hash import content_hash
    from optima.commit_reveal import Ledger, make_commitment

    ch = content_hash(args.bundle)
    com = make_commitment(ch, args.hotkey, args.salt)
    led = Ledger.load(args.ledger)
    seq = led.commit(args.hotkey, com, args.round)
    led.save(args.ledger)
    print(f"committed hotkey={args.hotkey} round={args.round} seq={seq}")
    print(f"commitment={com}")
    print("keep your --salt and bundle; you'll need both to reveal")
    return 0


def cmd_reveal(args: argparse.Namespace) -> int:
    from optima.bundle_hash import content_hash
    from optima.commit_reveal import Ledger, RevealError
    from optima.copy_fingerprint import (
        bundle_fingerprint,
        bundle_slot_file_fingerprints,
        bundle_slot_fingerprints,
        bundle_structural_fingerprint,
    )

    ch = content_hash(args.bundle)
    fp = bundle_fingerprint(args.bundle)  # reformat-invariant near-copy signal (auto-demotes)
    slot_fps = bundle_slot_fingerprints(args.bundle)  # per-slot: a padded bundle can't hide a stolen slot
    file_fps = bundle_slot_file_fingerprints(args.bundle)  # per-file: nor a RELOCATED stolen body
    sfp = bundle_structural_fingerprint(args.bundle)  # rename/constant-tweak skeleton (advisory)
    led = Ledger.load(args.ledger)
    # Query advisory structural matches BEFORE recording this reveal (so we don't match self).
    advisory = led.structural_near_copies(sfp, args.hotkey)
    try:
        rev = led.reveal(args.hotkey, ch, args.salt, args.round, fingerprint=fp,
                         structural_fingerprint=sfp, slot_fingerprints=slot_fps,
                         slot_file_fingerprints=file_fps)
    except RevealError as e:
        print(f"REJECTED: {e}")
        return 2
    led.save(args.ledger)
    print(f"revealed hotkey={args.hotkey} content={ch[:16]}... original={rev.original}")
    if not rev.original:
        print("  -> flagged as a COPY (an earlier commit to this exact content, its "
              "reformatted-but-identical structure, or a bundle whose kernel source "
              "this one contains exists); earns 0")
    elif advisory:
        print(f"  ⚠ ADVISORY: structurally similar to earlier submission(s) by {', '.join(advisory)} "
              "(possible rename/constant-tweak copy) — flagged for review, not auto-demoted")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    from optima.commit_reveal import Ledger

    led = Ledger.load(args.ledger)
    print(f"commitments={len(led.commitments)} reveals={len(led.reveals)} scores={len(led.scores)}")
    if getattr(args, "arena", None):
        from optima.arenas import ArenaPolicyError, get_arena

        try:
            arena = get_arena(args.arena)
        except ArenaPolicyError as exc:
            print(f"REFUSED: {exc}")
            return 2
        champions = led.arena_champions.get(arena.bracket, {})
        print(f"arena: {arena.bracket}")
        if not champions:
            print("champions: (none yet)")
        for target, champion in sorted(champions.items()):
            print(f"  {target}: hotkey={champion.hotkey} score={champion.score:.3f} "
                  f"round={champion.round_id} content={champion.content_hash[:16]}...")
        return 0
    if led.champion:
        c = led.champion
        print(f"champion: hotkey={c.hotkey} score={c.score:.3f} round={c.round_id} "
              f"content={c.content_hash[:16]}...")
    else:
        print("champion: (none yet)")
    return 0


def cmd_retries(args: argparse.Namespace) -> int:
    """Inspect or explicitly release non-terminal retry state.

    Both immutable scopes are mandatory. Merely pointing at a ledger path is not
    authority to mutate a similarly named file from another chain or arena.
    """
    from optima.arenas import ArenaPolicyError, get_arena
    from optima.chain.validator_loop import LedgerLockError, _exclusive_ledger_pass
    from optima.commit_reveal import (
        Ledger,
        RETRY_STATE_AUTOMATIC,
        RETRY_STATE_IN_PROGRESS,
    )

    try:
        arena = get_arena(args.arena)
    except ArenaPolicyError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if args.release is not None and args.release_validator_fault is not None:
        print("REFUSED: release one miner retry or one validator fault at a time")
        return 2

    try:
        with _exclusive_ledger_pass(args.ledger):
            led = Ledger.load(args.ledger)
            if not led.chain_scope or led.chain_scope != args.chain_scope:
                print(
                    "REFUSED: --chain-scope does not match the persisted ledger scope "
                    f"({led.chain_scope or 'unbound'})"
                )
                return 2

            if args.release is not None:
                hotkey, bundle_hash = args.release
                try:
                    released = led.release_held_retry(
                        hotkey,
                        bundle_hash,
                        arena_bracket=arena.bracket,
                        chain_scope=args.chain_scope,
                    )
                except (KeyError, ValueError) as exc:
                    print(f"REFUSED: {exc}")
                    return 2
                led.save(args.ledger)
                print(
                    f"released held miner retry: hotkey={released.hotkey} "
                    f"bundle={released.bundle_hash} arena={arena.bracket}"
                )

            if args.release_validator_fault is not None:
                hotkey, bundle_hash = args.release_validator_fault
                try:
                    released_fault = led.release_validator_fault(
                        hotkey,
                        bundle_hash,
                        arena_bracket=arena.bracket,
                        chain_scope=args.chain_scope,
                    )
                except (KeyError, ValueError) as exc:
                    print(f"REFUSED: {exc}")
                    return 2
                led.save(args.ledger)
                print(
                    f"released validator-fault hold: hotkey={released_fault.hotkey} "
                    f"bundle={released_fault.bundle_hash} arena={arena.bracket}"
                )

            try:
                retries = led.retries_for_scope(
                    arena_bracket=arena.bracket,
                    chain_scope=args.chain_scope,
                )
                validator_faults = led.validator_faults_for_scope(
                    arena_bracket=arena.bracket,
                    chain_scope=args.chain_scope,
                )
            except ValueError as exc:
                print(f"REFUSED: {exc}")
                return 2
            print(f"retry scope: chain={args.chain_scope} arena={arena.bracket}")
            if not retries:
                print("miner retries: (none)")
            for retry in retries:
                if retry.state == RETRY_STATE_AUTOMATIC:
                    next_block = str(retry.next_block)
                elif retry.state == RETRY_STATE_IN_PROGRESS:
                    next_block = f"leased@{retry.lease_block}"
                else:
                    next_block = "operator-release"
                print(
                    f"  miner-retry {retry.state:9s} {retry.kind:14s} "
                    f"attempts={retry.attempts} next={next_block} "
                    f"hotkey={retry.hotkey} bundle={retry.bundle_hash} "
                    f"reason={retry.last_reason}"
                )
            if not validator_faults:
                print("validator-fault holds: (none)")
            for hold in validator_faults:
                print(
                    f"  validator-fault held@{hold.created_block} "
                    f"evaluation={hold.evaluation_id} hotkey={hold.hotkey} "
                    f"bundle={hold.bundle_hash} reason={hold.reason}"
                )
            return 0
    except LedgerLockError as exc:
        print(f"REFUSED: {exc}")
        return 2


def cmd_settle(args: argparse.Namespace) -> int:
    from optima.arenas import ArenaPolicyError, get_arena
    from optima.chain.validator_loop import (
        LedgerLockError,
        _exclusive_ledger_pass,
        _recover_pending_settlements,
    )
    from optima.commit_reveal import Ledger

    try:
        arena = get_arena(args.arena)
    except ArenaPolicyError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if "--margin" in set(getattr(args, "_provided_options", ())):
        print("REFUSED: --margin is owned by the immutable arena settlement policy; "
              f"this arena uses {arena.settlement.dethrone_margin:g}")
        return 2
    if "--round" in set(getattr(args, "_provided_options", ())):
        print(
            "REFUSED: settlement order is owned by durable pending dispositions; "
            "an operator cannot select or replay a round"
        )
        return 2
    try:
        with _exclusive_ledger_pass(args.ledger):
            led = Ledger.load(args.ledger)
            try:
                led.bind_validator_hotkey(args.validator_hotkey_address)
            except Exception as exc:
                print(f"REFUSED: validator authority mismatch: {exc}")
                return 2
            verifier = _retained_host_attestation_verifier(
                args.oci_artifact_root
            )
            before = dict(led.arena_champions.get(arena.bracket, {}))
            _recover_pending_settlements(
                led,
                ledger_path=args.ledger,
                arena=arena,
                margin=arena.settlement.dethrone_margin,
                host_attestation_verifier=verifier,
                validator_hotkey=args.validator_hotkey_address,
            )
            champions = led.arena_champions.get(arena.bracket, {})
            weights = led.current_weights(
                arena=arena,
                host_attestation_verifier=verifier,
                validator_hotkey=args.validator_hotkey_address,
            )
            print(f"arena {arena.bracket}\nauthoritative per-target championships "
                  "after causal pending recovery:")
            for target, champ in sorted(champions.items()):
                changed = " (NEW)" if before.get(target) != champ else ""
                print(f"  {target or '(unlabeled)':40s} {champ.hotkey} "
                      f"score={champ.score:.3f}{changed}")
            print(f"weights: {weights}")
            return 0
    except LedgerLockError as exc:
        print(f"REFUSED: {exc}")
        return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optima",
        description=(
            "Optima validator harness.\n"
            "\n"
            "Commands by workflow:\n"
            "  develop a kernel (miner) ... slots, scan, verify, evaluate, bench\n"
            "  submit on-chain (miner) .... hash, chain-register, chain-package,\n"
            "                               chain-submit, chain-status\n"
            "  score + settle (validator) . chain-validate, settle, ledger, set-weights,\n"
            "                               commit, reveal (local-ledger simulation)\n"
            "  environment checks ......... compat, chain-compat\n"
            "\n"
            "New to Optima? Start with docs/MINER_GUIDE.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("slots", help="list the op-slot ABI")
    sp.set_defaults(func=cmd_slots)

    sp = sub.add_parser("compat", help="check our sglang integration points survived an upgrade")
    sp.set_defaults(func=cmd_compat)

    sp = sub.add_parser("chain-compat",
                        help="check the installed bittensor SDK exposes the chain API we use")
    sp.set_defaults(func=cmd_chain_compat)

    sp = sub.add_parser("set-weights",
                        help="inspect the ledger's global registered-arena weight vector")
    sp.add_argument("--ledger", default="optima_ledger.json")
    sp.add_argument("--arena", required=True,
                    help="registered arena selecting chain-scope policy; emission always "
                         "projects every registered arena, never this arena alone")
    sp.add_argument("--oci-artifact-root", default="~/.cache/optima/oci_artifacts",
                    help="private retained host-attestation store")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", default="finney",
                    help="named network or an explicit wss:// endpoint URL")
    sp.add_argument("--wallet", default="default")
    sp.add_argument("--hotkey", default="default")
    sp.add_argument("--per-target", "--per-slot", dest="per_target", action="store_true",
                    help="weights from per-target championships; --per-slot is a "
                         "compatibility alias for singleton-era ledgers")
    sp.add_argument("--dry-run", action="store_true",
                    help="build + print only; required unless releasing a held journal")
    sp.add_argument(
        "--release-publication-hold",
        action="store_true",
        help="after trusted audit, archive and release one held global publication intent",
    )
    sp.add_argument(
        "--release-reason",
        default=None,
        help="required bounded audit reason for --release-publication-hold",
    )
    sp.set_defaults(func=cmd_set_weights)

    # ---- chain: miner submission + the validator loop ----
    sp = sub.add_parser("chain-package",
                        help="tar.gz a bundle for hosting; prints the content hash to commit")
    sp.add_argument("bundle")
    sp.add_argument("--out", default=None, help="archive path (default <bundle>.tar.gz)")
    sp.set_defaults(func=cmd_chain_package)

    sp = sub.add_parser("chain-submit",
                        help="miner: commit a bundle (hash + fetch URL) on-chain via "
                             "timelock commit-reveal")
    sp.add_argument("bundle")
    sp.add_argument("--url", required=True, help="where the validator fetches the tar.gz")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", required=True,
                    help="named network or an explicit wss:// endpoint URL")
    sp.add_argument("--wallet", default="default")
    sp.add_argument("--hotkey", default="default", help="the MINER hotkey name")
    sp.add_argument("--blocks-until-reveal", type=int, default=10,
                    help="timelock length; the payload is unreadable until then")
    sp.add_argument("--dry-run", action="store_true",
                    help="build + print the payload, do NOT sign or submit")
    sp.set_defaults(func=cmd_chain_submit)

    sp = sub.add_parser("chain-status",
                        help="subnet snapshot: block, neurons, permits, revealed submissions")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", required=True)
    sp.add_argument("--wallet", default=None, help="also report this wallet's uid/permit")
    sp.add_argument("--hotkey", default="default")
    sp.set_defaults(func=cmd_chain_status)

    sp = sub.add_parser("chain-validate",
                        help="the validator loop: commitments -> fetch -> evaluate -> "
                             "settle -> weights")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", required=True)
    sp.add_argument("--wallet", default="default")
    sp.add_argument("--hotkey", default="default", help="the VALIDATOR hotkey name")
    sp.add_argument("--ledger", default="chain_ledger.json")
    sp.add_argument("--arena", required=True,
                    help="registered immutable arena evaluated and settled by this loop")
    sp.add_argument("--bundles-dir", default="chain_bundles",
                    help="where fetched submissions are cached (keyed by content hash)")
    sp.add_argument("--eval-cmd", default=None,
                    help="DEVELOPMENT ONLY, non-crownable: command template with "
                         "{bundle}, {report}, {arena}, and "
                         "{prompt_seed} placeholders "
                         "(missing/incomplete reports fail closed)")
    sp.add_argument("--verify-only", action="store_true",
                    help="CPU/GPU component plumbing only; explicitly non-crownable")
    sp.add_argument("--eval-device", default="cpu",
                    help="verify-mode device (default cpu)")
    sp.add_argument("--oci-gpus", default=None,
                    help="production evaluator GPU IDs, e.g. 0,1,2,3")
    sp.add_argument("--oci-source-dir", default=None,
                    help="referee checkout or prebuilt release (default: this checkout); "
                         "a checkout is sanitized before mounting")
    sp.add_argument("--oci-release-root", default="~/.cache/optima/referee_releases",
                    help="private content-addressed sanitized referee release cache")
    sp.add_argument("--oci-model-dir", default=None,
                    help="host model volume (default: arena model_path)")
    sp.add_argument("--oci-artifact-root", default="~/.cache/optima/oci_artifacts",
                    help="trusted candidate-private prebuilt artifact root")
    sp.add_argument("--oci-scratch-root", default="~/.cache/optima/oci_scratch",
                    help="launch-private OCI scratch parent")
    sp.add_argument("--eval-timeout", type=float, default=3600.0)
    sp.add_argument("--margin", type=float, default=0.02,
                    help="deprecated/refused for registered arenas: settlement margin "
                         "is part of the immutable arena policy")
    sp.add_argument("--interval", type=float, default=60.0, help="seconds between passes")
    sp.add_argument("--once", action="store_true", help="single pass, then exit")
    sp.add_argument("--dry-run-weights", action="store_true",
                    help="run the full loop but never submit weights")
    sp.set_defaults(func=cmd_chain_validate)

    sp = sub.add_parser("chain-register",
                        help="register this hotkey on a subnet (burned_register; needs "
                             "the coldkey password) + preflight")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", required=True)
    sp.add_argument("--wallet", default="default")
    sp.add_argument("--hotkey", default="default")
    sp.set_defaults(func=cmd_chain_register)

    sp = sub.add_parser("scan", help="static policy scan of a bundle")
    sp.add_argument("bundle")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser(
        "verify", help="op-level correctness vs reference",
        epilog=("examples:\n"
                "  # CPU dry-run (no GPU needed; the miner-guide inner loop)\n"
                "  optima verify examples/miner_silu_torch --device cpu --dtype float32\n"
                "  # real shapes/dtypes on a GPU box\n"
                "  optima verify my_bundle --device cuda --dtype bfloat16\n"
                "  # a collective slot at the arena's TP size\n"
                "  optima verify my_bundle --device cuda --world-size 4"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp.add_argument("bundle")
    sp.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    sp.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--world-size", type=int, default=None, dest="world_size",
                    help="ranks for DISTRIBUTED verify of collective slots (default 2; "
                         "use the arena TP size, e.g. 4, on a multi-GPU box)")
    sp.add_argument("--model", default=None,
                    help="validator model key for the per-model slot profile (activation + "
                         "low-bit metric), e.g. MiniMax-M3. Default: the model declared in the "
                         "op's metadata (dev convenience); production uses the served-model key.")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser(
        "evaluate", help="end-to-end throughput + KL on a model",
        epilog=("examples (always launch via `python -m optima.cli` on GPU —\n"
                "sglang spawns the scheduler with mp spawn):\n"
                "  # quick smoke on a small model\n"
                "  python -m optima.cli evaluate my_bundle --model Qwen/Qwen2.5-1.5B-Instruct \\\n"
                "      --num-prompts 64 --max-new-tokens 64\n"
                "  # nondeterministic arena: fidelity via the in-engine audit, KL advisory\n"
                "  python -m optima.cli evaluate my_bundle --model <model> \\\n"
                "      --fidelity-mode audit --no-deterministic"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp.add_argument("bundle")
    sp.add_argument("--arena", default=None,
                    help="registered immutable scoring arena; without this evaluate is "
                         "development-only and cannot emit --report/--ledger scores")
    sp.add_argument("--model", required=False, help="model path for ad-hoc sglang.Engine")
    sp.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    sp.add_argument("--max-new-tokens", type=int, default=64)
    sp.add_argument("--num-prompts", type=int, default=32)
    sp.add_argument("--timed-iters", type=int, default=3, help="median-of-K timed passes per launch")
    sp.add_argument("--warmup-iters", type=int, default=2,
                    help="heat-soak rounds before timed work; all are quality-graded. The "
                         "initial warmup_iters-conditioning_iters rounds may absorb legitimate "
                         "request-lazy setup; the final conditioning rounds and every gap are "
                         "host-charged continuously through the first timed response, and the "
                         "slowest charged batch/tail caps the arm's score. On boxes where "
                         "clock-locking is unavailable (tenant pods), thermal ramp lands in the "
                         "B/B' bookends as baseline 'noise' and inflates the crowning bar — raise "
                         "this until the bookends agree instead")
    sp.add_argument("--conditioning-iters", type=int, default=2,
                    help="final warmup rounds retained in the continuous charged tail; "
                         "registered arenas pin this policy and require at least two")
    sp.add_argument("--speedup-margin", type=float, default=0.005,
                    help="FLOOR on the required improvement; the actual bar is "
                         "1 + max(margin, 2*measured_noise). Keep low — real wins stack at 1-2%%; "
                         "the noise term, not this floor, guards an unstable box")
    sp.add_argument("--prompt-seed", type=int, default=0, help="per-epoch prompt sampling seed")
    sp.add_argument("--seed-round-id", type=int, default=None,
                    help="chain-derived prompt receipt: reveal settlement round")
    sp.add_argument("--seed-block", type=int, default=None,
                    help="chain-derived prompt receipt: finalized reveal block")
    sp.add_argument("--seed-block-hash", default=None,
                    help="chain-derived prompt receipt: canonical finalized block hash")
    sp.add_argument(
        "--chain-scope",
        default=None,
        help="settlement report only: exact genesis/netuid chain namespace",
    )
    sp.add_argument(
        "--validator-hotkey-address",
        default=None,
        help="settlement report only: exact signing validator SS58 identity",
    )
    sp.add_argument(
        "--evaluation-id",
        default=None,
        help="settlement report only: 64-hex persisted evaluation lease ID",
    )
    sp.add_argument(
        "--miner-hotkey-address",
        default=None,
        help="settlement report only: exact submission owner SS58 identity",
    )
    sp.add_argument(
        "--settlement-round-id",
        type=int,
        default=None,
        help="settlement report only: round containing the evaluation block",
    )
    sp.add_argument(
        "--evaluation-block",
        type=int,
        default=None,
        help="settlement report only: finalized block at which evaluation ran",
    )
    sp.add_argument("--oci-gpus", default=None,
                    help="registered-arena GPU IDs, e.g. 0,1,2,3 (required)")
    sp.add_argument("--oci-source-dir", default=None,
                    help="referee checkout or prebuilt release (default: this checkout); "
                         "a checkout is sanitized before mounting")
    sp.add_argument("--oci-release-root", default="~/.cache/optima/referee_releases",
                    help="private content-addressed sanitized referee release cache")
    sp.add_argument("--oci-model-dir", default=None,
                    help="host model volume (default: arena model_path)")
    sp.add_argument("--oci-artifact-root", default="~/.cache/optima/oci_artifacts")
    sp.add_argument("--oci-scratch-root", default="~/.cache/optima/oci_scratch")
    sp.add_argument("--input-len", type=int, default=None,
                    help="approximate tokens per prompt (default: the 10-20-token short corpus). "
                         "Set for prefill-heavy arenas: the short corpus is a pure-decode regime, "
                         "so a prefill-side kernel win is invisible to the scorer without this. "
                         "Prompts stay seed-deterministic, prefix-disjoint (no radix-cache "
                         "inflation) and duplicate-block-free (optima/eval/prompts.py). "
                         "PAIR WITH --engine-kwargs-json '{\"disable_radix_cache\": true}': the "
                         "timed iterations replay the SAME prompts, so with the prefix cache on, "
                         "iteration 2+ serves the whole input from cache and the run silently "
                         "degrades to pure decode again (measured 2026-07-10: median-of-3 read "
                         "267 tok/s cached vs 63 tok/s actually prefilling)")
    sp.add_argument("--top-logprobs", type=int, default=20)
    sp.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True,
                    help="force generation to the max token budget so baseline and candidate emit IDENTICAL "
                         "token counts (pure latency comparison, no EOS-timing gaming). ON for scoring; "
                         "--no-ignore-eos only for a natural-length probe")
    sp.add_argument("--kl-threshold", type=float, default=5e-3)
    sp.add_argument("--argmax-disagree-rate", type=float, default=0.01,
                    help="max fraction of positions whose top token may flip (sparse-cheat guard)")
    sp.add_argument("--p99-kl-threshold", type=float, default=None, help="optional p99 KL gate (catastrophic tail)")
    sp.add_argument("--kl-advisory", action="store_true", help="report KL but don't gate on it")
    sp.add_argument("--fidelity-mode", choices=("kl", "audit"), default="kl",
                    help="quality gate: 'kl' = rollout-KL vs the baseline launch (valid only on a "
                         "deterministic-capable arena); 'audit' = in-engine per-call comparison vs "
                         "the stock baseline under the slot's verify tolerances (extra untimed "
                         "quality launch; KL becomes advisory). Use 'audit' where two identical "
                         "launches aren't logit-identical (measured 2026-07-07: bit-stock "
                         "candidates scored mean_kl 0.8-0.96 on eager fa4/NVFP4).")
    sp.add_argument("--audit-rate", type=float, default=0.05,
                    help="fidelity-mode=audit: fraction of eligible dispatcher calls audited")
    sp.add_argument("--mem-fraction", type=float, default=0.6,
                    help="sglang mem_fraction_static (use ~0.9 for big models like gpt-oss-120b)")
    sp.add_argument("--no-deterministic", action="store_true")
    sp.add_argument("--attention-backend", default=None,
                    help="sglang attention backend (default: auto-pick best per-HW, e.g. fa3/flashinfer)")
    sp.add_argument("--candidate-attention-backend", default=None,
                    help="candidate-only attention backend override")
    sp.add_argument("--disable-cuda-graph", action="store_true",
                    help="eager mode for quick debugging; DEGRADES the baseline — never score with this")
    sp.add_argument("--tp-size", type=int, default=None, help="tensor-parallel size (multi-GPU)")
    sp.add_argument("--max-running-requests", type=int, default=None,
                    help="cap concurrent running requests = score at a serving-realistic batch (report M2)")
    sp.add_argument("--moe-runner-backend", default=None,
                    help="sglang MoE backend (e.g. 'triton')")
    sp.add_argument("--candidate-moe-runner-backend", default=None,
                    help="candidate-only MoE backend override (framework-mode backend swaps)")
    sp.add_argument("--disable-custom-all-reduce", action="store_true",
                    help="needed for TP>2 over PCIe (no NVLink)")
    sp.add_argument("--candidate-disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=None,
                    help="candidate-only custom-all-reduce override")
    sp.add_argument("--engine-kwargs-json", default=None,
                    help="JSON object merged into both SGLang Engine kwargs")
    sp.add_argument("--candidate-engine-kwargs-json", default=None,
                    help="JSON object merged into candidate SGLang Engine kwargs")
    # optional: record the result into a commit-reveal ledger
    sp.add_argument("--ledger", default=None, help="ledger json to record the score into")
    sp.add_argument("--hotkey", default=None, help="miner hotkey (with --ledger)")
    sp.add_argument("--round", type=int, default=0, help="round id (with --ledger)")
    sp.add_argument("--report", default=None,
                    help="atomically write the typed qualification JSON consumed by "
                         "chain-validate --eval-cmd; requires chain/validator/lease "
                         "identity supplied by that controller")
    sp.add_argument("--framework-mode", action="store_true",
                    help="miner may patch the engine (setup()); gate on token-match vs the stock baseline, not in-process KL")
    sp.add_argument("--token-match-threshold", type=float, default=0.99,
                    help="framework-mode minimum token match fraction")
    sp.add_argument("--isolate", action=argparse.BooleanOptionalAction, default=True,
                    help="run every candidate in a no-egress network namespace (default: on; "
                         "--no-isolate is accepted only with the explicit unsafe dev override)")
    sp.add_argument("--allow-unsafe-no-isolation", action="store_true",
                    help="DEV ONLY: continue if candidate no-egress isolation is unavailable")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser(
        "bench",
        help="real-task capability gate; throughput is diagnostic and never crownable",
        epilog=("examples:\n"
                "  # capability floor on a real task (start small, then raise --samples)\n"
                "  python -m optima.cli bench my_bundle --model Qwen/Qwen2.5-1.5B-Instruct \\\n"
                "      --benchmarks gsm8k --samples 128"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp.add_argument("bundle")
    sp.add_argument("--model", required=True)
    sp.add_argument("--benchmarks", default="gsm8k",
                    help="comma-separated: gsm8k, mmlu, long_math")
    sp.add_argument("--samples", type=int, default=32, help="problems per benchmark")
    sp.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    sp.add_argument("--max-new-tokens", type=int, default=None,
                    help="override the benchmark decode budget")
    sp.add_argument("--timed-iters", type=int, default=2)
    sp.add_argument("--prompt-seed", type=int, default=0)
    sp.add_argument("--acc-tolerance", type=float, default=0.02)
    sp.add_argument("--kl-threshold", type=float, default=5e-3, help="dense KL gate on the benchmark prompts")
    sp.add_argument("--argmax-disagree-rate", type=float, default=0.01,
                    help="max fraction of positions whose top token may flip (sparse-cheat guard)")
    sp.add_argument("--p99-kl-threshold", type=float, default=None, help="optional p99 KL gate (catastrophic tail)")
    sp.add_argument("--kl-advisory", action="store_true",
                    help="report KL but don't gate on it (big MoE: noise-dominated; rely on accuracy)")
    sp.add_argument("--top-logprobs", type=int, default=20, help="top-k logprobs for the KL gate (0 disables)")
    sp.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True,
                    help="force generation to the max token budget so baseline and candidate emit IDENTICAL "
                         "token counts (pure latency comparison, no EOS-timing gaming). ON for scoring; "
                         "--no-ignore-eos only for a natural-length probe")
    sp.add_argument("--mem-fraction", type=float, default=0.6,
                    help="sglang mem_fraction_static (use ~0.9 for big models like gpt-oss-120b)")
    sp.add_argument("--no-deterministic", action="store_true")
    sp.add_argument("--attention-backend", default=None,
                    help="sglang attention backend (default: auto-pick best per-HW, e.g. fa3/flashinfer)")
    sp.add_argument("--candidate-attention-backend", default=None,
                    help="candidate-only attention backend override")
    sp.add_argument("--disable-cuda-graph", action="store_true",
                    help="eager mode for quick debugging; DEGRADES the baseline — never score with this")
    sp.add_argument("--tp-size", type=int, default=None, help="tensor-parallel size (multi-GPU)")
    sp.add_argument("--max-running-requests", type=int, default=None,
                    help="cap concurrent running requests = score at a serving-realistic batch (report M2)")
    sp.add_argument("--moe-runner-backend", default=None,
                    help="sglang MoE backend (e.g. 'triton')")
    sp.add_argument("--candidate-moe-runner-backend", default=None,
                    help="candidate-only MoE backend override (framework-mode backend swaps)")
    sp.add_argument("--disable-custom-all-reduce", action="store_true",
                    help="needed for TP>2 over PCIe (no NVLink)")
    sp.add_argument("--candidate-disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=None,
                    help="candidate-only custom-all-reduce override")
    sp.add_argument("--engine-kwargs-json", default=None,
                    help="JSON object merged into both SGLang Engine kwargs")
    sp.add_argument("--candidate-engine-kwargs-json", default=None,
                    help="JSON object merged into candidate SGLang Engine kwargs")
    sp.add_argument("--framework-mode", action="store_true",
                    help="miner may patch/swap the engine; gate on token-match vs the stock baseline, not in-process KL")
    sp.add_argument("--token-match-threshold", type=float, default=0.99,
                    help="framework-mode minimum token match fraction")
    sp.add_argument("--isolate", action=argparse.BooleanOptionalAction, default=True,
                    help="run every candidate in a no-egress network namespace (default: on; "
                         "--no-isolate is accepted only with the explicit unsafe dev override)")
    sp.add_argument("--allow-unsafe-no-isolation", action="store_true",
                    help="DEV ONLY: continue if candidate no-egress isolation is unavailable")
    sp.add_argument("--ledger", default=None,
                    help="deprecated and refused: bench cannot record crownable scores")
    sp.add_argument("--hotkey", default=None,
                    help="deprecated and refused with bench ledger scoring")
    sp.add_argument("--round", type=int, default=0)
    sp.set_defaults(func=cmd_bench)

    # ---- commit-reveal / scoring ledger ----
    sp = sub.add_parser("hash", help="print a bundle's deterministic content hash")
    sp.add_argument("bundle")
    sp.set_defaults(func=cmd_hash)

    sp = sub.add_parser("commit", help="post a commitment for a bundle (commit phase)")
    sp.add_argument("bundle")
    sp.add_argument("--hotkey", required=True)
    sp.add_argument("--salt", required=True)
    sp.add_argument("--round", type=int, default=0)
    sp.add_argument("--ledger", default="optima_ledger.json")
    sp.set_defaults(func=cmd_commit)

    sp = sub.add_parser("reveal", help="reveal a previously committed bundle (reveal phase)")
    sp.add_argument("bundle")
    sp.add_argument("--hotkey", required=True)
    sp.add_argument("--salt", required=True)
    sp.add_argument("--round", type=int, default=0)
    sp.add_argument("--ledger", default="optima_ledger.json")
    sp.set_defaults(func=cmd_reveal)

    sp = sub.add_parser("ledger", help="show ledger state (champion, counts)")
    sp.add_argument("--ledger", default="optima_ledger.json")
    sp.add_argument("--arena", default=None,
                    help="show champions in this registered arena (legacy namespace if omitted)")
    sp.set_defaults(func=cmd_ledger)

    sp = sub.add_parser(
        "retries",
        help="inspect or release chain+arena-scoped held evaluation retries",
        description=(
            "Inspect or release non-terminal evaluation retries. Infrastructure "
            "budgets count the initial failed evaluation as attempt one; reaching "
            "the arena cap enters operator hold and prevents further automatic GPU work."
        ),
    )
    sp.add_argument("--ledger", required=True)
    sp.add_argument("--arena", required=True)
    sp.add_argument(
        "--chain-scope",
        required=True,
        help="exact genesis/netuid scope persisted in the validator ledger",
    )
    sp.add_argument(
        "--release",
        nargs=2,
        metavar=("HOTKEY", "BUNDLE_HASH"),
        help="reset one operator-held miner retry; the next pass may evaluate it",
    )
    sp.add_argument(
        "--release-validator-fault",
        nargs=2,
        metavar=("HOTKEY", "BUNDLE_HASH"),
        help="release one distinct controller-fault hold after trusted repair/audit",
    )
    sp.set_defaults(func=cmd_retries)

    sp = sub.add_parser(
        "settle",
        help="recover durable pending settlements in canonical reveal order",
    )
    sp.add_argument(
        "--round", type=int, default=0,
        help="deprecated/refused: pending dispositions own causal round order",
    )
    sp.add_argument("--margin", type=float, default=0.02,
                    help="deprecated/refused: use the registered arena policy")
    sp.add_argument("--ledger", default="optima_ledger.json")
    sp.add_argument("--arena", required=True,
                    help="registered immutable arena championship to settle")
    sp.add_argument("--oci-artifact-root", default="~/.cache/optima/oci_artifacts",
                    help="private retained host-attestation store")
    sp.add_argument(
        "--validator-hotkey-address",
        required=True,
        help="independently-known active validator identity bound to this ledger",
    )
    sp.add_argument("--per-target", "--per-slot", dest="per_target", action="store_true",
                    help="one champion per competition target; atomic multi-slot bundles "
                         "remain one target (--per-slot is a compatibility alias)")
    sp.set_defaults(func=cmd_settle)

    return p


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    # Preserve which optional knobs the operator actually supplied. Registered
    # arenas reject score-affecting overrides even when the value happens to equal
    # today's default; otherwise an argparse default change could silently mutate a
    # supposedly immutable bracket.
    args._provided_options = {
        token.split("=", 1)[0]
        for token in raw_argv
        if token.startswith("--")
    }
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
