"""Miner development checks using the serving loader, node audit and receipts.

Run in the published arena image on a disposable host. These diagnostics carry
no qualification or reward authority. Each engine starts in a fresh process;
candidate imports stay out of the command's controller.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def _manifest(bundle: str):
    from cacheon.manifest import load_manifest
    from cacheon.target_catalog import default_target_catalog

    manifest = load_manifest(bundle)
    catalog = default_target_catalog()
    resolved = catalog.resolve_manifest(manifest).require_registered()
    if not catalog.require(resolved.target_id).node_roots:
        raise ValueError("check requires a node-address bundle")
    return manifest


def _smoke(bundle: str) -> list[dict]:
    """Load the actual entries without inventing a model for their forward calls."""
    from cacheon import seam
    from cacheon.registry import REGISTRY

    manifest = _manifest(bundle)
    seam._load_bundle_into_registry(bundle)
    rows = []
    for op in manifest.ops:
        impl = next(v for v in REGISTRY.variants(op.slot) if v.variant == op.variant)
        signature = inspect.signature(impl.entry)
        signature.bind_partial(None)  # the node dispatcher supplies prepared positionally
        if impl.prepare is not None:
            inspect.signature(impl.prepare).bind(None)
        rows.append({"node": op.slot, "variant": op.variant, "entry": str(signature)})
    return rows


def _child(arguments: list[str], log: Path, *, timeout: float) -> None:
    """Bound the owned process group, retaining its output even after a failure."""
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "cacheon.miner_check", *arguments],
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            finally:
                # The leader can exit while a rank still holds the process group.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            raise
    if returncode:
        raise RuntimeError(f"check process exited {returncode}; see {log}")


def verify_nodes(bundle: str) -> int:
    """Scan and smoke-test imports/signatures; numerical checking needs the model."""
    from cacheon.cli import cmd_scan

    try:
        _manifest(bundle)
        if cmd_scan(argparse.Namespace(bundle=bundle)):
            return 2
        with tempfile.TemporaryDirectory(prefix="cacheon_node_smoke_") as directory:
            root = Path(directory)
            try:
                _child(["smoke", str(Path(bundle).resolve()), str(root / "result.json")],
                       root / "smoke.log", timeout=120)
                for row in json.loads((root / "result.json").read_text()):
                    print(f"  [INTERFACE OK] {row['node']} variant={row['variant']!r} entry{row['entry']}")
            finally:
                if (root / "smoke.log").exists():
                    print((root / "smoke.log").read_text(), end="")
        print("Scan and import/signature smoke passed. Forward math, preparation and graphs "
              "require cacheon check in the arena image.")
        return 0
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _engine(config: dict, bundle: str, requests: list[dict], root: Path,
            *, phase: str, seed: int) -> None:
    """The exp_cell procedure, using the same scheduler binder as qualification."""
    from cacheon import bootstrap, receipts, seam
    from cacheon.eval.engine_worker import (
        _active_execution_members, _candidate_receipt_failure, _engine_child_failed,
        _report_running_engine_failures, _require_execution_completion,
    )

    receipt_dir = root / "receipts"
    receipt_dir.mkdir()
    os.environ.update(
        CACHEON_ACTIVE="1", CACHEON_BUNDLE_PATH=bundle, CACHEON_FRAMEWORK_MODE="0",
        CACHEON_SEAM_RECEIPT_DIR=str(receipt_dir), SGLANG_PLUGINS="cacheon",
        CACHEON_SLOT_AUDIT="1" if phase == "audit" else "0",
        CACHEON_SLOT_AUDIT_SEED=str(seed), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    seam.mark_driver()
    bootstrap.install()
    import sglang as sgl

    options = dict(config)
    if phase == "audit":
        options["disable_cuda_graph"] = True
    options["custom_sigquit_handler"] = _engine_child_failed
    report = {"phase": phase, "sglang_version": sgl.__version__, "engine": options,
              "decision": "ERROR", "detail": "engine did not complete"}
    engine = None
    try:
        engine = sgl.Engine(**options)
        _report_running_engine_failures(engine)
        for request in requests:
            engine.generate(**request)
        members = receipts.require(str(receipt_dir), "active", context="miner check")
        count = int(config.get("tp_size", 1))
        slots = _active_execution_members(members, expected_member_count=count)
        _require_execution_completion(
            str(receipt_dir), active_receipts=members, expected_slots=slots,
            expected_member_count=count, require_captured=phase == "graph",
        )
        report.update(decision="COMPLETED", detail="engine calls completed")
    except BaseException as exc:
        report["detail"] = _candidate_receipt_failure(str(receipt_dir), receipts) or f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["receipts"] = {
            kind: receipts.collect(str(receipt_dir), kind)
            for kind in ("active", "audit", "completed", "failed", "load_failed", "not_selected")
        }
        (root / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        if engine is not None:
            engine.shutdown()


def _table(report: dict, slots: list[str], ranks: int) -> None:
    audits = {(r.get("slot"), r.get("rank")): r for r in report["receipts"]["audit"]}
    completed = {(r.get("slot"), r.get("rank")): r for r in report["receipts"]["completed"]}
    print("NODE ADDRESS | RANK | AUDIT WINDOWS | VIOLATIONS | WORST PASS | CAPTURED")
    for slot in slots:
        for rank in range(ranks):
            audit = audits.get((slot, rank), {})
            done = completed.get((slot, rank), {})
            print(f"{slot} | {rank} | {audit.get('n', 0)} | "
                  f"{audit.get('violations', 0)} | {audit.get('worst_frac', '-')} | "
                  f"{done.get('captured', False)}")


def check(args: argparse.Namespace) -> int:
    """Run eager audit then graph execution on the supplied public workload."""
    from cacheon.audit_gate import gate
    from cacheon.cli import cmd_scan

    try:
        if (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0
                or args.minimum_audit_windows < 1):
            raise ValueError("timeout and minimum audit windows must be positive")
        manifest = _manifest(args.bundle)
        if cmd_scan(argparse.Namespace(bundle=args.bundle)):
            return 2
        config = json.loads(Path(args.engine_config).read_text())
        requests = json.loads(Path(args.requests).read_text())
        if not isinstance(config, dict) or not isinstance(requests, list) or not requests:
            raise ValueError("engine config must be an object and requests a nonempty array")
        if not all(isinstance(request, dict) for request in requests):
            raise ValueError("each request must be an Engine.generate keyword-argument object")
        model = str(Path(args.model).resolve(strict=True))
        if "model_path" in config and str(Path(config["model_path"]).resolve()) != model:
            raise ValueError("--model differs from engine config model_path")
        config["model_path"] = model
        if config.get("disable_cuda_graph", False):
            raise ValueError("the graph check requires the published graphs-on engine config")
        root = Path(args.output).resolve()
        if root == Path(args.bundle).resolve() or Path(args.bundle).resolve() in root.parents:
            raise ValueError("check output must be outside the bundle")
        root.mkdir(parents=True, exist_ok=False)
        (root / "inputs.json").write_text(json.dumps({
            "config": config, "requests": requests, "bundle": str(Path(args.bundle).resolve()),
            "seed": args.seed,
        }, indent=2) + "\n")
        slots = sorted({op.slot for op in manifest.ops})
        ranks = int(config.get("tp_size", 1))
        for phase in ("audit", "graph"):
            folder = root / phase
            folder.mkdir()
            print(f"Running {phase}; log: {folder / 'engine.log'}", flush=True)
            try:
                _child([phase, str(root / "inputs.json"), str(folder)],
                       folder / "engine.log", timeout=args.timeout_seconds)
                if not (folder / "result.json").is_file():
                    raise RuntimeError(f"check process returned no report; see {folder / 'engine.log'}")
            finally:
                result = folder / "result.json"
                if result.exists():
                    report = json.loads(result.read_text())
                    _table(report, slots, ranks)
                    if report["decision"] == "ERROR":
                        print(report["detail"], file=sys.stderr)
            if phase == "audit":
                decision, detail = gate(report["receipts"]["audit"],
                                        min_calls=args.minimum_audit_windows,
                                        expected_slots=slots, expected_member_count=ranks)
                report.update(decision=decision, detail=detail)
                result.write_text(json.dumps(report, indent=2) + "\n")
                print(f"{decision}: {detail}")
                if decision != "PASS":
                    return 2 if decision == "FAIL" else 3
        print(f"Development audit and capture checks passed. Evidence: {root}. "
              "This is not end-to-end quality, a speed win or qualification.")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def add_parser(subparsers) -> None:
    """Register the miner check command without importing candidate/runtime code."""
    parser = subparsers.add_parser("check", help="run node audit and graph checks in the arena image")
    parser.add_argument("bundle")
    parser.add_argument("--model", required=True, help="local model directory inside the image")
    parser.add_argument("--engine-config", required=True, help="published SGLang engine options JSON")
    parser.add_argument("--requests", required=True, help="JSON array of Engine.generate argument objects")
    parser.add_argument("--output", required=True, help="new directory for logs, inputs and raw receipts")
    parser.add_argument("--seed", type=int, default=7, help="shared audit sampling seed")
    parser.add_argument("--minimum-audit-windows", type=int, default=4, help="arena audit coverage minimum per address/rank")
    parser.add_argument("--timeout-seconds", type=float, default=1800, help="wall bound per fresh engine, including load")
    parser.set_defaults(func=check)


def _main() -> None:
    phase, source, destination = sys.argv[1:]
    if phase == "smoke":
        Path(destination).write_text(json.dumps(_smoke(source)))
    else:
        inputs = json.loads(Path(source).read_text())
        _engine(inputs["config"], inputs["bundle"], inputs["requests"], Path(destination),
                phase=phase, seed=inputs["seed"])


if __name__ == "__main__":
    _main()
