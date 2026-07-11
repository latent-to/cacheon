"""Rebuild-plan helper for the framework-mode escape hatch.

Framework mode lets a candidate open a larger backend surface than the narrow
tensor-in/out dispatcher can express (a backend swap, a source recompile). That is
the *fenced escape hatch*, NOT the core slot contract — see docs/SLOT_CONTRACT.md.

The hard rule here: a ``rebuild.json`` step may reference **only a validator-shipped,
reviewed patcher** that lives in this repo's dedicated patcher directory
(``optima/patchers/``), selected via ``repo_python``. It must NOT execute
bundle-supplied code — that would be arbitrary miner RCE in the candidate process,
which no-egress isolation bounds but does not prevent (it can still touch the
filesystem, the shared sglang install, secrets on the box). This mirrors how PyTorch
gates backends: you submit a patch to core to add one; you do not ship arbitrary
code into the dispatcher. A miner who needs a patcher gets it *reviewed and merged*
into ``optima/patchers/`` first; then a bundle's ``rebuild.json`` may select it by name.

CONTAINMENT IS NOT ENOUGH: allowing any ``.py`` under the repo root (the earlier
behavior) let a bundle ``runpy`` an arbitrary repo module as ``__main__`` — e.g. a
CLI whose ``__main__`` has side effects — which is not a "reviewed patcher" in any
meaningful sense. So the resolved script must live under ``optima/patchers/`` AND be a
``.py`` file; anything else is refused. The repo root is derived from THIS package's
location (deterministic), not the process CWD, and is overridable only by the operator
via ``OPTIMA_REPO_ROOT``.

(The earlier ``bundle_python`` step type — run an arbitrary script from the bundle —
is deliberately removed; it is rejected with a clear error.)
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Literal

# Reviewed patchers live ONLY here (repo-relative). A repo_python step may select a file
# under this dir and nowhere else, so "reviewed patcher" is an enforced boundary, not an
# honor system. Empty today (the feature is forward-looking); a plan naming a missing
# patcher fails closed.
_PATCHER_SUBDIR = ("optima", "patchers")


class RebuildError(RuntimeError):
    pass


RebuildPhase = Literal["all", "build", "load"]
_REBUILD_PHASES = frozenset({"all", "build", "load"})
_BUNDLE_HASH_ENV = "OPTIMA_BUNDLE_CONTENT_HASH"
_EXPECTED_BUNDLE_HASH_ENV = "OPTIMA_EXPECTED_BUNDLE_HASH"


def _repo_root() -> Path:
    """The repo root. Deterministic (this package's parent), NOT the process CWD — the
    old ``Path.cwd()`` default made the patcher boundary depend on where the validator
    happened to launch from. ``OPTIMA_REPO_ROOT`` overrides for relocated deployments."""
    env = os.environ.get("OPTIMA_REPO_ROOT")
    if env:
        return Path(env).resolve()
    # optima/rebuild.py -> optima/ -> repo root
    return Path(__file__).resolve().parents[1]


def apply_rebuild_plan(bundle_path: str | Path, *, phase: RebuildPhase = "all") -> bool:
    """Apply ``rebuild.json`` from ``bundle_path`` if present.

    Returns True when a plan was found and applied. All paths are repo-relative or
    bundle-relative and containment-checked. ``phase`` lets reviewed patchers split
    artifact construction from runtime loading: the trusted timing process invokes a
    separate ``build`` worker, while only the untrusted scheduler invokes ``load``.
    Patchers that do not need two phases may safely perform the same idempotent work in
    both. Network/process isolation is handled by the caller.
    """
    if phase not in _REBUILD_PHASES:
        raise RebuildError(f"unsupported rebuild phase: {phase!r}")
    bundle = Path(bundle_path).resolve()
    # ``validator_device`` components do not carry a miner-selected rebuild plan.
    # Their offline cubin builder is an unconditional validator-owned phase, so a
    # bundle cannot swap it for a host extension patcher in ``rebuild.json``.
    prepared_device = False
    if (bundle / "manifest.toml").is_file():
        from optima.device_component import prepare_device_artifacts

        prepared_device = prepare_device_artifacts(bundle, phase=phase)
    plan_path = bundle / "rebuild.json"
    if not plan_path.exists():
        return prepared_device
    if not plan_path.is_file():
        raise RebuildError(f"rebuild plan is not a file: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise RebuildError("rebuild.json must be an object")
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise RebuildError("rebuild.json 'steps' must be a list")

    # A rebuild artifact is native code, so a miner-controlled display name is not a
    # cache identity.  Bind every patcher invocation to the deterministic hash of the
    # COMPLETE submitted tree (manifest, Python shims, .cu/.cuh closure, patches and
    # rebuild plan).  The subprocess caller may additionally pin the hash it observed
    # before spawning; any mutation between controller and worker then fails closed.
    from optima.bundle_hash import content_hash

    bundle_hash = content_hash(bundle)
    expected_hash = os.environ.get(_EXPECTED_BUNDLE_HASH_ENV, "").strip()
    if expected_hash and expected_hash != bundle_hash:
        raise RebuildError(
            "bundle changed before rebuild: controller expected "
            f"{expected_hash}, worker observed {bundle_hash}"
        )

    repo_root = _repo_root()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise RebuildError(f"rebuild step {i} must be an object")
        typ = step.get("type")
        if typ == "repo_python":
            # ONLY validator-shipped, reviewed patchers under optima/patchers/. Never
            # bundle code, and never an arbitrary repo module.
            script = _safe_patcher_path(repo_root, str(step.get("path", "")))
            _run_python_script(
                script, bundle=bundle, phase=phase, bundle_hash=bundle_hash
            )
        elif typ == "bundle_python":
            raise RebuildError(
                "rebuild step 'bundle_python' is not allowed: a bundle may not execute its "
                "own code in the candidate process (arbitrary RCE). Use a validator-shipped, "
                "reviewed 'repo_python' patcher instead. See docs/SLOT_CONTRACT.md."
            )
        else:
            raise RebuildError(f"unsupported rebuild step type: {typ!r}")
    return True


def apply_rebuild_plan_subprocess(
    bundle_path: str | Path,
    *,
    phase: RebuildPhase = "build",
    timeout_s: float | None = None,
) -> bool:
    """Apply a rebuild plan in a disposable child process.

    This is the only API the trusted timing/controller process should use. Even a
    validator-reviewed compiler patcher consumes attacker-controlled CUDA source; a
    compiler bug or an accidental extension import must not gain access to the Python
    process that owns elapsed time and result serialization.
    """
    if phase not in _REBUILD_PHASES:
        raise RebuildError(f"unsupported rebuild phase: {phase!r}")
    bundle = Path(bundle_path).resolve()
    plan = bundle / "rebuild.json"
    has_device_product = False
    if (bundle / "manifest.toml").is_file():
        from optima.manifest import VALIDATOR_DEVICE_EXECUTION, load_manifest

        has_device_product = any(
            op.execution_class == VALIDATOR_DEVICE_EXECUTION
            for op in load_manifest(bundle).ops
        )
    if not plan.exists() and not has_device_product:
        return False
    if timeout_s is None:
        timeout_s = float(os.environ.get("OPTIMA_REBUILD_TIMEOUT_S", "1800"))
    from optima.bundle_hash import content_hash

    bundle_hash = content_hash(bundle)
    cmd = [sys.executable, "-m", "optima.rebuild", "--phase", phase, str(bundle)]
    child_env = os.environ.copy()
    # A site-wide optima bootstrap may run before ``-m optima.rebuild``. Ensure the
    # build worker cannot inherit an active seam and import a candidate as a side
    # effect of interpreter startup.
    child_env.update(
        OPTIMA_ACTIVE="0",
        OPTIMA_BUNDLE_PATH="",
        OPTIMA_REBUILD_PHASE=phase,
        OPTIMA_EXPECTED_BUNDLE_HASH=bundle_hash,
    )
    try:
        subprocess.run(  # noqa: S603 - fixed argv
            cmd, check=True, timeout=timeout_s, env=child_env
        )
    except subprocess.TimeoutExpired as exc:
        raise RebuildError(
            f"rebuild {phase} worker exceeded {timeout_s:g}s for {bundle}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RebuildError(
            f"rebuild {phase} worker failed with exit code {exc.returncode} for {bundle}"
        ) from exc
    return True


def _safe_patcher_path(repo_root: Path, rel: str) -> Path:
    """Resolve ``rel`` to a reviewed patcher, refusing anything outside ``optima/patchers/``.

    ``rel`` may be given relative to the repo root (``optima/patchers/foo.py``) or to the
    patcher dir itself (``foo.py``); either way the RESOLVED path must be contained in the
    patcher dir, be a regular ``.py`` file, and not be a symlink (which could re-point
    outside the reviewed set)."""
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        raise RebuildError(f"patcher path must be a simple relative path: {rel!r}")
    patcher_dir = repo_root.joinpath(*_PATCHER_SUBDIR).resolve()
    # Accept either a repo-relative or a patcher-dir-relative spelling.
    candidate = (repo_root / rel) if rel.startswith(os.path.join(*_PATCHER_SUBDIR)) else (patcher_dir / rel)
    if candidate.is_symlink():
        raise RebuildError(f"patcher path must not be a symlink: {rel!r}")
    p = candidate.resolve()
    if p != patcher_dir and patcher_dir not in p.parents:
        raise RebuildError(
            f"patcher path escapes the reviewed patcher dir {os.path.join(*_PATCHER_SUBDIR)!r}: {rel!r}"
        )
    if p.suffix != ".py":
        raise RebuildError(f"patcher must be a .py file: {rel!r}")
    if p.is_symlink() or not p.is_file():
        raise RebuildError(f"reviewed patcher not found under {os.path.join(*_PATCHER_SUBDIR)!r}: {rel!r}")
    return p


def _run_python_script(
    script: Path, *, bundle: Path, phase: RebuildPhase, bundle_hash: str
) -> None:
    """Run a reviewed patcher with the triggering bundle's path in the environment.

    ``OPTIMA_BUNDLE_PATH`` is the patcher contract: every caller of
    ``apply_rebuild_plan`` (engine launch, distributed-verify ranks, CLI smoke) hands
    the bundle path as an argument, so the plan must not depend on who set what env —
    the earlier env-only convention silently no-op'd patchers in verify ranks (the
    build skipped, the shim fell back to its reference path, and the "verify"
    validated nothing)."""
    old_argv = sys.argv
    old_bundle = os.environ.get("OPTIMA_BUNDLE_PATH")
    old_phase = os.environ.get("OPTIMA_REBUILD_PHASE")
    old_bundle_hash = os.environ.get(_BUNDLE_HASH_ENV)
    sys.argv = [str(script)]
    os.environ["OPTIMA_BUNDLE_PATH"] = str(bundle)
    os.environ["OPTIMA_REBUILD_PHASE"] = phase
    os.environ[_BUNDLE_HASH_ENV] = bundle_hash
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
        if old_bundle is None:
            os.environ.pop("OPTIMA_BUNDLE_PATH", None)
        else:
            os.environ["OPTIMA_BUNDLE_PATH"] = old_bundle
        if old_phase is None:
            os.environ.pop("OPTIMA_REBUILD_PHASE", None)
        else:
            os.environ["OPTIMA_REBUILD_PHASE"] = old_phase
        if old_bundle_hash is None:
            os.environ.pop(_BUNDLE_HASH_ENV, None)
        else:
            os.environ[_BUNDLE_HASH_ENV] = old_bundle_hash


def _main(argv: list[str] | None = None) -> int:
    """Internal subprocess entry point; not a miner-facing command."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m optima.rebuild")
    parser.add_argument("--phase", choices=sorted(_REBUILD_PHASES), required=True)
    parser.add_argument("bundle")
    args = parser.parse_args(argv)
    apply_rebuild_plan(args.bundle, phase=args.phase)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(_main())
