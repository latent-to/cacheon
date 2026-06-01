"""Trusted rebuild-plan helper for framework-mode bundles.

Framework mode lets a candidate open a larger backend surface than the narrow
tensor-in/out dispatcher can express. The untrusted bundle still should not run an
arbitrary shell script in the trusted driver. Instead, the bundle may include a
small data-only ``rebuild.json`` plan, and this validator-owned helper applies the
allowed source patch/rebuild steps inside the isolated candidate process before
SGLang imports.

This is an early rebuild tier, not a general package manager. It is deliberately
narrow: run a vetted repo-local Python patcher, then let the backend JIT/rebuild
happen during candidate engine startup in the same no-egress namespace.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


class RebuildError(RuntimeError):
    pass


def apply_rebuild_plan(bundle_path: str | Path) -> bool:
    """Apply ``rebuild.json`` from ``bundle_path`` if present.

    Returns True when a plan was found and applied. All paths are repo-relative or
    bundle-relative and containment-checked. Network isolation is handled by the
    caller before this function is invoked.
    """
    bundle = Path(bundle_path).resolve()
    plan_path = bundle / "rebuild.json"
    if not plan_path.exists():
        return False
    if not plan_path.is_file():
        raise RebuildError(f"rebuild plan is not a file: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise RebuildError("rebuild.json must be an object")
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise RebuildError("rebuild.json 'steps' must be a list")

    repo_root = Path(os.environ.get("OPTIMA_REPO_ROOT", Path.cwd())).resolve()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise RebuildError(f"rebuild step {i} must be an object")
        typ = step.get("type")
        if typ == "repo_python":
            script = _safe_repo_path(repo_root, str(step.get("path", "")))
            _run_python_script(script)
        elif typ == "bundle_python":
            script = _safe_bundle_path(bundle, str(step.get("path", "")))
            _run_python_script(script)
        else:
            raise RebuildError(f"unsupported rebuild step type: {typ!r}")
    return True


def _safe_repo_path(repo_root: Path, rel: str) -> Path:
    if not rel or rel.startswith("/"):
        raise RebuildError(f"repo script path must be relative: {rel!r}")
    p = (repo_root / rel).resolve()
    if repo_root != p and repo_root not in p.parents:
        raise RebuildError(f"repo script path escapes repo: {rel!r}")
    if not p.is_file():
        raise RebuildError(f"repo script not found: {rel!r}")
    return p


def _safe_bundle_path(bundle: Path, rel: str) -> Path:
    if not rel or rel.startswith("/"):
        raise RebuildError(f"bundle script path must be relative: {rel!r}")
    p = (bundle / rel).resolve()
    if bundle != p and bundle not in p.parents:
        raise RebuildError(f"bundle script path escapes bundle: {rel!r}")
    if not p.is_file():
        raise RebuildError(f"bundle script not found: {rel!r}")
    return p


def _run_python_script(script: Path) -> None:
    old_argv = sys.argv
    sys.argv = [str(script)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
