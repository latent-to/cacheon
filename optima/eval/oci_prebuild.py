"""Trusted artifact builder entry for the immutable OCI arena image.

This process never imports bundle Python and never dlopens candidate native code.
It invokes only validator-owned rebuild patchers in ``phase='build'``, materializes
declared dependency/system source overlays, validates their content-addressed
stamps, and writes a small receipt into the controller's artifact mount.  The same
image/GPU/toolchain later used for scoring therefore owns every build identity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from optima.eval.oci_protocol import (
    CONTAINER_ARTIFACT_PATH,
    CONTAINER_BUNDLE_PATH,
    CONTAINER_SOURCE_PATH,
)


_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")


class OCIPrebuildError(RuntimeError):
    pass


def prebuild(*, request_id: str) -> Path:
    if _REQUEST_ID.fullmatch(request_id) is None:
        raise OCIPrebuildError("prebuild request id is malformed")
    bundle = Path(CONTAINER_BUNDLE_PATH)
    artifact = Path(CONTAINER_ARTIFACT_PATH)
    source = Path(CONTAINER_SOURCE_PATH)
    if not bundle.is_dir() or not source.is_dir() or not artifact.is_dir():
        raise OCIPrebuildError("prebuild input/artifact mounts are missing")

    from optima.bundle_hash import content_hash
    from optima.manifest import (
        all_declared_cuda_sources,
        load_manifest,
    )

    manifest = load_manifest(bundle)
    bundle_hash = content_hash(bundle)
    os.environ.update(
        OPTIMA_ACTIVE="0",
        OPTIMA_BUNDLE_PATH=str(bundle),
        OPTIMA_REBUILD_PHASE="build",
        OPTIMA_EXPECTED_BUNDLE_HASH=bundle_hash,
        OPTIMA_REPO_ROOT=str(source),
        OPTIMA_CUDA_EXT_CACHE=f"{artifact}/cuda_ext",
        OPTIMA_DEP_OVERLAY_CACHE=f"{artifact}/dep_overlay",
    )
    rebuilt = False
    from optima.rebuild import apply_rebuild_plan

    rebuilt = apply_rebuild_plan(bundle, phase="build")
    if (manifest.dep_patches or all_declared_cuda_sources(bundle, manifest)) and not rebuilt:
        raise OCIPrebuildError(
            "declared dependency/CUDA artifacts require a validator-reviewed rebuild plan"
        )

    dep_targets: list[str] = []
    if manifest.dep_patches:
        from optima.dep_policy import read_validated_overlay

        for target in sorted({entry.target for entry in manifest.dep_patches}):
            read_validated_overlay(bundle, target)
            dep_targets.append(target)

    system_dest = ""
    system_key = ""
    if manifest.system is not None:
        arena = os.environ.get("OPTIMA_OCI_ARENA_NAME", "").strip()
        target = os.environ.get("OPTIMA_OCI_COMPETITION_TARGET", "").strip()
        if not arena or not target:
            raise OCIPrebuildError("system prebuild lacks profile-owned arena/target")
        from optima.system_patch import (
            materialize_system_overlay,
            read_validated_system_overlay,
        )

        root = artifact / "system_overlay"
        system_dest = str(materialize_system_overlay(
            bundle,
            competition_target=target,
            arena_name=arena,
            cache_root=root,
        ))
        identity, _stamp, _dest = read_validated_system_overlay(
            bundle,
            competition_target=target,
            arena_name=arena,
            cache_root=root,
        )
        system_key = identity.cache_key

    receipt = {
        "schema": "optima-oci-prebuild-v1",
        "request_id": request_id,
        "bundle_hash": bundle_hash,
        "rebuild_plan": rebuilt,
        "dep_targets": dep_targets,
        "system_cache_key": system_key,
        "system_dest": system_dest,
    }
    receipt_dir = artifact / "prebuild_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    destination = receipt_dir / f"{request_id}.json"
    fd, temporary = tempfile.mkstemp(prefix=f".{request_id}.", dir=receipt_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m optima.eval.oci_prebuild")
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args(argv)
    prebuild(request_id=args.request_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
