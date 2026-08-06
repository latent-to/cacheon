"""Hash-bound bundle manifests accept only the Cacheon ABI spelling."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheon.bundle_hash import content_hash
from cacheon.manifest import (
    ABI_VERSION,
    ManifestError,
    load_manifest,
)


def _bundle(root: Path, abi_version: str) -> Path:
    root.mkdir()
    kernels = root / "kernels"
    kernels.mkdir()
    (kernels / "candidate.py").write_text(
        "def candidate(x):\n    return x\n",
        encoding="utf-8",
    )
    (root / "manifest.toml").write_text(
        "bundle_id = \"rename-compat\"\n"
        f"abi_version = \"{abi_version}\"\n"
        "\n"
        "[[ops]]\n"
        "slot = \"activation.silu_and_mul\"\n"
        "source = \"kernels/candidate.py\"\n"
        "entry = \"candidate\"\n",
        encoding="utf-8",
    )
    return root


def test_cacheon_abi_spelling_loads_without_normalizing(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle", ABI_VERSION)
    manifest_bytes = (bundle / "manifest.toml").read_bytes()
    committed_digest = content_hash(bundle)

    parsed = load_manifest(bundle)

    assert parsed.abi_version == ABI_VERSION
    assert parsed.raw["abi_version"] == ABI_VERSION
    assert (bundle / "manifest.toml").read_bytes() == manifest_bytes
    assert content_hash(bundle) == committed_digest


def test_retired_optima_abi_spelling_is_refused(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle", "optima-op-abi-v0")
    with pytest.raises(ManifestError, match="unsupported abi_version"):
        load_manifest(bundle)
