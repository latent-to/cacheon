"""Hash-bound bundle manifests preserve both cutover reader spellings."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheon.bundle_hash import content_hash
from cacheon.manifest import (
    ABI_VERSION,
    ManifestError,
    load_manifest,
)


_TARGET_PROFILES = (
    "activation.silu_and_mul",
    "collective.ar_residual_rmsnorm",
)


def _bundle(root: Path, abi_version: str, *, slot: str) -> Path:
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
        f"slot = \"{slot}\"\n"
        "source = \"kernels/candidate.py\"\n"
        "entry = \"candidate\"\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("slot", _TARGET_PROFILES)
def test_cacheon_abi_spelling_loads_without_normalizing(
    tmp_path: Path, slot: str
) -> None:
    bundle = _bundle(tmp_path / "bundle", ABI_VERSION, slot=slot)
    manifest_bytes = (bundle / "manifest.toml").read_bytes()
    committed_digest = content_hash(bundle)

    parsed = load_manifest(bundle)

    assert parsed.abi_version == ABI_VERSION
    assert parsed.raw["abi_version"] == ABI_VERSION
    assert (bundle / "manifest.toml").read_bytes() == manifest_bytes
    assert content_hash(bundle) == committed_digest


@pytest.mark.parametrize("slot", _TARGET_PROFILES)
def test_pre_cutover_abi_spelling_loads_without_normalizing(
    tmp_path: Path, slot: str
) -> None:
    pre_cutover = "optima-op-abi-v0"
    bundle = _bundle(tmp_path / "bundle", pre_cutover, slot=slot)
    manifest_bytes = (bundle / "manifest.toml").read_bytes()
    committed_digest = content_hash(bundle)

    parsed = load_manifest(bundle)

    assert parsed.abi_version == pre_cutover
    assert parsed.raw["abi_version"] == pre_cutover
    assert (bundle / "manifest.toml").read_bytes() == manifest_bytes
    assert content_hash(bundle) == committed_digest


def test_unknown_abi_spelling_is_refused(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "bundle",
        "unknown-op-abi-v0",
        slot="activation.silu_and_mul",
    )
    with pytest.raises(ManifestError, match="unsupported abi_version"):
        load_manifest(bundle)


def test_screen_consumers_use_the_single_compatibility_reader() -> None:
    from cacheon.eval import b300_screen_stages, resident_screen_lane

    assert b300_screen_stages.load_manifest is load_manifest
    assert resident_screen_lane.load_manifest is load_manifest
