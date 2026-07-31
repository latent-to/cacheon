"""Rename compatibility for hash-bound miner bundle manifests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheon.bundle_hash import content_hash
from cacheon.engine_tree import _runtime_manifest
from cacheon.manifest import (
    ABI_VERSION,
    ARTIFACT_TARGET_AUTHORITY_SCHEMA,
    CACHEON_ABI_VERSION_ALIAS,
    SUPPORTED_ABI_VERSIONS,
    ManifestError,
    load_manifest,
    static_artifact_target_authority,
)
from cacheon.stack_identity import canonical_digest


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


@pytest.mark.parametrize("abi_version", [ABI_VERSION, CACHEON_ABI_VERSION_ALIAS])
def test_rename_accepts_only_declared_v0_spellings_without_normalizing(
    tmp_path: Path,
    abi_version: str,
) -> None:
    bundle = _bundle(tmp_path / "bundle", abi_version)
    manifest_bytes = (bundle / "manifest.toml").read_bytes()
    committed_digest = content_hash(bundle)

    parsed = load_manifest(bundle)

    assert parsed.abi_version == abi_version
    assert parsed.raw["abi_version"] == abi_version
    assert (bundle / "manifest.toml").read_bytes() == manifest_bytes
    assert content_hash(bundle) == committed_digest


def test_rename_keeps_published_and_alias_bundle_identities_distinct(
    tmp_path: Path,
) -> None:
    published = _bundle(tmp_path / "published", ABI_VERSION)
    alias = _bundle(tmp_path / "alias", CACHEON_ABI_VERSION_ALIAS)

    assert content_hash(published) != content_hash(alias)
    assert load_manifest(published).abi_version == ABI_VERSION
    assert load_manifest(alias).abi_version == CACHEON_ABI_VERSION_ALIAS


def test_rename_writers_keep_the_published_protocol_abi() -> None:
    assert ABI_VERSION == "optima-op-abi-v0"
    assert b'abi_version = "optima-op-abi-v0"' in _runtime_manifest([], [])


def test_rename_examples_keep_authoring_the_published_protocol_abi() -> None:
    examples = Path(__file__).parents[1] / "examples"
    manifests = sorted(examples.glob("*/manifest.toml"))

    assert manifests
    for manifest_path in manifests:
        assert load_manifest(manifest_path.parent).abi_version == ABI_VERSION


def test_rename_preserves_static_artifact_authority_identity_domains() -> None:
    authority = static_artifact_target_authority("activation.silu_and_mul")

    assert ARTIFACT_TARGET_AUTHORITY_SCHEMA == "optima.artifact-target-authority.v1"
    assert authority.call_abi_digest == canonical_digest(
        "optima.artifact-call-abi", authority.call_abi.identity_snapshot()
    )
    assert authority.digest == canonical_digest(
        "optima.artifact-target-authority", authority.snapshot()
    )


def test_rename_abi_compatibility_is_a_closed_set(tmp_path: Path) -> None:
    assert SUPPORTED_ABI_VERSIONS == frozenset(
        {"optima-op-abi-v0", "cacheon-op-abi-v0"}
    )
    bundle = _bundle(tmp_path / "bundle", "optima-op-abi-v0-suffix")

    with pytest.raises(ManifestError, match="unsupported abi_version"):
        load_manifest(bundle)
