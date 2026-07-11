from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from optima.competition import (
    ATOMIC_MODE,
    SLOT_MODE,
    SYSTEM_MODE,
    CompetitionError,
    resolve_competition,
)
from optima.device_component import UNTRUSTED_HOST_SYSTEM_TARGET
from optima.manifest import CompetitionEntry, ManifestError, load_manifest


DEEP_TARGET = "collective.moe_epilogue.v1"
DEEP_MEMBERS = (
    "collective.ar_residual_rmsnorm",
    "collective.moe_finalize_ar_rmsnorm",
)


def _bundle(tmp_path: Path, *, competition: str = "", slots: tuple[str, ...]) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "kernel.py").write_text("def entry(*args):\n    return None\n")
    rows = [
        'bundle_id = "test-bundle"',
        'abi_version = "optima-op-abi-v0"',
        "",
    ]
    if competition:
        rows.extend([competition, ""])
    for slot in slots:
        rows.extend(
            [
                "[[ops]]",
                f'slot = "{slot}"',
                'source = "kernel.py"',
                'entry = "entry"',
                "",
            ]
        )
    (root / "manifest.toml").write_text("\n".join(rows))
    return root


def _deep_bundle(tmp_path: Path) -> Path:
    return _bundle(
        tmp_path,
        competition=(
            "[competition]\n"
            f'target = "{UNTRUSTED_HOST_SYSTEM_TARGET}"\n'
            'mode = "system"'
        ),
        slots=DEEP_MEMBERS,
    )


def test_legacy_singleton_resolves_to_its_slot(tmp_path):
    manifest = load_manifest(_bundle(tmp_path, slots=("norm.rmsnorm",)))

    resolved = resolve_competition(manifest)

    assert resolved.target == "norm.rmsnorm"
    assert resolved.mode == SLOT_MODE
    assert resolved.members == ("norm.rmsnorm",)
    assert resolved.crownable is False
    assert resolved.legacy is True
    assert "untrusted_host" in (resolved.reason or "")
    with pytest.raises(CompetitionError, match="isolated system lane"):
        resolve_competition(manifest, for_settlement=True)


def test_explicit_singleton_parses_and_resolves(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            competition=(
                "[competition]\n"
                'target = "norm.rmsnorm"\n'
                'mode = "slot"'
            ),
            slots=("norm.rmsnorm",),
        )
    )

    assert manifest.competition == CompetitionEntry(
        target="norm.rmsnorm", mode="slot"
    )
    resolved = resolve_competition(manifest)
    assert resolved.target == "norm.rmsnorm"
    assert resolved.legacy is False
    assert not resolved.crownable


@pytest.mark.parametrize(
    "table, match",
    [
        ("competition = []", "must be a .* table"),
        (
            "[competition]\n"
            'target = "norm.rmsnorm"\n'
            'mode = "slot"\n'
            'members = "miner-controlled"',
            "unknown keys",
        ),
        ("[competition]\nmode = \"slot\"", "target.*simple identifier"),
        (
            "[competition]\n"
            'target = "norm.rmsnorm"\n'
            'mode = "per_slot"',
            "mode.*slot.*atomic",
        ),
    ],
)
def test_manifest_rejects_malformed_competition(tmp_path, table, match):
    with pytest.raises(ManifestError, match=match):
        load_manifest(_bundle(tmp_path, competition=table, slots=("norm.rmsnorm",)))


def test_deep_bundle_resolves_as_one_explicit_whole_system_product(tmp_path):
    manifest = load_manifest(_deep_bundle(tmp_path))
    # Miner-controlled op order must not leak into component settlement identity.
    manifest = replace(manifest, ops=tuple(reversed(manifest.ops)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_competition(manifest, for_settlement=True)

    assert not caught
    assert resolved.target == UNTRUSTED_HOST_SYSTEM_TARGET
    assert resolved.mode == SYSTEM_MODE
    assert resolved.members == ()
    assert resolved.crownable is True
    assert resolved.legacy is False


def test_deep_manifest_explicitly_declares_validator_owned_system_target(tmp_path):
    manifest = load_manifest(_deep_bundle(tmp_path))
    assert manifest.competition == CompetitionEntry(
        target=UNTRUSTED_HOST_SYSTEM_TARGET,
        mode=SYSTEM_MODE,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_competition(
            manifest,
            for_settlement=True,
            warn_legacy=False,
        )
    assert not caught
    assert resolved.crownable
    assert resolved.target == UNTRUSTED_HOST_SYSTEM_TARGET
    assert resolved.mode == SYSTEM_MODE
    assert resolved.members == ()
    assert not resolved.legacy


def test_explicit_deep_pair_resolves_without_legacy_warning(tmp_path):
    manifest = replace(
        load_manifest(_deep_bundle(tmp_path)),
        competition=CompetitionEntry(target=DEEP_TARGET, mode=ATOMIC_MODE),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_competition(manifest)
    assert not caught
    assert resolved.members == DEEP_MEMBERS
    assert resolved.legacy is False
    assert not resolved.crownable


def test_atomic_target_counts_semantic_members_not_variant_rows(tmp_path):
    base = load_manifest(_deep_bundle(tmp_path))
    shallow = base.ops[0]
    manifest = replace(
        base,
        ops=(
            replace(shallow, variant="small"),
            replace(shallow, variant="large"),
            base.ops[1],
        ),
        competition=CompetitionEntry(target=DEEP_TARGET, mode=ATOMIC_MODE),
    )

    resolved = resolve_competition(manifest)

    assert resolved.members == DEEP_MEMBERS
    assert not resolved.crownable


def test_unknown_legacy_multi_op_can_intake_but_not_settle(tmp_path):
    manifest = load_manifest(
        _bundle(tmp_path, slots=("norm.rmsnorm", "activation.silu_and_mul"))
    )

    resolved = resolve_competition(manifest)
    assert resolved.crownable is False
    assert resolved.target is None
    assert "may verify but cannot settle" in (resolved.reason or "")

    with pytest.raises(CompetitionError, match="no registered exact competition"):
        resolve_competition(manifest, for_settlement=True)


def test_explicit_atomic_target_must_be_registered(tmp_path):
    manifest = replace(
        load_manifest(_deep_bundle(tmp_path)),
        competition=CompetitionEntry(target="miner.chosen.target", mode=ATOMIC_MODE),
    )
    with pytest.raises(CompetitionError, match="unknown atomic competition target"):
        resolve_competition(manifest, for_settlement=True)


def test_explicit_atomic_target_requires_exact_members(tmp_path):
    deep = load_manifest(_deep_bundle(tmp_path))
    manifest = replace(
        deep,
        ops=(deep.ops[0],),
        competition=CompetitionEntry(target=DEEP_TARGET, mode=ATOMIC_MODE),
    )
    with pytest.raises(CompetitionError, match="requires exact members"):
        resolve_competition(manifest, for_settlement=True)


def test_slot_mode_rejects_multi_op_manifest(tmp_path):
    manifest = replace(
        load_manifest(_deep_bundle(tmp_path)),
        competition=CompetitionEntry(
            target="collective.ar_residual_rmsnorm", mode=SLOT_MODE
        ),
    )
    with pytest.raises(CompetitionError, match="requires exactly one op"):
        resolve_competition(manifest, for_settlement=True)
