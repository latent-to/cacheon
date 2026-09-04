from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from cacheon.manifest import (
    ABI_VERSION,
    CompetitionEntry,
    Manifest,
    ManifestError,
    load_manifest,
)
from cacheon.target_catalog import (
    CorrectnessContractRef,
    FEATURE_CUDA_SOURCES,
    FEATURE_ENTRY,
    FEATURE_PREPARE,
    FEATURE_REBUILD_BUILD_CUDA_EXT,
    FEATURE_SETUP,
    FEATURE_VARIANTS,
    SINGLETON_TARGET_IDS,
    ResolvedTarget,
    TargetCatalog,
    TargetCatalogError,
    TargetContractRef,
    TargetKind,
    TargetResolutionError,
    TargetSpec,
    ToleranceContractRef,
    default_target_catalog,
    manifest_declared_features,
    resolve_intake_target,
    resolve_target,
)


SILU = "activation.silu_and_mul"


def _bundle(
    tmp_path: Path,
    *,
    rows: tuple[dict[str, object], ...] = ({"slot": SILU},),
    competition: str = "",
) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    lines = [
        'bundle_id = "target-test"',
        f'abi_version = "{ABI_VERSION}"',
        "",
    ]
    if competition:
        lines.extend([competition, ""])
    for index, row in enumerate(rows):
        slot = str(row.get("slot", SILU))
        source = str(row.get("source", f"kernels/k{index}.py"))
        entry = str(row.get("entry", f"entry_{index}"))
        source_path = root / source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(f"def {entry}(*args):\n    return None\n")
        lines.extend(["[[ops]]", f'slot = "{slot}"'])
        for key in (
            "variant",
            "prepare",
            "setup",
            "base_kernel",
            "override_point",
        ):
            if key in row:
                lines.append(f'{key} = "{row[key]}"')
        lines.extend([f'source = "{source}"', f'entry = "{entry}"'])
        if row.get("cuda_sources"):
            cuda = str(row.get("cuda_path", "kernels/k.cu"))
            cuda_path = root / cuda
            cuda_path.parent.mkdir(parents=True, exist_ok=True)
            cuda_path.write_text("// inspected source\n")
            lines.append(f'cuda_sources = ["{cuda}"]')
        if "extra_key" in row:
            lines.append(f'{row["extra_key"]} = "future"')
        lines.append("")
    (root / "manifest.toml").write_text("\n".join(lines))
    return root


def _competition(target: str, mode: str) -> str:
    return f'[competition]\ntarget = "{target}"\nmode = "{mode}"'


def _slot_spec(
    target_id: str,
    *,
    displaces: frozenset[str] = frozenset(),
    conflicts_with: frozenset[str] = frozenset(),
    requires: frozenset[str] = frozenset(),
    features: frozenset[str] = frozenset({FEATURE_ENTRY}),
) -> TargetSpec:
    return TargetSpec(
        target_id=target_id,
        kind=TargetKind.SLOT,
        members=(target_id,),
        displaces=displaces,
        conflicts_with=conflicts_with,
        requires=requires,
        allowed_features=features,
        contract_ref=TargetContractRef(
            schema_version=1,
            slot_id=target_id,
            kind="op",
            entry="entry",
            prepare=None,
            graph_dynamic_inputs=("x",),
            input_abi_id=f"{target_id}.input.v1",
            output_abi_id=f"{target_id}.output.v1",
            reference_id=f"{target_id}.reference.v1",
            verification_profile_id=f"{target_id}.verify.v1",
            binding_family_id="test.binding.v1",
            correctness=CorrectnessContractRef(),
            tolerances=(ToleranceContractRef("float32", "0.0001", "0.0001"),),
        ),
    )


def _atomic_spec(
    target_id: str = "atomic.ab",
    *,
    members: tuple[str, ...] = ("slot.a", "slot.b"),
    displaces: frozenset[str] | None = None,
) -> TargetSpec:
    return TargetSpec(
        target_id=target_id,
        kind=TargetKind.ATOMIC,
        members=members,
        displaces=frozenset(members) if displaces is None else displaces,
        allowed_features=frozenset({FEATURE_ENTRY}),
        atomic_semantics_id=f"{target_id}.semantics.v1",
    )


# -- syntax-only manifest request -------------------------------------------


def test_manifest_parses_syntax_only_competition_request(tmp_path):
    manifest = load_manifest(
        _bundle(tmp_path, competition=_competition(SILU, "slot"))
    )

    assert manifest.competition == CompetitionEntry(target=SILU, mode="slot")


@pytest.mark.parametrize(
    "table, message",
    [
        ("competition = []", "must be a .* table"),
        (
            '[competition]\ntarget = "activation.silu_and_mul"\n'
            'mode = "slot"\nmembers = ["miner"]',
            "unknown keys",
        ),
        ('[competition]\nmode = "slot"', "target.*string"),
        ('[competition]\ntarget = 7\nmode = "slot"', "target.*string"),
        ('[competition]\ntarget = "bad id"\nmode = "slot"', "simple identifier"),
        (f'[competition]\ntarget = "{SILU}"', "mode.*string"),
        (f'[competition]\ntarget = "{SILU}"\nmode = 7', "mode.*string"),
        (
            f'[competition]\ntarget = "{SILU}"\nmode = "per_slot"',
            "slot.*atomic",
        ),
    ],
)
def test_manifest_rejects_malformed_competition(tmp_path, table, message):
    with pytest.raises(ManifestError, match=message):
        load_manifest(_bundle(tmp_path, competition=table))


def test_legacy_system_request_parses_but_never_registers_a_title(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            competition=_competition("sglang.inference.bundle.v1", "system"),
        )
    )

    assert manifest.competition == CompetitionEntry(
        target="sglang.inference.bundle.v1", mode="system"
    )
    resolved = resolve_target(manifest)
    assert not resolved.registered and not resolved.implicit
    assert resolved.target_id is None
    assert "legacy competition mode 'system'" in (resolved.reason or "")
    with pytest.raises(TargetResolutionError, match="legacy competition mode"):
        resolve_intake_target(manifest, observed_features=())


def test_manifest_field_append_preserves_historical_positional_arguments():
    manifest = Manifest("bundle", ABI_VERSION, (), {"old": "raw"})

    assert manifest.raw == {"old": "raw"}
    assert manifest.competition is None


def test_tracked_examples_preserve_legacy_or_name_modern_target_identity():
    examples = Path(__file__).resolve().parents[1] / "examples"
    manifests = sorted(examples.glob("*/manifest.toml"))
    assert manifests
    explicit = {
        "miner_allreduce_torch": CompetitionEntry(
            "collective.all_reduce", "slot"
        ),
        "miner_dense_torch": CompetitionEntry("linear.dense", "slot"),
        "miner_dp_attention_exchange_torch": CompetitionEntry(
            "collective.dp_attention_exchange.v1", "atomic"
        ),
        "miner_fused_add_rmsnorm_torch": CompetitionEntry(
            "norm.fused_add_rmsnorm", "slot"
        ),
        "miner_moe_fused_routed_torch": CompetitionEntry(
            "moe.fused_routed_experts", "slot"
        ),
    }
    for path in manifests:
        assert load_manifest(path.parent).competition == explicit.get(path.parent.name)


# -- canonical resolution ---------------------------------------------------


def test_implicit_and_explicit_singleton_resolve_to_catalog_identity(tmp_path):
    implicit = resolve_target(load_manifest(_bundle(tmp_path / "implicit")))
    explicit = resolve_target(
        load_manifest(
            _bundle(
                tmp_path / "explicit",
                competition=_competition(SILU, "slot"),
            )
        )
    )

    assert implicit == ResolvedTarget(
        target_id=SILU,
        kind=TargetKind.SLOT,
        members=(SILU,),
        registered=True,
        implicit=True,
        observed_features=frozenset({FEATURE_ENTRY}),
        features_complete=False,
    )
    assert explicit.target_id == SILU
    assert explicit.members == (SILU,)
    assert not explicit.implicit


def test_multiple_variants_are_one_semantic_member(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=(
                {"slot": SILU, "variant": "small"},
                {"slot": SILU, "variant": "large"},
            ),
        )
    )

    resolved = resolve_target(manifest)
    assert resolved.members == (SILU,)
    assert resolved.observed_features == frozenset(
        {FEATURE_ENTRY, FEATURE_VARIANTS}
    )


@pytest.mark.parametrize(
    "competition, rows, message",
    [
        (_competition("unknown.target", "slot"), ({"slot": SILU},), "unknown"),
        (
            _competition(SILU, "atomic"),
            ({"slot": SILU},),
            "catalog kind.*not requested",
        ),
        (
            _competition(SILU, "slot"),
            ({"slot": "norm.rmsnorm"},),
            "requires exact members",
        ),
    ],
)
def test_explicit_target_request_fails_closed(tmp_path, competition, rows, message):
    manifest = load_manifest(
        _bundle(tmp_path, competition=competition, rows=rows)
    )
    with pytest.raises(TargetResolutionError, match=message):
        resolve_target(manifest)


def test_programmatic_invalid_mode_still_fails_closed(tmp_path):
    manifest = replace(
        load_manifest(_bundle(tmp_path)),
        competition=CompetitionEntry(target=SILU, mode="per_slot"),
    )
    with pytest.raises(TargetResolutionError, match="unknown competition mode"):
        resolve_target(manifest)


@pytest.mark.parametrize(
    "competition_value, message",
    [
        (object(), "must be a CompetitionEntry"),
        (CompetitionEntry(target=[], mode="slot"), "malformed"),  # type: ignore[arg-type]
        (CompetitionEntry(target=SILU, mode=7), "malformed"),  # type: ignore[arg-type]
    ],
)
def test_programmatic_malformed_request_fails_closed(
    tmp_path, competition_value, message
):
    manifest = replace(
        load_manifest(_bundle(tmp_path)), competition=competition_value
    )
    with pytest.raises(TargetResolutionError, match=message):
        resolve_target(manifest)


def test_unknown_implicit_multi_op_routes_to_discovery(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=({"slot": SILU}, {"slot": "norm.rmsnorm"}),
        )
    )

    resolved = resolve_target(manifest)
    assert not resolved.registered
    assert resolved.target_id is None and resolved.kind is None
    assert resolved.members == (SILU, "norm.rmsnorm")
    assert "future discovery" in (resolved.reason or "")
    with pytest.raises(TargetResolutionError, match="discovery"):
        resolved.require_registered()
    with pytest.raises(TargetResolutionError, match="future discovery"):
        resolve_intake_target(manifest, observed_features=())


# -- validator catalog invariants ------------------------------------------


def test_default_catalog_has_exactly_one_singleton_per_live_slot():
    from cacheon.slots import SLOTS

    catalog = default_target_catalog()
    assert set(SINGLETON_TARGET_IDS) == set(SLOTS)
    for target_id in SINGLETON_TARGET_IDS:
        assert catalog.require(target_id).target_id == target_id


def _decimal(value: object) -> str:
    number = Decimal(str(value))
    if number == 0:
        return "0"
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def test_default_contract_refs_match_every_live_serializable_slot_field():
    from cacheon.slots import SLOTS, slot_for_model

    catalog = default_target_catalog()
    for slot_id, slot in sorted(SLOTS.items()):
        if slot_id in {"moe.fused_experts", "moe.fused_experts_reduce"}:
            slot = slot_for_model(slot_id, "MiniMax-M3-NVFP4")
        ref = catalog.require(slot_id).contract_ref
        assert ref is not None
        assert (ref.slot_id, ref.kind, ref.entry, ref.prepare) == (
            slot.name,
            slot.kind,
            slot.entry,
            slot.prepare,
        )
        assert ref.graph_dynamic_inputs == slot.graph_dynamic_inputs
        assert ref.correctness.snapshot() == {
            "mode": slot.correctness.mode,
            "top_k": slot.correctness.top_k,
            "min_ratio": _decimal(slot.correctness.min_ratio),
            "min_cosine": _decimal(slot.correctness.min_cosine),
            "max_rel_norm_err": _decimal(slot.correctness.max_rel_norm_err),
            "min_overlap": _decimal(slot.correctness.min_overlap),
        }
        assert [row.snapshot() for row in ref.tolerances] == [
            {
                "dtype": str(dtype).removeprefix("torch."),
                "atol": _decimal(tolerance.atol),
                "rtol": _decimal(tolerance.rtol),
            }
            for dtype, tolerance in sorted(
                slot.tolerances.items(), key=lambda row: str(row[0])
            )
        ]
        assert ref.kl_threshold == (
            None if slot.kl_threshold is None else _decimal(slot.kl_threshold)
        )
        assert all(
            re.fullmatch(r".+\.v[1-9][0-9]*", value)
            for value in (
                ref.input_abi_id,
                ref.output_abi_id,
                ref.reference_id,
                ref.verification_profile_id,
                ref.binding_family_id,
            )
        )


def test_target_catalog_import_is_stdlib_only_and_does_not_import_torch():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import cacheon.target_catalog; "
            "assert 'torch' not in sys.modules",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_catalog_snapshot_and_digests_are_canonical_complete_and_immutable():
    a, b = _slot_spec("slot.a"), _slot_spec("slot.b")
    first = TargetCatalog([b, a])
    second = TargetCatalog([a, b])
    assert first.snapshot() == second.snapshot()
    assert first.digest == second.digest
    assert first.target_spec_digest("slot.a") == second.target_spec_digest("slot.a")
    assert len(first.digest) == len(first.contract_digest("slot.a")) == 64

    snapshot = first.snapshot()
    target = snapshot["targets"][0]
    assert set(target) == {
        "target_id",
        "kind",
        "members",
        "displaces",
        "conflicts_with",
        "requires",
        "allowed_features",
        "contract_ref",
        "contract_digest",
    }
    target["members"].append("tamper")
    assert first.snapshot() == second.snapshot()


def test_contract_and_target_digest_rotate_on_manual_or_policy_change():
    original = _slot_spec("slot.a")
    assert original.contract_ref is not None
    revised_ref = replace(original.contract_ref, output_abi_id="slot.a.output.v2")
    revised = replace(original, contract_ref=revised_ref)
    base = TargetCatalog([original])
    changed = TargetCatalog([revised])
    assert base.contract_digest("slot.a") != changed.contract_digest("slot.a")
    assert base.target_spec_digest("slot.a") != changed.target_spec_digest("slot.a")
    assert base.digest != changed.digest

    with_feature = TargetCatalog(
        [replace(original, allowed_features=frozenset({FEATURE_ENTRY, FEATURE_PREPARE}))]
    )
    assert base.target_spec_digest("slot.a") != with_feature.target_spec_digest("slot.a")


def test_contract_decimal_projection_has_one_canonical_zero_and_no_floats():
    row = CorrectnessContractRef(
        min_ratio="1.0000",
        min_cosine="-0.000",
        max_rel_norm_err="1e-5",
        min_overlap=0,
    ).snapshot()
    assert row["min_ratio"] == "1"
    assert row["min_cosine"] == row["min_overlap"] == "0"
    assert row["max_rel_norm_err"] == "0.00001"
    assert not any(isinstance(value, float) for value in row.values())


def test_requires_means_an_active_contribution_and_is_acyclic():
    a = _slot_spec("slot.a")
    b = _slot_spec("slot.b", requires=frozenset({"slot.a"}))
    catalog = TargetCatalog([b, a])
    with pytest.raises(TargetResolutionError, match="stock does not satisfy"):
        catalog.validate_active_targets(("slot.b",))
    assert catalog.validate_active_targets(("slot.b", "slot.a")) == (
        "slot.a",
        "slot.b",
    )
    assert catalog.requires_closure("slot.b") == frozenset({"slot.a"})

    with pytest.raises(TargetCatalogError, match="requires graph.*cycle"):
        TargetCatalog(
            [
                _slot_spec("slot.a", requires=frozenset({"slot.b"})),
                _slot_spec("slot.b", requires=frozenset({"slot.a"})),
            ]
        )
    with pytest.raises(TargetCatalogError, match="requires mutually exclusive"):
        TargetCatalog(
            [
                _slot_spec(
                    "slot.a",
                    requires=frozenset({"slot.b"}),
                    displaces=frozenset({"slot.b"}),
                ),
                _slot_spec("slot.b"),
            ]
        )


@pytest.mark.parametrize(
    "specs, message",
    [
        (
            [
                _slot_spec(
                    "slot.a",
                    requires=frozenset({"slot.b"}),
                    displaces=frozenset({"slot.c"}),
                ),
                _slot_spec("slot.b", requires=frozenset({"slot.c"})),
                _slot_spec("slot.c"),
            ],
            "requires its displacement closure",
        ),
        (
            [
                _slot_spec("slot.a", requires=frozenset({"slot.b"})),
                _slot_spec("slot.b", displaces=frozenset({"slot.a"})),
            ],
            "requires targets that displace it",
        ),
    ],
)
def test_transitive_dependency_displacement_contradictions_reject(specs, message):
    with pytest.raises(TargetCatalogError, match=message):
        TargetCatalog(specs)


@pytest.mark.parametrize(
    "specs, message",
    [
        ([], "must not be empty"),
        ({"slot.a": _slot_spec("slot.a")}, "iterable of TargetSpec"),
        ([_slot_spec("slot.a"), _slot_spec("slot.a")], "duplicate target ID"),
        (
            [TargetSpec("slot.a", TargetKind.SLOT, ())],
            "non-empty sequence",
        ),
        (
            [TargetSpec("slot.a", TargetKind.SLOT, ("slot.a", "slot.a"))],
            "duplicate members",
        ),
        ([_slot_spec("slot.a"), _atomic_spec(members=("slot.a",))], "at least two"),
        (
            [
                _slot_spec("slot.a"),
                _slot_spec("slot.b"),
                _atomic_spec(displaces=frozenset({"slot.a"})),
            ],
            "explicitly displace",
        ),
        (
            [
                _slot_spec("slot.a"),
                _atomic_spec(displaces=frozenset({"slot.a"})),
            ],
            "without registered singletons",
        ),
        (
            [TargetSpec("alias", TargetKind.SLOT, ("slot.a",))],
            "itself as its sole member",
        ),
        (
            [_slot_spec("slot.a", displaces=frozenset({"missing"}))],
            "unknown targets",
        ),
        (
            [_slot_spec("slot.a", displaces=frozenset({"slot.a"}))],
            "itself",
        ),
        (
            [_slot_spec("slot.a", features=frozenset({FEATURE_ENTRY, "future"}))],
            "unknown features",
        ),
        (
            [_slot_spec("slot.a", features=frozenset({FEATURE_ENTRY, FEATURE_SETUP}))],
            "may not allow.*setup",
        ),
        ([_slot_spec("slot.a", features=frozenset())], "must allow the entry"),
    ],
)
def test_catalog_rejects_invalid_validator_policy(specs, message):
    with pytest.raises(TargetCatalogError, match=message):
        TargetCatalog(specs)


def test_catalog_rejects_duplicate_atomic_member_sets():
    specs = [
        _slot_spec("slot.a"),
        _slot_spec("slot.b"),
        _atomic_spec("atomic.ab"),
        _atomic_spec("atomic.alias", members=("slot.b", "slot.a")),
    ]
    with pytest.raises(TargetCatalogError, match="same exact member set"):
        TargetCatalog(specs)


def test_partial_atomic_overlap_rejects_without_member_ownership_semantics():
    singletons = [_slot_spec(f"slot.{name}") for name in ("a", "b", "c")]
    atomic_ab = _atomic_spec("atomic.ab", members=("slot.a", "slot.b"))
    atomic_bc = _atomic_spec("atomic.bc", members=("slot.b", "slot.c"))

    with pytest.raises(TargetCatalogError, match="share members.*explicit"):
        TargetCatalog([*singletons, atomic_ab, atomic_bc])

    atomic_ab = replace(atomic_ab, conflicts_with=frozenset({"atomic.bc"}))
    atomic_bc = replace(atomic_bc, conflicts_with=frozenset({"atomic.ab"}))
    catalog = TargetCatalog([*singletons, atomic_ab, atomic_bc])
    with pytest.raises(TargetResolutionError, match="conflicts"):
        catalog.validate_active_targets(("atomic.ab", "atomic.bc"))


def test_schema_versions_are_type_exact():
    contract = default_target_catalog().require(SILU).contract_ref
    assert contract is not None
    with pytest.raises(TargetCatalogError, match="schema_version"):
        replace(contract, schema_version=True)


def test_conflict_must_be_symmetric_and_not_displaced():
    with pytest.raises(TargetCatalogError, match="must be symmetric"):
        TargetCatalog(
            [
                _slot_spec("slot.a", conflicts_with=frozenset({"slot.b"})),
                _slot_spec("slot.b"),
            ]
        )
    with pytest.raises(TargetCatalogError, match="both displaces and conflicts"):
        TargetCatalog(
            [
                _slot_spec(
                    "slot.a",
                    displaces=frozenset({"slot.b"}),
                    conflicts_with=frozenset({"slot.b"}),
                ),
                _slot_spec("slot.b", conflicts_with=frozenset({"slot.a"})),
            ]
        )


def test_catalog_rejects_displacement_cycle():
    with pytest.raises(TargetCatalogError, match="cycle"):
        TargetCatalog(
            [
                _slot_spec("slot.a", displaces=frozenset({"slot.b"})),
                _slot_spec("slot.b", displaces=frozenset({"slot.a"})),
            ]
        )


def test_catalog_registration_order_does_not_change_resolution():
    a = _slot_spec("slot.a")
    b = _slot_spec("slot.b")
    source = [b, a]
    first = TargetCatalog(source)
    source.clear()
    second = TargetCatalog([a, b])
    assert first.require("slot.a") == second.require("slot.a") == a
    assert first.require("slot.b") == second.require("slot.b") == b


def test_default_displacement_and_conflicts_are_explicit():
    catalog = default_target_catalog()

    assert catalog.require("moe.fused_experts_reduce").displaces == frozenset(
        {"moe.fused_experts"}
    )
    assert catalog.require("moe.fused_routed_experts").conflicts_with == frozenset(
        {"moe.fused_experts_reduce"}
    )
    with pytest.raises(TargetResolutionError, match="displaces"):
        catalog.validate_active_targets(
            ("moe.fused_experts", "moe.fused_experts_reduce")
        )
    assert all(
        not any(feature.startswith("aot:") for feature in catalog.require(slot).allowed_features)
        for slot in ("moe.fused_experts", "moe.fused_experts_reduce")
    )
    with pytest.raises(TargetResolutionError, match="must be strings"):
        catalog.validate_active_targets((["unhashable"],))  # type: ignore[list-item]


# -- contribution feature admission ---------------------------------------


def test_standard_target_admits_variants_prepare_override_and_cuda(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=(
                {
                    "slot": "moe.fused_experts",
                    "variant": "sm120",
                    "prepare": "prepare",
                    "base_kernel": "nvfp4_moe",
                    "override_point": "epilogue",
                    "cuda_sources": True,
                },
            ),
        )
    )

    resolved = resolve_target(manifest)
    assert resolved.registered
    assert not resolved.features_complete
    with pytest.raises(TargetResolutionError, match="lacks complete"):
        resolved.require_complete_features()
    assert manifest_declared_features(manifest) >= {
        FEATURE_ENTRY,
        FEATURE_VARIANTS,
        FEATURE_PREPARE,
        FEATURE_CUDA_SOURCES,
    }


def test_setup_manifest_parses_but_registered_target_rejects_it(tmp_path):
    manifest = load_manifest(
        _bundle(tmp_path, rows=({"slot": SILU, "setup": "setup"},))
    )
    assert FEATURE_SETUP in manifest_declared_features(manifest)
    with pytest.raises(TargetResolutionError, match="fenced discovery lane"):
        resolve_target(manifest)


def test_unknown_op_extension_cannot_bypass_feature_policy(tmp_path):
    manifest = load_manifest(
        _bundle(tmp_path, rows=({"slot": SILU, "extra_key": "future_knob"},))
    )
    with pytest.raises(TargetResolutionError, match="op_extra:future_knob"):
        resolve_target(manifest)


def test_shallow_native_bundle_admits_exact_reviewed_builder(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=(
                {
                    "slot": "collective.ar_residual_rmsnorm",
                    "cuda_sources": True,
                },
            ),
            competition=_competition("collective.ar_residual_rmsnorm", "slot"),
        )
    )

    assert FEATURE_REBUILD_BUILD_CUDA_EXT not in manifest_declared_features(manifest)
    resolved = resolve_intake_target(
        manifest, observed_features=(FEATURE_REBUILD_BUILD_CUDA_EXT,)
    )
    assert resolved.registered and resolved.features_complete
    assert resolved.observed_features >= {
        FEATURE_CUDA_SOURCES,
        FEATURE_REBUILD_BUILD_CUDA_EXT,
    }


def test_exact_observed_rebuild_capability_is_target_policy_not_manifest_data(tmp_path):
    manifest = load_manifest(_bundle(tmp_path))
    assert FEATURE_REBUILD_BUILD_CUDA_EXT not in manifest_declared_features(manifest)
    with pytest.raises(TargetResolutionError, match="rebuild:unknown"):
        resolve_intake_target(manifest, observed_features=("rebuild:unknown",))


def test_complete_feature_evidence_pairs_cuda_units_with_builder(tmp_path):
    cuda = load_manifest(
        _bundle(
            tmp_path / "cuda",
            rows=({"slot": SILU, "cuda_sources": True},),
            competition=_competition(SILU, "slot"),
        )
    )
    with pytest.raises(TargetResolutionError, match="CUDA sources without"):
        resolve_intake_target(cuda, observed_features=())

    plain = load_manifest(_bundle(tmp_path / "plain"))
    with pytest.raises(TargetResolutionError, match="without declared CUDA"):
        resolve_intake_target(
            plain, observed_features=(FEATURE_REBUILD_BUILD_CUDA_EXT,)
        )

    header_only = load_manifest(
        _bundle(
            tmp_path / "header",
            rows=(
                {
                    "slot": SILU,
                    "cuda_sources": True,
                    "cuda_path": "kernels/header.cuh",
                },
            ),
            competition=_competition(SILU, "slot"),
        )
    )
    with pytest.raises(TargetResolutionError, match="compilation unit"):
        resolve_intake_target(
            header_only,
            observed_features=(FEATURE_REBUILD_BUILD_CUDA_EXT,),
        )


def test_observed_features_argument_is_strict(tmp_path):
    manifest = load_manifest(_bundle(tmp_path))
    with pytest.raises(TargetResolutionError, match="iterable"):
        default_target_catalog().resolve_intake(
            manifest, observed_features="rebuild:build_cuda_ext"
        )
    with pytest.raises(TargetResolutionError, match="non-empty strings"):
        default_target_catalog().resolve_intake(
            manifest, observed_features=("",)
        )


def test_target_catalog_has_no_economic_or_trust_policy_imports():
    source = (
        Path(__file__).resolve().parents[1] / "cacheon/target_catalog.py"
    ).read_text()
    forbidden = (
        "cacheon.chain",
        "cacheon.commit_reveal",
        "cacheon.device_component",
        "cacheon.system_patch",
        "crownable",
        "for_settlement",
    )
    assert not [token for token in forbidden if token in source]
