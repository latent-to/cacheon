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
    FEATURE_OVERRIDE,
    FEATURE_PREPARE,
    FEATURE_REBUILD_BUILD_CUDA_EXT,
    FEATURE_SETUP,
    FEATURE_VARIANTS,
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
)


def resolve_target(manifest):
    return default_target_catalog().resolve_manifest(manifest)


def resolve_intake_target(manifest, *, observed_features):
    return default_target_catalog().resolve_intake(manifest, observed_features=observed_features)


FORWARD = "forward_pass"
CACHE = "prefix_cache"
NODE = "model.layers.*.mlp"
CACHE_NODE = "tree_cache"


def _bundle(
    tmp_path: Path,
    *,
    rows: tuple[dict[str, object], ...] = ({"slot": NODE},),
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
        slot = str(row.get("slot", NODE))
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
    features: frozenset[str] = frozenset({FEATURE_ENTRY}),
    roots: tuple[str, ...] = (),
) -> TargetSpec:
    return TargetSpec(
        target_id=target_id,
        kind=TargetKind.SLOT,
        members=(target_id,),
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
        node_roots=roots,
    )


# -- syntax-only manifest request -------------------------------------------


def test_manifest_parses_syntax_only_competition_request(tmp_path):
    manifest = load_manifest(
        _bundle(tmp_path, competition=_competition(FORWARD, "slot"))
    )

    assert manifest.competition == CompetitionEntry(target=FORWARD, mode="slot")


@pytest.mark.parametrize(
    "table, message",
    [
        ("competition = []", "must be a .* table"),
        (
            f'[competition]\ntarget = "{FORWARD}"\n'
            'mode = "slot"\nmembers = ["miner"]',
            "unknown keys",
        ),
        ('[competition]\nmode = "slot"', "target.*string"),
        ('[competition]\ntarget = 7\nmode = "slot"', "target.*string"),
        ('[competition]\ntarget = "bad id"\nmode = "slot"', "simple identifier"),
        (f'[competition]\ntarget = "{FORWARD}"', "mode.*string"),
        (f'[competition]\ntarget = "{FORWARD}"\nmode = 7', "mode.*string"),
        (
            f'[competition]\ntarget = "{FORWARD}"\nmode = "per_slot"',
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
        "miner_node_identity": CompetitionEntry(FORWARD, "slot"),
        "miner_node_wrong": CompetitionEntry(FORWARD, "slot"),
    }
    for path in manifests:
        assert load_manifest(path.parent).competition == explicit.get(path.parent.name)


# -- canonical resolution ---------------------------------------------------


@pytest.mark.parametrize("slot, target", [(NODE, FORWARD), (CACHE_NODE, CACHE)])
def test_implicit_and_explicit_nodes_resolve_to_catalog_identity(tmp_path, slot, target):
    rows = ({"slot": slot},)
    implicit = resolve_target(load_manifest(_bundle(tmp_path / "implicit", rows=rows)))
    explicit = resolve_target(
        load_manifest(
            _bundle(
                tmp_path / "explicit",
                rows=rows,
                competition=_competition(target, "slot"),
            )
        )
    )

    assert implicit == ResolvedTarget(
        target_id=target,
        kind=TargetKind.SLOT,
        members=(slot,),
        registered=True,
        implicit=True,
        observed_features=frozenset({FEATURE_ENTRY}),
        features_complete=False,
    )
    assert explicit.target_id == target
    assert explicit.members == (slot,)
    assert not explicit.implicit


def test_multiple_variants_are_one_semantic_member(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=(
                {"slot": NODE, "variant": "small"},
                {"slot": NODE, "variant": "large"},
            ),
        )
    )

    resolved = resolve_target(manifest)
    assert resolved.members == (NODE,)
    assert resolved.observed_features == frozenset(
        {FEATURE_ENTRY, FEATURE_VARIANTS}
    )


@pytest.mark.parametrize(
    "competition, rows, message",
    [
        (_competition("unknown.target", "slot"), ({"slot": NODE},), "unknown"),
        # The retired atomic mode no longer names a catalog kind.
        (
            _competition(FORWARD, "atomic"),
            ({"slot": NODE},),
            "unknown competition mode 'atomic'",
        ),
        (_competition(FORWARD, "slot"), ({"slot": CACHE_NODE},), "must sit under"),
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
        competition=CompetitionEntry(target=FORWARD, mode="per_slot"),
    )
    with pytest.raises(TargetResolutionError, match="unknown competition mode"):
        resolve_target(manifest)


@pytest.mark.parametrize(
    "competition_value, message",
    [
        (object(), "must be a CompetitionEntry"),
        (CompetitionEntry(target=[], mode="slot"), "malformed"),  # type: ignore[arg-type]
        (CompetitionEntry(target=FORWARD, mode=7), "malformed"),  # type: ignore[arg-type]
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
    # No single target's roots hold both a model node and the prefix cache.
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=({"slot": NODE}, {"slot": CACHE_NODE}),
        )
    )

    resolved = resolve_target(manifest)
    assert not resolved.registered
    assert resolved.target_id is None and resolved.kind is None
    assert resolved.members == (NODE, CACHE_NODE)
    assert "future discovery" in (resolved.reason or "")
    with pytest.raises(TargetResolutionError, match="discovery"):
        resolved.require_registered()
    with pytest.raises(TargetResolutionError, match="future discovery"):
        resolve_intake_target(manifest, observed_features=())


def test_implicit_resolution_never_picks_between_targets_sharing_a_root(tmp_path):
    manifest = load_manifest(_bundle(tmp_path))
    owned = TargetCatalog([_slot_spec("slot.a", roots=("model",))])
    assert owned.resolve_manifest(manifest).target_id == "slot.a"

    shared = TargetCatalog(
        [_slot_spec("slot.a", roots=("model",)), _slot_spec("slot.b", roots=("model",))]
    )
    resolved = shared.resolve_manifest(manifest)
    assert not resolved.registered and resolved.target_id is None
    assert "future discovery" in (resolved.reason or "")


# -- validator catalog invariants ------------------------------------------


def test_default_catalog_registers_the_forward_pass_and_the_prefix_cache():
    catalog = default_target_catalog()
    targets = [row["target_id"] for row in catalog.snapshot()["targets"]]
    assert targets == [FORWARD, CACHE]
    assert {target: catalog.require(target).node_roots for target in targets} == {
        FORWARD: ("logits_processor", "model"),
        CACHE: (CACHE_NODE,),
    }


def _decimal(value: object) -> str:
    number = Decimal(str(value))
    if number == 0:
        return "0"
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def test_default_contract_refs_match_the_live_node_audit():
    from cacheon.integrations import sglang_nodes

    catalog = default_target_catalog()
    forward = catalog.require(FORWARD).contract_ref
    assert forward is not None
    assert forward.correctness.snapshot() == {
        "mode": sglang_nodes._MODE,
        "top_k": 0,
        "min_ratio": _decimal(sglang_nodes._ROW_BAR),
        "min_cosine": "0",
        "max_rel_norm_err": "0",
        "min_overlap": "0",
    }
    assert [row.snapshot() for row in forward.tolerances] == [
        {"dtype": "bfloat16", "atol": "0", "rtol": _decimal(sglang_nodes._FLOOR)}
    ]
    for target in (FORWARD, CACHE):
        ref = catalog.require(target).contract_ref
        assert ref is not None
        assert (ref.slot_id, ref.graph_dynamic_inputs, ref.kl_threshold) == (target, (), None)
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
    # Retained empty so recorded target-spec digests do not rotate.
    assert target["displaces"] == target["conflicts_with"] == target["requires"] == []
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


@pytest.mark.parametrize(
    "specs, message",
    [
        ([], "must not be empty"),
        ({"slot.a": _slot_spec("slot.a")}, "iterable of TargetSpec"),
        ([_slot_spec("slot.a"), _slot_spec("slot.a")], "duplicate target ID"),
        ([TargetSpec("slot.a", TargetKind.SLOT, ())], "itself as its sole member"),
        (
            [TargetSpec("slot.a", TargetKind.SLOT, ("slot.a", "slot.a"))],
            "itself as its sole member",
        ),
        (
            [TargetSpec("alias", TargetKind.SLOT, ("slot.a",))],
            "itself as its sole member",
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


def test_schema_versions_are_type_exact():
    contract = default_target_catalog().require(FORWARD).contract_ref
    assert contract is not None
    with pytest.raises(TargetCatalogError, match="schema_version"):
        replace(contract, schema_version=True)


def test_catalog_registration_order_does_not_change_resolution():
    a = _slot_spec("slot.a")
    b = _slot_spec("slot.b")
    source = [b, a]
    first = TargetCatalog(source)
    source.clear()
    second = TargetCatalog([a, b])
    assert first.require("slot.a") == second.require("slot.a") == a
    assert first.require("slot.b") == second.require("slot.b") == b


def test_default_targets_are_one_active_stack_without_aot_features():
    catalog = default_target_catalog()

    # Model and cache replacements never displace each other.
    assert catalog.validate_active_targets((CACHE, FORWARD)) == (FORWARD, CACHE)
    assert all(
        not any(feature.startswith("aot:") for feature in catalog.require(target).allowed_features)
        for target in (FORWARD, CACHE)
    )
    with pytest.raises(TargetResolutionError, match="must be strings"):
        catalog.validate_active_targets((["unhashable"],))  # type: ignore[list-item]


# -- contribution feature admission ---------------------------------------


def test_standard_target_admits_variants_prepare_and_cuda(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=(
                {
                    "slot": NODE,
                    "variant": "sm120",
                    "prepare": "prepare",
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


def test_override_points_are_refused_at_resolution(tmp_path):
    # FEATURE_OVERRIDE stays in the admitted set only because forward_pass's
    # target-spec digest binds it; no manifest may declare it any more.
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=({"slot": NODE, "base_kernel": "nvfp4_moe", "override_point": "epilogue"},),
        )
    )
    with pytest.raises(TargetResolutionError, match="override points are retired"):
        resolve_target(manifest)
    assert FEATURE_OVERRIDE in default_target_catalog().require("forward_pass").allowed_features


def test_setup_manifest_parses_but_registered_target_rejects_it(tmp_path):
    manifest = load_manifest(
        _bundle(tmp_path, rows=({"slot": NODE, "setup": "setup"},))
    )
    assert FEATURE_SETUP in manifest_declared_features(manifest)
    with pytest.raises(TargetResolutionError, match="fenced discovery lane"):
        resolve_target(manifest)


def test_unknown_op_extension_cannot_bypass_feature_policy(tmp_path):
    manifest = load_manifest(
        _bundle(tmp_path, rows=({"slot": NODE, "extra_key": "future_knob"},))
    )
    with pytest.raises(TargetResolutionError, match="op_extra:future_knob"):
        resolve_target(manifest)


def test_shallow_native_bundle_admits_exact_reviewed_builder(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=(
                {
                    "slot": NODE,
                    "cuda_sources": True,
                },
            ),
            competition=_competition(FORWARD, "slot"),
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
            rows=({"slot": NODE, "cuda_sources": True},),
            competition=_competition(FORWARD, "slot"),
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
                    "slot": NODE,
                    "cuda_sources": True,
                    "cuda_path": "kernels/header.cuh",
                },
            ),
            competition=_competition(FORWARD, "slot"),
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
