from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
    sha256_hex,
)
from optima.stack_manifest import (
    EngineReleaseManifest,
    EvaluationStackContext,
    EvaluationStackManifest,
    IntegratedContributionRef,
    ProposalContributionRef,
    ReleaseStackContext,
    StackManifestError,
    contribution_ref_from_dict,
    stack_manifest_from_dict,
)


def _d(char: str) -> str:
    return char * 64


TARGET_A = "attention.msa_prefill_block_score"
TARGET_B = "collective.moe_epilogue.v1"


def _catalog(*, marker: str = "base") -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy_version": "target-catalog.v1",
        "targets": [
            {"target_id": TARGET_A, "marker": marker},
            {"target_id": TARGET_B, "marker": marker},
        ],
        "composition_rules": [],
    }


def _catalog_digest(snapshot: dict[str, object]) -> str:
    return canonical_digest("optima.target-catalog", snapshot)


def _catalog_specs(snapshot: dict[str, object]) -> dict[str, str]:
    return {
        row["target_id"]: canonical_digest("optima.target-spec", row)
        for row in snapshot["targets"]  # type: ignore[union-attr]
    }


SPEC_A = _catalog_specs(_catalog())[TARGET_A]
SPEC_B = _catalog_specs(_catalog())[TARGET_B]


def _proposal(
    target: str = TARGET_A,
    *,
    spec: str = SPEC_A,
    artifact: str = _d("3"),
    selected: str = _d("4"),
    attribution: str = _d("5"),
) -> ProposalContributionRef:
    return ProposalContributionRef(
        target_id=target,
        target_spec_digest=spec,
        artifact_digest=artifact,
        selected_payload_digest=selected,
        attribution_digest=attribution,
    )


def _integrated(
    target: str = TARGET_B,
    *,
    spec: str = SPEC_B,
    source: str = _d("6"),
    selected: str = _d("7"),
    attribution: str = _d("8"),
    integration: str = _d("9"),
) -> IntegratedContributionRef:
    return IntegratedContributionRef(
        target_id=target,
        target_spec_digest=spec,
        integrated_source_tree_digest=source,
        selected_payload_digest=selected,
        attribution_digest=attribution,
        integration_record_digest=integration,
    )


def _eval(
    *,
    entries: object | None = None,
    runtime: str = _d("a"),
    base: str = _d("b"),
    arena: str = _d("c"),
    catalog: dict[str, object] | None = None,
) -> EvaluationStackManifest:
    snapshot = catalog or _catalog()
    return EvaluationStackManifest(
        runtime_digest=runtime,
        base_engine_digest=base,
        arena_digest=arena,
        catalog_snapshot=snapshot,
        catalog_digest=_catalog_digest(snapshot),
        entries={} if entries is None else entries,  # type: ignore[arg-type]
    )


def _eval_context(
    *,
    runtime: str = _d("a"),
    base: str = _d("b"),
    arena: str = _d("c"),
    catalog: dict[str, object] | None = None,
    specs: dict[str, str] | None = None,
) -> EvaluationStackContext:
    snapshot = catalog or _catalog()
    return EvaluationStackContext(
        runtime_digest=runtime,
        base_engine_digest=base,
        arena_digest=arena,
        catalog_snapshot=snapshot,
        catalog_digest=_catalog_digest(snapshot),
        target_spec_digests=_catalog_specs(snapshot) if specs is None else specs,
    )


def test_canonical_identity_is_order_stable_domain_separated_and_strict() -> None:
    left = {"z": [1, True, None], "a": {"unicode": "λ"}}
    right = {"a": {"unicode": "λ"}, "z": (1, True, None)}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_digest("optima.test.left", left) == canonical_digest(
        "optima.test.left", right
    )
    assert canonical_digest("optima.test.left", left) != canonical_digest(
        "optima.test.right", left
    )
    assert sha256_hex(b"payload") == sha256_hex(b"payload")
    assert require_sha256_hex(_d("a")) == _d("a")

    for invalid in (1.0, {"x": float("nan")}, {1: "non-string"}, {"x"}):
        with pytest.raises(StackIdentityError):
            canonical_json_bytes(invalid)
    with pytest.raises(StackIdentityError):
        canonical_digest("Bad Domain", {})
    with pytest.raises(StackIdentityError):
        require_sha256_hex(_d("A"))
    with pytest.raises(TypeError):
        sha256_hex("payload")  # type: ignore[arg-type]


def test_contribution_identities_keep_artifact_selected_and_attribution_separate() -> None:
    base = _proposal()
    padded = replace(base, artifact_digest=_d("d"))
    reattributed = replace(base, attribution_digest=_d("e"))
    changed_payload = replace(base, selected_payload_digest=_d("f"))

    assert padded.digest != base.digest
    assert padded.selected_delta_digest == base.selected_delta_digest
    assert reattributed.digest != base.digest
    assert reattributed.selected_delta_digest == base.selected_delta_digest
    assert changed_payload.selected_delta_digest != base.selected_delta_digest

    reviewed = _integrated(
        target=base.target_id,
        spec=base.target_spec_digest,
        selected=base.selected_payload_digest,
    )
    assert reviewed.selected_delta_digest == base.selected_delta_digest
    assert reviewed.digest != base.digest


def test_contribution_parsing_is_discriminated_and_rejects_cross_type_fields() -> None:
    proposal = _proposal()
    integrated = _integrated()
    assert contribution_ref_from_dict(proposal.to_dict()) == proposal
    assert contribution_ref_from_dict(integrated.to_dict()) == integrated

    proposal_with_integration = {
        **proposal.to_dict(),
        "integration_record_digest": _d("a"),
    }
    integrated_with_artifact = {
        **integrated.to_dict(),
        "artifact_digest": _d("b"),
    }
    with pytest.raises(StackManifestError, match="fields mismatch"):
        contribution_ref_from_dict(proposal_with_integration)
    with pytest.raises(StackManifestError, match="fields mismatch"):
        contribution_ref_from_dict(integrated_with_artifact)
    with pytest.raises(StackManifestError, match="requires type"):
        contribution_ref_from_dict({"target_id": TARGET_A})


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_id", " ../escape"),
        ("target_spec_digest", _d("A")),
        ("artifact_digest", "0" * 63),
        ("selected_payload_digest", "not-a-digest"),
        ("attribution_digest", None),
        ("schema_version", True),
        ("schema_version", 2),
    ],
)
def test_proposal_ref_rejects_every_malformed_identity(field: str, value: object) -> None:
    kwargs = {
        "target_id": TARGET_A,
        "target_spec_digest": SPEC_A,
        "artifact_digest": _d("3"),
        "selected_payload_digest": _d("4"),
        "attribution_digest": _d("5"),
        "schema_version": 1,
    }
    kwargs[field] = value
    with pytest.raises(StackManifestError):
        ProposalContributionRef(**kwargs)  # type: ignore[arg-type]


def test_evaluation_manifest_is_canonical_immutable_and_round_trips() -> None:
    proposal = _proposal()
    integrated = _integrated()
    left = _eval(entries=[(TARGET_B, integrated), (TARGET_A, proposal)])
    right = _eval(entries={TARGET_A: proposal, TARGET_B: integrated})

    assert left == right
    assert left.digest == right.digest
    assert list(left.entries) == sorted((TARGET_A, TARGET_B))
    assert stack_manifest_from_dict(left.to_dict()) == left
    encoded = canonical_json_bytes(left.to_dict())
    assert stack_manifest_from_dict(json.loads(encoded)) == left

    with pytest.raises(TypeError):
        left.entries[TARGET_A] = integrated  # type: ignore[index]
    detached = left.catalog_snapshot
    detached["targets"] = []
    assert left.catalog_snapshot["targets"]


def test_stock_only_manifest_and_pure_replacement() -> None:
    incumbent = _eval()
    first = incumbent.with_contribution(_proposal())
    second = first.with_contribution(
        _integrated(target=TARGET_B), remove=(TARGET_A,)
    )

    assert not incumbent.entries
    assert set(first.entries) == {TARGET_A}
    assert set(second.entries) == {TARGET_B}
    assert incumbent.digest != first.digest != second.digest
    with pytest.raises(StackManifestError, match="inactive target"):
        incumbent.with_contribution(_proposal(), remove=(TARGET_B,))
    with pytest.raises(StackManifestError, match="duplicate"):
        first.with_contribution(_proposal(), remove=(TARGET_A, TARGET_A))


def test_entry_key_must_match_ref_and_duplicate_pairs_reject() -> None:
    proposal = _proposal()
    with pytest.raises(StackManifestError, match="does not match"):
        _eval(entries={TARGET_B: proposal})
    with pytest.raises(StackManifestError, match="duplicate"):
        _eval(entries=[(TARGET_A, proposal), (TARGET_A, proposal)])


def test_catalog_digest_must_bind_embedded_snapshot() -> None:
    snapshot = _catalog()
    with pytest.raises(StackManifestError, match="does not match"):
        EvaluationStackManifest(
            runtime_digest=_d("a"),
            base_engine_digest=_d("b"),
            arena_digest=_d("c"),
            catalog_snapshot=snapshot,
            catalog_digest=_d("d"),
            entries={},
        )
    with pytest.raises(StackManifestError, match="float"):
        EvaluationStackManifest(
            runtime_digest=_d("a"),
            base_engine_digest=_d("b"),
            arena_digest=_d("c"),
            catalog_snapshot={"schema_version": 1, "bad": 1.5},
            catalog_digest=_d("d"),
            entries={},
        )


@pytest.mark.parametrize(
    "context",
    [
        _eval_context(runtime=_d("d")),
        _eval_context(base=_d("d")),
        _eval_context(arena=_d("d")),
        _eval_context(catalog=_catalog(marker="changed")),
    ],
)
def test_explicit_evaluation_context_rejects_every_stale_binding(
    context: EvaluationStackContext,
) -> None:
    manifest = _eval(entries={TARGET_A: _proposal()})
    with pytest.raises(StackManifestError):
        manifest.validate_against(context)


@pytest.mark.parametrize(
    "specs",
    [
        {TARGET_A: _d("d"), TARGET_B: SPEC_B},
        {TARGET_B: SPEC_B},
    ],
)
def test_evaluation_context_rejects_split_brain_target_specs(specs) -> None:
    with pytest.raises(StackManifestError, match="complete catalog_snapshot"):
        _eval_context(specs=specs)


def test_structural_parse_is_context_free_then_expected_context_authorizes() -> None:
    stale = _eval(entries={TARGET_A: _proposal()}, runtime=_d("d"))
    reopened = EvaluationStackManifest.from_dict(stale.to_dict())
    assert reopened == stale
    with pytest.raises(StackManifestError, match="runtime"):
        reopened.validate_against(_eval_context())

    current = _eval(entries={TARGET_A: _proposal()})
    assert current.validate_against(_eval_context()) is None


def test_release_is_integrated_only_round_trips_and_has_no_arena() -> None:
    snapshot = _catalog()
    integrated = _integrated()
    release = EngineReleaseManifest(
        runtime_digest=_d("a"),
        base_engine_digest=_d("b"),
        catalog_snapshot=snapshot,
        catalog_digest=_catalog_digest(snapshot),
        entries={TARGET_B: integrated},
    )
    context = ReleaseStackContext(
        runtime_digest=_d("a"),
        base_engine_digest=_d("b"),
        catalog_snapshot=snapshot,
        catalog_digest=_catalog_digest(snapshot),
        target_spec_digests={TARGET_A: SPEC_A, TARGET_B: SPEC_B},
    )

    assert "arena_digest" not in release.to_dict()
    assert not hasattr(release, "with_contribution")
    assert EngineReleaseManifest.from_dict(release.to_dict()) == release
    assert stack_manifest_from_dict(release.to_dict()) == release
    assert release.validate_against(context) is None

    with pytest.raises(StackManifestError, match="integrated contributions only"):
        EngineReleaseManifest(
            runtime_digest=_d("a"),
            base_engine_digest=_d("b"),
            catalog_snapshot=snapshot,
            catalog_digest=_catalog_digest(snapshot),
            entries={TARGET_A: _proposal()},  # type: ignore[dict-item]
        )
    hostile = release.to_dict()
    hostile["entries"] = {TARGET_A: _proposal().to_dict()}
    with pytest.raises(StackManifestError, match="integrated contributions only"):
        EngineReleaseManifest.from_dict(hostile)


def test_release_context_rejects_runtime_base_catalog_and_target_spec() -> None:
    snapshot = _catalog()
    release = EngineReleaseManifest(
        runtime_digest=_d("a"),
        base_engine_digest=_d("b"),
        catalog_snapshot=snapshot,
        catalog_digest=_catalog_digest(snapshot),
        entries={TARGET_B: _integrated()},
    )

    for context in (
        ReleaseStackContext(
            runtime_digest=_d("c"),
            base_engine_digest=_d("b"),
            catalog_snapshot=snapshot,
            catalog_digest=_catalog_digest(snapshot),
            target_spec_digests={TARGET_A: SPEC_A, TARGET_B: SPEC_B},
        ),
        ReleaseStackContext(
            runtime_digest=_d("a"),
            base_engine_digest=_d("c"),
            catalog_snapshot=snapshot,
            catalog_digest=_catalog_digest(snapshot),
            target_spec_digests={TARGET_A: SPEC_A, TARGET_B: SPEC_B},
        ),
        ReleaseStackContext(
            runtime_digest=_d("a"),
            base_engine_digest=_d("b"),
            catalog_snapshot=_catalog(marker="stale"),
            catalog_digest=_catalog_digest(_catalog(marker="stale")),
            target_spec_digests=_catalog_specs(_catalog(marker="stale")),
        ),
    ):
        with pytest.raises(StackManifestError):
            release.validate_against(context)

    with pytest.raises(StackManifestError, match="complete catalog_snapshot"):
        ReleaseStackContext(
            runtime_digest=_d("a"),
            base_engine_digest=_d("b"),
            catalog_snapshot=snapshot,
            catalog_digest=_catalog_digest(snapshot),
            target_spec_digests={TARGET_A: SPEC_A, TARGET_B: _d("c")},
        )


def test_import_surface_is_stdlib_only_and_does_not_require_bittensor_or_torch() -> None:
    code = """
import os, sys
sys.path.insert(0, os.getcwd())
import optima.stack_manifest
assert 'torch' not in sys.modules
assert 'bittensor' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        check=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
