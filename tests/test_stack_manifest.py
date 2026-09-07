from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from cacheon.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
    sha256_hex,
)
from cacheon.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
    StackManifestError,
    contribution_ref_from_dict,
)


def _d(char: str) -> str:
    return char * 64


TARGET_A = "attention.msa_prefill_block_score"
TARGET_B = "collective.dp_attention_exchange.v1"


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
    return canonical_digest("cacheon.target-catalog", snapshot)


def _catalog_specs(snapshot: dict[str, object]) -> dict[str, str]:
    return {
        row["target_id"]: canonical_digest("cacheon.target-spec", row)
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
    assert canonical_digest("cacheon.test.left", left) == canonical_digest(
        "cacheon.test.left", right
    )
    assert canonical_digest("cacheon.test.left", left) != canonical_digest(
        "cacheon.test.right", left
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


def test_contribution_parsing_is_discriminated_and_rejects_foreign_fields() -> None:
    proposal = _proposal()
    assert contribution_ref_from_dict(proposal.to_dict()) == proposal

    proposal_with_extra_field = {
        **proposal.to_dict(),
        "integration_record_digest": _d("a"),
    }
    with pytest.raises(StackManifestError, match="fields mismatch"):
        contribution_ref_from_dict(proposal_with_extra_field)
    with pytest.raises(StackManifestError, match="requires type"):
        contribution_ref_from_dict({**proposal.to_dict(), "type": "integrated"})
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
    second = _proposal(TARGET_B, spec=SPEC_B)
    left = _eval(entries=[(TARGET_B, second), (TARGET_A, proposal)])
    right = _eval(entries={TARGET_A: proposal, TARGET_B: second})

    assert left == right
    assert left.digest == right.digest
    assert list(left.entries) == sorted((TARGET_A, TARGET_B))

    with pytest.raises(TypeError):
        left.entries[TARGET_A] = second  # type: ignore[index]
    detached = left.catalog_snapshot
    detached["targets"] = []
    assert left.catalog_snapshot["targets"]


def test_stock_only_manifest_and_pure_replacement() -> None:
    incumbent = _eval()
    first = incumbent.with_contribution(_proposal())
    second = first.with_contribution(
        _proposal(TARGET_B, spec=SPEC_B), remove=(TARGET_A,)
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


def test_import_surface_is_stdlib_only_and_does_not_require_bittensor_or_torch() -> None:
    code = """
import os, sys
sys.path.insert(0, os.getcwd())
import cacheon.stack_manifest
assert 'torch' not in sys.modules
assert 'bittensor' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        check=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
