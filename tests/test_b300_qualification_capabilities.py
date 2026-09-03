from __future__ import annotations

import hashlib
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cacheon.eval import b300_qualification_capabilities as capabilities
from cacheon.eval.oci_process import OCIQuiescenceReceipt
from cacheon.eval.qualification import SelectionCommitment


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _commitment(seed: str, source: str) -> SelectionCommitment:
    return SelectionCommitment(
        _h(f"plan:{seed}"),
        _h(f"reference:{seed}"),
        _h(f"workload:{seed}"),
        source,
        _h(f"secret:{seed}"),
        tuple(sorted((_h(f"prompt:{seed}:0"), _h(f"prompt:{seed}:1")))),
        1,
    )


def _teardown(
    *,
    namespace: str = "qualified-namespace",
    manager: str = "1" * 32,
    sequence: int = 1,
    observed: float = 100.0,
) -> OCIQuiescenceReceipt:
    return OCIQuiescenceReceipt(
        "cacheon.oci-quiescence.v1",
        "qualified-executor",
        manager,
        _h(namespace),
        sequence,
        observed,
        (),
        (),
        (),
    )


def test_secret_store_restart_idempotency_isolation_and_sealed_mode(tmp_path: Path) -> None:
    root = tmp_path / "selection-secrets"
    first_ref, second_ref = _h("secret-ref:a"), _h("secret-ref:b")
    first, second = b"a" * 32, b"b" * 32
    store = capabilities.DurableSelectionSecretStore(root)
    store.put(first_ref, first)
    store.put(first_ref, first)
    store.put(second_ref, second)

    restarted = capabilities.DurableSelectionSecretStore(root)
    assert restarted(first_ref) == first
    assert restarted(second_ref) == second
    assert stat.S_IMODE((root / f"secret-{first_ref}.json").stat().st_mode) == 0o400
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="different"):
        restarted.put(first_ref, b"c" * 32)
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="unavailable"):
        restarted(_h("missing"))


def test_secret_store_corruption_and_partial_publication_fail_closed(tmp_path: Path) -> None:
    root, reference = tmp_path / "secrets", _h("reference")
    store = capabilities.DurableSelectionSecretStore(root)
    store.put(reference, b"s" * 32)
    path = root / f"secret-{reference}.json"
    path.chmod(0o600)
    path.write_bytes(b"{}")
    path.chmod(0o400)
    with pytest.raises(capabilities.B300QualificationCapabilityError):
        store(reference)

    other_root, other_ref = tmp_path / "partial-secrets", _h("partial-reference")
    partial_store = capabilities.DurableSelectionSecretStore(other_root)
    (other_root / f".secret-{other_ref}.json.partial").write_bytes(b"partial")
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="partial"):
        partial_store.put(other_ref, b"p" * 32)


def test_entropy_provider_restart_and_compatible_teardown_reopen(tmp_path: Path) -> None:
    source = _h("entropy-source")
    commitment = _commitment("a", source)
    calls: list[str] = []

    def entropy_source(observed, teardown):
        calls.append(observed.digest + teardown.digest)
        return b"post-commit entropy".ljust(32, b"!")

    root = tmp_path / "entropy"
    provider = capabilities.DurableSelectionEntropyProvider(
        root, source_digest=source, entropy_source=entropy_source
    )
    first = provider(commitment, _teardown())
    assert len(calls) == 1

    def must_not_resample(*_args):
        raise AssertionError("durable entropy was resampled")

    restarted = capabilities.DurableSelectionEntropyProvider(
        root, source_digest=source, entropy_source=must_not_resample
    )
    assert restarted(commitment, _teardown()) == first
    later = _teardown(manager="2" * 32, observed=101.0)
    assert restarted(commitment, later) == first

    with pytest.raises(capabilities.B300QualificationCapabilityError, match="teardown"):
        restarted(commitment, _teardown(namespace="foreign", observed=102.0))


def test_entropy_provider_isolates_commitments_and_serializes_concurrency(tmp_path: Path) -> None:
    source = _h("entropy-source")
    first, second = _commitment("first", source), _commitment("second", source)
    calls: list[str] = []

    def entropy_source(commitment, _teardown):
        calls.append(commitment.digest)
        return bytes.fromhex(commitment.digest)

    provider = capabilities.DurableSelectionEntropyProvider(
        tmp_path / "entropy", source_digest=source, entropy_source=entropy_source
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = tuple(pool.map(lambda _index: provider(first, _teardown()), range(4)))
    assert len(set(rows)) == 1
    assert calls == [first.digest]
    second_receipt = provider(second, _teardown())
    assert second_receipt.commitment_digest == second.digest
    assert second_receipt != rows[0]
    assert calls == [first.digest, second.digest]


def test_entropy_corruption_partial_state_and_source_substitution_fail(tmp_path: Path) -> None:
    source = _h("entropy-source")
    commitment = _commitment("corrupt", source)
    root = tmp_path / "entropy"
    provider = capabilities.DurableSelectionEntropyProvider(
        root, source_digest=source, entropy_source=lambda *_args: b"e" * 32
    )
    provider(commitment, _teardown())
    path = root / f"entropy-{commitment.digest}.json"
    path.chmod(0o600)
    value = path.read_bytes().replace(b'"entropy_hex":"', b'"entropy_hex":"00')
    path.write_bytes(value)
    path.chmod(0o400)
    with pytest.raises(capabilities.B300QualificationCapabilityError):
        provider(commitment, _teardown())

    partial_root = tmp_path / "partial-entropy"
    partial = capabilities.DurableSelectionEntropyProvider(
        partial_root, source_digest=source, entropy_source=lambda *_args: b"e" * 32
    )
    other = _commitment("partial", source)
    (partial_root / f".entropy-{other.digest}.json.partial").write_bytes(b"partial")
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="partial"):
        partial(other, _teardown())

    wrong_source = capabilities.DurableSelectionEntropyProvider(
        tmp_path / "wrong-source",
        source_digest=_h("other-source"),
        entropy_source=lambda *_args: b"e" * 32,
    )
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="another"):
        wrong_source(commitment, _teardown())


def test_closed_resolver_binds_both_source_kinds_and_detects_stale_bytes(
    tmp_path: Path,
) -> None:
    proposal, integrated = tmp_path / "proposal", tmp_path / "integrated"
    proposal.mkdir()
    integrated.mkdir()
    (proposal / "manifest.toml").write_text("proposal = 1\n")
    (integrated / "manifest.toml").write_text("integrated = 1\n")
    proposal_digest, integrated_digest = _h("proposal"), _h("integrated")
    resolver = capabilities.ClosedContributionSourceResolver(
        {proposal_digest: proposal}, {integrated_digest: integrated}
    )
    reopened = capabilities.ClosedContributionSourceResolver(
        {proposal_digest: proposal}, {integrated_digest: integrated}
    )

    assert resolver.resolve_proposal(proposal_digest) == proposal
    assert resolver.resolve_integrated(integrated_digest) == integrated
    assert resolver.digest == reopened.digest
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="closed"):
        resolver.resolve_proposal(_h("missing"))

    (proposal / "manifest.toml").write_text("proposal = 2\n")
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="changed"):
        resolver.resolve_proposal(proposal_digest)


def test_empty_resolver_is_the_genesis_authority_and_fails_closed(tmp_path: Path) -> None:
    empty = capabilities.ClosedContributionSourceResolver({}, {})
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="closed"):
        empty.resolve_integrated(_h("incumbent"))
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="closed"):
        empty.resolve_proposal(_h("incumbent"))
    assert empty.digest == capabilities.ClosedContributionSourceResolver({}, {}).digest

    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="ambiguous"):
        capabilities.ClosedContributionSourceResolver(
            {_h("same"): source}, {_h("same"): source}
        )


def _shape(
    seed: str,
    *,
    applicable: bool = True,
    replay_count: int = 3,
    replay_passed: bool = True,
    complete: bool = True,
    candidate_failure: bool = False,
) -> capabilities.StructuredGraphShapeRecord:
    return capabilities.StructuredGraphShapeRecord(
        _h(seed),
        applicable,
        applicable,
        applicable,
        replay_count if applicable else 0,
        replay_passed if applicable else False,
        complete,
        candidate_failure,
    )


def _variant(
    slot: str,
    seed: str,
    *,
    domain_complete: bool = True,
    shapes: tuple[capabilities.StructuredGraphShapeRecord, ...] | None = None,
) -> capabilities.StructuredGraphVariantRecord:
    values = shapes or tuple(sorted((_shape(seed + ":a"), _shape(seed + ":b")), key=lambda row: row.descriptor_digest))
    return capabilities.StructuredGraphVariantRecord(
        slot, "default", True, domain_complete, values
    )


def test_structured_graph_conversion_is_generic_for_two_target_profiles() -> None:
    singleton = _variant("attention.block_score", "singleton")
    collective = _variant("collective.all_reduce", "collective")
    records = tuple(sorted((singleton, collective), key=lambda row: (row.slot_id, row.variant_id)))
    facts = capabilities.structured_focused_graph_facts(3, records)

    assert facts.expected_graph_replays == 3
    assert tuple(row.slot_id for row in facts.variants) == tuple(
        row.slot_id for row in records
    )
    assert tuple(row.slot_id for row in facts.observations) == tuple(
        row.slot_id for row in records
    )
    assert all(row.domain_coverage_complete for row in facts.observations)


def test_graph_converter_rejects_incomplete_ambiguous_or_aggregate_inputs() -> None:
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="domain"):
        capabilities.structured_focused_graph_facts(
            3, (_variant("attention.block_score", "domain", domain_complete=False),)
        )

    partial = _shape("partial", complete=False)
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="partial"):
        capabilities.structured_focused_graph_facts(
            3, (_variant("attention.block_score", "partial", shapes=(partial,)),)
        )

    short = _shape("short", replay_count=2)
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="coverage"):
        capabilities.structured_focused_graph_facts(
            3, (_variant("attention.block_score", "short", shapes=(short,)),)
        )

    infrastructure = _shape(
        "infrastructure",
        replay_count=1,
        replay_passed=False,
        candidate_failure=False,
    )
    with pytest.raises(capabilities.B300QualificationCapabilityError, match="infrastructure"):
        capabilities.structured_focused_graph_facts(
            3,
            (_variant("attention.block_score", "infra", shapes=(infrastructure,)),),
        )

    with pytest.raises(capabilities.B300QualificationCapabilityError, match="exact"):
        capabilities.structured_focused_graph_facts(3, True)  # type: ignore[arg-type]


def test_graph_converter_preserves_complete_candidate_failure_without_grading() -> None:
    failed = capabilities.StructuredGraphShapeRecord(
        _h("candidate-capture-failure"),
        True,
        True,
        False,
        0,
        False,
        True,
        True,
    )
    variant = _variant("collective.all_reduce", "failure", shapes=(failed,))
    facts = capabilities.structured_focused_graph_facts(3, (variant,))
    observed = facts.observations[0].shapes[0]

    assert observed.capture_succeeded is False
    assert observed.failure_kind == "capture"
    assert not hasattr(facts, "decision")


def test_sealed_commitment_names_the_declared_entropy_provider(tmp_path: Path) -> None:
    """The cross-module contract that field mirroring in fixtures cannot prove.

    Production seals commitments with the declared (domain-separated) entropy
    identity while the deployed capability declares the same identity for its
    durable provider; both derive it from one selection policy digest.  A raw
    policy digest on either side must fail closed before entropy is minted.
    """

    from cacheon.eval.qualification import declared_qualification_entropy_digest

    policy = _h("selection-policy")
    declared = declared_qualification_entropy_digest(policy)
    commitment = _commitment("declared", declared)
    provider = capabilities.DurableSelectionEntropyProvider(
        tmp_path / "entropy",
        source_digest=declared,
        entropy_source=lambda _commitment, _teardown: b"e" * 32,
    )

    receipt = provider(commitment, _teardown())

    assert receipt.source_digest == declared
    raw_commitment = _commitment("raw", policy)
    with pytest.raises(
        capabilities.B300QualificationCapabilityError,
        match="names another entropy source",
    ):
        provider(raw_commitment, _teardown())
