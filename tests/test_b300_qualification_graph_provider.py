"""Contracts for dynamic, commissioned B300 qualification graph facts."""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from cacheon.arena_service import ArenaCandidateBinding
from cacheon.bundle_hash import content_hash
from cacheon.chain.publication import publish_worker_bundle
from cacheon.eval.b300_qualification_capabilities import (
    StructuredGraphShapeRecord,
    StructuredGraphVariantRecord,
)
from cacheon.eval.b300_qualification_graph_provider import (
    ARTIFACT_DOMAIN,
    ARTIFACT_MEDIA_TYPE,
    ARTIFACT_SCHEMA,
    BUILDER_SCHEMA_VERSION,
    B300QualificationGraphArtifact,
    B300QualificationGraphBinding,
    B300QualificationGraphFactsBuilder,
    B300QualificationGraphProviderError,
)
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.marginal_runtime import PreparedCandidateRuntime
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.stack_identity import canonical_json_bytes
from tests.test_marginal_runtime import FUSED, SILU, _case, _prepared


POLICY = hashlib.sha256(b"commissioned graph verification policy").hexdigest()
PROBE_AUTHORITY = hashlib.sha256(b"commissioned graph probe authority").hexdigest()
REOPENER_AUTHORITY = hashlib.sha256(b"commissioned evidence reopener").hexdigest()


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass(frozen=True)
class _Profile:
    candidate: ArenaCandidateBinding
    prepared: PreparedCandidateRuntime


def _profile(root: Path, source: Path, label: str) -> _Profile:
    root.mkdir()
    runtime = root / "runtime"
    runtime.mkdir()
    case = _case(runtime, source, suffix="-" + label)
    prepared = _prepared(case).candidates[0]

    private_source = root / "private-source"
    shutil.copytree(source, private_source)
    for path in sorted(private_source.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    private_source.chmod(0o700)
    assert content_hash(private_source) == content_hash(source)
    publication_root = root / "publications"
    publication_root.mkdir(mode=0o700)
    publication = publish_worker_bundle(
        private_source,
        publication_root,
        content_hash(private_source),
    )
    target_id = case.arm.transition.target_id
    reservation = QualificationReservation(
        _h(label + ":reservation"),
        publication.digest,
        target_id,
        case.arm.selected_delta_digest,
        0,
        label + "-hotkey",
        100,
        2,
        0,
        case.catalog.require(target_id).members,
    )
    return _Profile(
        ArenaCandidateBinding(reservation, publication, 1),
        prepared,
    )


@pytest.fixture(scope="module")
def profiles(tmp_path_factory: pytest.TempPathFactory) -> tuple[_Profile, _Profile]:
    root = tmp_path_factory.mktemp("qualification-graph-provider")
    # FUSED is the existing two-member fixture profile containing
    # collective.ar_residual_rmsnorm and collective.moe_finalize_ar_rmsnorm.
    return (
        _profile(root / "activation", SILU, "activation"),
        _profile(root / "collective", FUSED, "collective"),
    )


def _shape(
    label: str,
    *,
    expected_replays: int = 3,
    replay_count: int | None = None,
    replay_passed: bool = True,
    capture_succeeded: bool = True,
    observation_complete: bool = True,
    candidate_failure: bool = False,
) -> StructuredGraphShapeRecord:
    capture = capture_succeeded
    count = expected_replays if replay_count is None else replay_count
    if not capture:
        count = 0
        replay_passed = False
    return StructuredGraphShapeRecord(
        _h(label),
        True,
        True,
        capture,
        count,
        replay_passed,
        observation_complete,
        candidate_failure,
    )


def _records(
    binding: B300QualificationGraphBinding,
    *,
    expected_replays: int = 3,
    domain_complete: bool = True,
    shape_factory=None,
) -> tuple[StructuredGraphVariantRecord, ...]:
    rows = []
    for member in binding.target_members:
        shapes = (
            tuple(
                sorted(
                    (
                        _shape(
                            member + ":a",
                            expected_replays=expected_replays,
                        ),
                        _shape(
                            member + ":b",
                            expected_replays=expected_replays,
                        ),
                    ),
                    key=lambda row: row.descriptor_digest,
                )
            )
            if shape_factory is None
            else shape_factory(member)
        )
        rows.append(
            StructuredGraphVariantRecord(
                member,
                "commissioned",
                True,
                domain_complete,
                shapes,
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.slot_id, row.variant_id)))


def _artifact(
    binding: B300QualificationGraphBinding,
    *,
    policy: str = POLICY,
    expected_replays: int = 3,
    records: tuple[StructuredGraphVariantRecord, ...] | None = None,
) -> B300QualificationGraphArtifact:
    return B300QualificationGraphArtifact(
        binding,
        policy,
        expected_replays,
        _records(binding, expected_replays=expected_replays)
        if records is None
        else records,
    )


def _reference(
    payload: bytes,
    *,
    domain: str = ARTIFACT_DOMAIN,
    media_type: str = ARTIFACT_MEDIA_TYPE,
    schema: str = ARTIFACT_SCHEMA,
) -> EvidenceArtifactRef:
    return EvidenceArtifactRef(
        domain,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        media_type,
        schema,
    )


class _CommissionedAuthority:
    """In-memory test double for separately commissioned probe/reopen seams."""

    def __init__(self) -> None:
        self.references: dict[str, EvidenceArtifactRef] = {}
        self.payloads: dict[EvidenceArtifactRef, object] = {}
        self.probe_sequences: dict[str, list[object]] = {}
        self.probe_calls: list[
            tuple[
                B300QualificationGraphBinding,
                ArenaCandidateBinding,
                PreparedCandidateRuntime,
            ]
        ] = []

    def install(
        self,
        binding: B300QualificationGraphBinding,
        payload: bytes,
        *,
        reference: EvidenceArtifactRef | None = None,
        reopened: object | None = None,
    ) -> EvidenceArtifactRef:
        ref = _reference(payload) if reference is None else reference
        self.references[binding.digest] = ref
        self.payloads[ref] = payload if reopened is None else reopened
        return ref

    def commission(
        self,
        profile: _Profile,
        *,
        artifact: B300QualificationGraphArtifact | None = None,
    ) -> B300QualificationGraphArtifact:
        binding = B300QualificationGraphBinding.derive(
            profile.candidate,
            profile.prepared,
        )
        value = _artifact(binding) if artifact is None else artifact
        self.install(binding, value.canonical_bytes)
        return value

    def probe(
        self,
        binding: B300QualificationGraphBinding,
        candidate: ArenaCandidateBinding,
        prepared: PreparedCandidateRuntime,
    ) -> EvidenceArtifactRef:
        assert type(binding) is B300QualificationGraphBinding
        assert type(candidate) is ArenaCandidateBinding
        assert type(prepared) is PreparedCandidateRuntime
        assert B300QualificationGraphBinding.derive(candidate, prepared) == binding
        self.probe_calls.append((binding, candidate, prepared))
        sequence = self.probe_sequences.get(binding.digest)
        value = sequence.pop(0) if sequence else self.references[binding.digest]
        return value  # type: ignore[return-value]

    def reopen(self, reference: EvidenceArtifactRef) -> bytes:
        assert type(reference) is EvidenceArtifactRef
        return self.payloads[reference]  # type: ignore[return-value]


def _builder(
    authority: _CommissionedAuthority,
    *,
    policy: str = POLICY,
    probe_digest: str = PROBE_AUTHORITY,
    reopener_digest: str = REOPENER_AUTHORITY,
) -> B300QualificationGraphFactsBuilder:
    return B300QualificationGraphFactsBuilder(
        policy,
        probe_digest,
        reopener_digest,
        authority.probe,
        authority.reopen,
    )


def test_future_activation_and_collective_candidates_use_one_stable_builder(
    profiles: tuple[_Profile, _Profile],
) -> None:
    activation, collective = profiles
    authority = _CommissionedAuthority()
    builder = _builder(authority)
    initial_digest = builder.digest

    authority.commission(activation)
    activation_facts = builder(activation.candidate, activation.prepared)
    after_activation = builder.digest
    authority.commission(collective)
    collective_facts = builder(collective.candidate, collective.prepared)

    assert initial_digest == after_activation == builder.digest
    assert activation.candidate.reservation.target_id == "activation.silu_and_mul"
    assert "collective.ar_residual_rmsnorm" in collective.candidate.reservation.target_members
    assert tuple(row.slot_id for row in activation_facts.variants) == (
        activation.candidate.reservation.target_members
    )
    assert tuple(row.slot_id for row in collective_facts.variants) == (
        collective.candidate.reservation.target_members
    )
    assert [row[1] for row in authority.probe_calls] == [
        activation.candidate,
        collective.candidate,
    ]


def test_binding_is_complete_path_free_and_rejects_crossed_inputs(
    profiles: tuple[_Profile, _Profile],
) -> None:
    activation, collective = profiles
    binding = B300QualificationGraphBinding.derive(
        activation.candidate,
        activation.prepared,
    )
    value = binding.to_dict()

    assert B300QualificationGraphBinding.from_dict(value) == binding
    assert set(value) == {
        "candidate_binding_digest",
        "materialized_stack_digest",
        "materialized_tree_digest",
        "native_build_spec_digest",
        "prepared_arm_digest",
        "prepared_contribution_digest",
        "prepared_launch_digest",
        "publication_address_digest",
        "publication_content_hash",
        "publication_digest",
        "publication_receipt_digest",
        "reservation_digest",
        "reservation_identity_digest",
        "screen_attempt",
        "selected_delta_digest",
        "target_id",
        "target_members",
        "target_spec_digest",
        "trusted_tree_identity_digest",
    }
    encoded = json.dumps(value, sort_keys=True)
    assert str(activation.candidate.publication.root) not in encoded
    assert str(activation.prepared.binding.tree.root) not in encoded
    with pytest.raises(B300QualificationGraphProviderError, match="one exact"):
        B300QualificationGraphBinding.derive(
            activation.candidate,
            collective.prepared,
        )
    with pytest.raises(B300QualificationGraphProviderError, match="one exact"):
        B300QualificationGraphBinding.derive(
            collective.candidate,
            activation.prepared,
        )


def test_builder_digest_binds_only_schema_policy_and_commissioned_authorities() -> None:
    first = _CommissionedAuthority()
    second = _CommissionedAuthority()
    baseline = _builder(first)

    assert baseline.digest == _builder(second).digest
    assert baseline.digest != _builder(first, policy=_h("other-policy")).digest
    assert baseline.digest != _builder(first, probe_digest=_h("other-probe")).digest
    assert baseline.digest != _builder(first, reopener_digest=_h("other-reopener")).digest
    assert baseline.to_dict()["schema_version"] == BUILDER_SCHEMA_VERSION
    assert baseline.to_dict()["artifact_schema"] == ARTIFACT_SCHEMA
    with pytest.raises(B300QualificationGraphProviderError, match="schema"):
        B300QualificationGraphFactsBuilder(
            POLICY,
            PROBE_AUTHORITY,
            REOPENER_AUTHORITY,
            first.probe,
            first.reopen,
            schema_version=2,
        )


def test_exact_lookup_is_idempotent_and_ignores_unrelated_session_timing(
    profiles: tuple[_Profile, _Profile],
) -> None:
    profile = profiles[0]
    authority = _CommissionedAuthority()
    artifact = authority.commission(profile)
    builder = _builder(authority)
    first = builder(profile.candidate, profile.prepared)
    second = builder(profile.candidate, profile.prepared)

    plan = profile.prepared.session_plan
    changed_plan = replace(
        plan,
        prompt_batches=(plan.prompt_batches[0], ("extra warmup",), *plan.prompt_batches[1:]),
        warmup_count=2,
    )
    changed_prepared = replace(profile.prepared, session_plan=changed_plan)
    assert B300QualificationGraphBinding.derive(
        profile.candidate,
        changed_prepared,
    ) == B300QualificationGraphBinding.derive(
        profile.candidate,
        profile.prepared,
    )
    third = builder(profile.candidate, changed_prepared)

    assert first is second is third
    serialized = json.dumps(
        {
            "artifact": artifact.to_dict(),
            "builder": builder.to_dict(),
        },
        sort_keys=True,
    )
    assert all(
        name not in serialized
        for name in ("elapsed", "latency", "speed", "throughput", "warmup_count")
    )


@pytest.mark.parametrize(
    "field,replacement_value",
    (
        ("reservation_digest", _h("stale-reservation")),
        ("reservation_identity_digest", _h("stale-reservation-identity")),
        ("candidate_binding_digest", _h("stale-candidate")),
        ("screen_attempt", 2),
        ("target_id", "foreign.target"),
        ("target_spec_digest", _h("stale-target-spec")),
        ("selected_delta_digest", _h("stale-selected-delta")),
        ("publication_content_hash", _h("stale-content")),
        ("publication_address_digest", _h("stale-address")),
        ("publication_digest", _h("stale-publication")),
        ("publication_receipt_digest", _h("stale-publication-receipt")),
        ("prepared_arm_digest", _h("stale-arm")),
        ("prepared_contribution_digest", _h("stale-contribution")),
        ("prepared_launch_digest", _h("stale-launch")),
        ("materialized_stack_digest", _h("stale-stack")),
        ("materialized_tree_digest", _h("stale-tree")),
        ("trusted_tree_identity_digest", _h("stale-trusted-tree")),
        ("native_build_spec_digest", _h("stale-native-build")),
    ),
)
def test_every_artifact_binding_drift_fails_closed(
    profiles: tuple[_Profile, _Profile],
    field: str,
    replacement_value: object,
) -> None:
    profile = profiles[0]
    exact = B300QualificationGraphBinding.derive(profile.candidate, profile.prepared)
    stale = replace(exact, **{field: replacement_value})
    artifact = _artifact(stale)
    authority = _CommissionedAuthority()
    authority.install(exact, artifact.canonical_bytes)

    with pytest.raises(B300QualificationGraphProviderError, match="binding differs"):
        _builder(authority)(profile.candidate, profile.prepared)


def test_attempt_target_member_and_policy_mismatch_fail_closed(
    profiles: tuple[_Profile, _Profile],
) -> None:
    profile = profiles[0]
    binding = B300QualificationGraphBinding.derive(profile.candidate, profile.prepared)

    changed_attempt = replace(profile.candidate, screen_attempt=2)
    authority = _CommissionedAuthority()
    authority.install(
        B300QualificationGraphBinding.derive(changed_attempt, profile.prepared),
        _artifact(binding).canonical_bytes,
    )
    with pytest.raises(B300QualificationGraphProviderError, match="binding differs"):
        _builder(authority)(changed_attempt, profile.prepared)

    with pytest.raises(B300QualificationGraphProviderError, match="member domain"):
        B300QualificationGraphArtifact(
            binding,
            POLICY,
            3,
            (
                StructuredGraphVariantRecord(
                    "foreign.member",
                    "commissioned",
                    True,
                    True,
                    (_shape("foreign"),),
                ),
            ),
        )

    wrong_policy = _artifact(binding, policy=_h("wrong-policy"))
    authority = _CommissionedAuthority()
    authority.install(binding, wrong_policy.canonical_bytes)
    with pytest.raises(B300QualificationGraphProviderError, match="policy differs"):
        _builder(authority)(profile.candidate, profile.prepared)


def test_closed_artifact_and_content_address_fail_closed(
    profiles: tuple[_Profile, _Profile],
) -> None:
    profile = profiles[0]
    binding = B300QualificationGraphBinding.derive(profile.candidate, profile.prepared)
    canonical = _artifact(binding).canonical_bytes

    missing = _CommissionedAuthority()
    with pytest.raises(B300QualificationGraphProviderError, match="probe failed"):
        _builder(missing)(profile.candidate, profile.prepared)

    wrong_type = _CommissionedAuthority()
    wrong_type.probe_sequences[binding.digest] = [object()]
    with pytest.raises(B300QualificationGraphProviderError, match="EvidenceArtifactRef"):
        _builder(wrong_type)(profile.candidate, profile.prepared)

    wrong_domain = _CommissionedAuthority()
    wrong_ref = _reference(canonical, domain="cacheon.eval.foreign-graph")
    wrong_domain.install(binding, canonical, reference=wrong_ref)
    with pytest.raises(B300QualificationGraphProviderError, match="closed schema"):
        _builder(wrong_domain)(profile.candidate, profile.prepared)

    wrong_bytes = _CommissionedAuthority()
    wrong_bytes.install(binding, canonical, reopened=canonical + b"\n")
    with pytest.raises(B300QualificationGraphProviderError, match="content-addressed"):
        _builder(wrong_bytes)(profile.candidate, profile.prepared)

    noncanonical = canonical + b"\n"
    noncanonical_authority = _CommissionedAuthority()
    noncanonical_authority.install(binding, noncanonical)
    with pytest.raises(B300QualificationGraphProviderError, match="canonical JSON"):
        _builder(noncanonical_authority)(profile.candidate, profile.prepared)

    extra = json.loads(canonical)
    extra["throughput"] = 123
    extra_bytes = canonical_json_bytes(extra)
    extra_authority = _CommissionedAuthority()
    extra_authority.install(binding, extra_bytes)
    with pytest.raises(B300QualificationGraphProviderError, match="closed schema"):
        _builder(extra_authority)(profile.candidate, profile.prepared)


@pytest.mark.parametrize(
    "records_factory,match",
    (
        (
            lambda binding: _records(binding, domain_complete=False),
            "incomplete graph-domain",
        ),
        (
            lambda binding: _records(
                binding,
                shape_factory=lambda member: (
                    _shape(member + ":partial", observation_complete=False),
                ),
            ),
            "partial graph-shape",
        ),
        (
            lambda binding: _records(
                binding,
                shape_factory=lambda member: (
                    _shape(member + ":short", replay_count=2),
                ),
            ),
            "replay coverage",
        ),
        (
            lambda binding: _records(
                binding,
                shape_factory=lambda member: (
                    _shape(
                        member + ":infrastructure",
                        capture_succeeded=False,
                        candidate_failure=False,
                    ),
                ),
            ),
            "infrastructure-scoped",
        ),
        (
            lambda binding: (
                *_records(binding),
                _records(binding)[0],
            ),
            "canonical and complete",
        ),
    ),
)
def test_converter_rejects_partial_short_ambiguous_or_infrastructure_evidence(
    profiles: tuple[_Profile, _Profile],
    records_factory,
    match: str,
) -> None:
    profile = profiles[0]
    binding = B300QualificationGraphBinding.derive(profile.candidate, profile.prepared)
    artifact = _artifact(binding, records=records_factory(binding))
    authority = _CommissionedAuthority()
    authority.install(binding, artifact.canonical_bytes)

    with pytest.raises(B300QualificationGraphProviderError, match=match):
        _builder(authority)(profile.candidate, profile.prepared)


def test_complete_candidate_failure_remains_raw_facts_not_a_provider_grade(
    profiles: tuple[_Profile, _Profile],
) -> None:
    profile = profiles[0]
    binding = B300QualificationGraphBinding.derive(profile.candidate, profile.prepared)
    records = _records(
        binding,
        shape_factory=lambda member: (
            _shape(
                member + ":candidate-capture",
                capture_succeeded=False,
                candidate_failure=True,
            ),
        ),
    )
    artifact = _artifact(binding, records=records)
    authority = _CommissionedAuthority()
    authority.install(binding, artifact.canonical_bytes)
    facts = _builder(authority)(profile.candidate, profile.prepared)

    assert all(
        observation.shapes[0].failure_kind == "capture"
        for observation in facts.observations
    )
    assert not hasattr(facts, "decision")
    assert not hasattr(facts, "grade")


def test_same_binding_concurrency_is_serial_and_conflicting_evidence_fails_closed(
    profiles: tuple[_Profile, _Profile],
) -> None:
    profile = profiles[0]
    binding = B300QualificationGraphBinding.derive(profile.candidate, profile.prepared)
    first_payload = _artifact(binding).canonical_bytes
    authority = _CommissionedAuthority()
    first_ref = authority.install(binding, first_payload)
    builder = _builder(authority)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(builder, profile.candidate, profile.prepared)
            for _ in range(2)
        ]
    assert futures[0].result() is futures[1].result()

    changed_records = _records(binding, expected_replays=4)
    second_payload = _artifact(
        binding,
        expected_replays=4,
        records=changed_records,
    ).canonical_bytes
    second_ref = _reference(second_payload)
    authority.payloads[second_ref] = second_payload
    authority.probe_sequences[binding.digest] = [first_ref, second_ref]
    conflict_builder = _builder(authority)
    with ThreadPoolExecutor(max_workers=2) as pool:
        conflicting = [
            pool.submit(conflict_builder, profile.candidate, profile.prepared)
            for _ in range(2)
        ]
    outcomes = []
    for future in conflicting:
        try:
            outcomes.append(future.result())
        except B300QualificationGraphProviderError as exc:
            outcomes.append(exc)
    assert sum(not isinstance(row, Exception) for row in outcomes) == 1
    errors = [row for row in outcomes if isinstance(row, Exception)]
    assert len(errors) == 1
    assert "ambiguous evidence references" in str(errors[0])


def test_distinct_candidates_can_be_commissioned_concurrently(
    profiles: tuple[_Profile, _Profile],
) -> None:
    authority = _CommissionedAuthority()
    for profile in profiles:
        authority.commission(profile)
    builder = _builder(authority)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(builder, profile.candidate, profile.prepared)
            for profile in profiles
        ]
    facts = [future.result() for future in futures]

    assert [tuple(row.slot_id for row in value.variants) for value in facts] == [
        profile.candidate.reservation.target_members for profile in profiles
    ]
