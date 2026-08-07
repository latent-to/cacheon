"""Contracts for the generic, path-free sealed B300 graph-facts registry."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from cacheon.arena_service import ArenaCandidateBinding
from cacheon.bundle_hash import content_hash
from cacheon.chain.publication import publish_worker_bundle
from cacheon.eval.b300_registered_qualification_inputs import B300FocusedGraphFacts
from cacheon.eval.b300_sealed_graph_facts import (
    REGISTRY_POLICY_VERSION,
    B300SealedGraphFactsEntry,
    B300SealedGraphFactsError,
    B300SealedGraphFactsIdentity,
    B300SealedGraphFactsRegistry,
    focused_graph_facts_digest,
)
from cacheon.eval.marginal_runtime import PreparedCandidateRuntime
from cacheon.eval.qualification import GraphVariantRequirement
from cacheon.eval.qualification_intake import (
    GraphShapeObservation,
    GraphVariantObservation,
    QualificationReservation,
)
from tests.test_marginal_runtime import FUSED, SILU, _case, _prepared


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass(frozen=True)
class _Profile:
    candidate: ArenaCandidateBinding
    prepared: PreparedCandidateRuntime
    facts: B300FocusedGraphFacts


def _facts(members: tuple[str, ...], label: str, *, replays: int = 3):
    requirements = []
    observations = []
    for member in members:
        descriptors = tuple(
            sorted((_h(f"{label}:{member}:shape-a"), _h(f"{label}:{member}:shape-b")))
        )
        requirements.append(
            GraphVariantRequirement(
                member,
                "commissioned",
                descriptors,
                True,
                descriptors,
            )
        )
        observations.append(
            GraphVariantObservation(
                member,
                "commissioned",
                True,
                True,
                tuple(
                    GraphShapeObservation(
                        descriptor,
                        True,
                        True,
                        True,
                        replays,
                        True,
                    )
                    for descriptor in descriptors
                ),
            )
        )
    return B300FocusedGraphFacts(
        replays,
        tuple(requirements),
        tuple(observations),
    )


def _profile(root: Path, source: Path, label: str) -> _Profile:
    root.mkdir()
    runtime = root / "runtime"
    runtime.mkdir()
    case = _case(runtime, source, suffix=label)
    prepared = _prepared(case).candidates[0]
    private_source = root / "private-source"
    shutil.copytree(source, private_source)
    for path in sorted(private_source.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    private_source.chmod(0o700)
    assert content_hash(private_source) == content_hash(source)
    publications = root / "publications"
    publications.mkdir(mode=0o700)
    publication = publish_worker_bundle(
        private_source,
        publications,
        content_hash(private_source),
    )
    target_id = case.arm.transition.target_id
    reservation = QualificationReservation(
        _h(f"{label}:reservation"),
        publication.digest,
        target_id,
        case.arm.selected_delta_digest,
        0,
        f"{label}-hotkey",
        100,
        2,
        0,
        case.catalog.require(target_id).members,
    )
    candidate = ArenaCandidateBinding(reservation, publication, 1)
    return _Profile(
        candidate,
        prepared,
        _facts(reservation.target_members, label),
    )


@pytest.fixture(scope="module")
def profiles(tmp_path_factory: pytest.TempPathFactory) -> tuple[_Profile, _Profile]:
    tmp_path = tmp_path_factory.mktemp("sealed-graph-facts")
    return (
        _profile(tmp_path / "noncollective", SILU, "noncollective"),
        _profile(tmp_path / "collective", FUSED, "collective"),
    )


def _entry(profile: _Profile, label: str) -> B300SealedGraphFactsEntry:
    return B300SealedGraphFactsEntry.seal(
        profile.candidate,
        profile.prepared,
        profile.facts,
        raw_evidence_digest=_h(f"{label}:raw-evidence"),
    )


def test_registry_resolves_arbitrary_noncollective_and_collective_profiles(
    profiles: tuple[_Profile, _Profile],
) -> None:
    noncollective, collective = profiles
    assert len(noncollective.candidate.reservation.target_members) == 1
    assert len(collective.candidate.reservation.target_members) == 2
    first = _entry(noncollective, "noncollective")
    second = _entry(collective, "collective")
    reverse = B300SealedGraphFactsRegistry(
        _h("verification-policy"),
        (second, first),
    )
    forward = B300SealedGraphFactsRegistry(
        _h("verification-policy"),
        (first, second),
    )

    assert (
        reverse(noncollective.candidate, noncollective.prepared)
        is noncollective.facts
    )
    assert reverse(collective.candidate, collective.prepared) is collective.facts
    assert reverse.entries == tuple(
        sorted((first, second), key=lambda row: row.identity.digest)
    )
    assert reverse.digest == forward.digest
    assert reverse.digest != B300SealedGraphFactsRegistry(
        _h("other-verification-policy"),
        (first, second),
    ).digest
    assert first.facts_digest == focused_graph_facts_digest(noncollective.facts)
    assert first.facts_digest != second.facts_digest
    assert reverse.to_dict()["policy_version"] == REGISTRY_POLICY_VERSION


@pytest.mark.parametrize(
    "field",
    (
        "candidate_binding_digest",
        "reservation_digest",
        "target_id",
        "selected_delta_digest",
        "publication_content_hash",
        "publication_digest",
        "publication_receipt_digest",
        "target_spec_digest",
        "prepared_contribution_digest",
        "prepared_arm_digest",
        "prepared_launch_digest",
        "materialized_stack_digest",
        "materialized_tree_digest",
    ),
)
def test_every_path_free_identity_drift_fails_closed(
    profiles: tuple[_Profile, _Profile],
    field: str,
) -> None:
    profile, _ = profiles
    entry = _entry(profile, "base")
    value = "another-target" if field == "target_id" else _h(f"stale:{field}")
    stale_identity = replace(entry.identity, **{field: value})
    stale = replace(entry, identity=stale_identity)
    registry = B300SealedGraphFactsRegistry(_h("policy"), (stale,))

    with pytest.raises(B300SealedGraphFactsError, match="no sealed graph facts"):
        registry(profile.candidate, profile.prepared)


def test_unknown_cross_target_cross_publication_and_cross_runtime_fail_closed(
    profiles: tuple[_Profile, _Profile],
) -> None:
    noncollective, collective = profiles
    registry = B300SealedGraphFactsRegistry(
        _h("policy"),
        (_entry(noncollective, "noncollective"),),
    )

    with pytest.raises(B300SealedGraphFactsError, match="no sealed graph facts"):
        registry(collective.candidate, collective.prepared)
    with pytest.raises(B300SealedGraphFactsError, match="do not form one exact"):
        registry(noncollective.candidate, collective.prepared)

    foreign_publication = collective.candidate.publication
    reservation = replace(
        noncollective.candidate.reservation,
        submission_digest=foreign_publication.digest,
    )
    cross_publication = ArenaCandidateBinding(reservation, foreign_publication, 1)
    with pytest.raises(B300SealedGraphFactsError, match="do not form one exact"):
        registry(cross_publication, noncollective.prepared)


def test_registry_rejects_duplicate_and_ambiguous_lookup_keys(
    profiles: tuple[_Profile, _Profile],
) -> None:
    profile, _ = profiles
    entry = _entry(profile, "base")
    with pytest.raises(B300SealedGraphFactsError, match="duplicate identity"):
        B300SealedGraphFactsRegistry(_h("policy"), (entry, entry))

    other_raw = replace(entry, raw_evidence_digest=_h("different raw evidence"))
    with pytest.raises(B300SealedGraphFactsError, match="ambiguous identity"):
        B300SealedGraphFactsRegistry(_h("policy"), (entry, other_raw))

    changed_facts = _facts(
        profile.candidate.reservation.target_members,
        "changed-facts",
        replays=4,
    )
    other_facts = replace(entry, facts=changed_facts)
    with pytest.raises(B300SealedGraphFactsError, match="ambiguous identity"):
        B300SealedGraphFactsRegistry(_h("policy"), (entry, other_facts))


@pytest.mark.parametrize(
    "field",
    (
        "candidate_binding_digest",
        "reservation_digest",
        "selected_delta_digest",
        "publication_content_hash",
        "publication_digest",
        "publication_receipt_digest",
        "target_spec_digest",
        "prepared_contribution_digest",
        "prepared_arm_digest",
        "prepared_launch_digest",
        "materialized_stack_digest",
        "materialized_tree_digest",
    ),
)
def test_identity_rejects_every_malformed_digest(
    profiles: tuple[_Profile, _Profile],
    field: str,
) -> None:
    identity = _entry(profiles[0], "base").identity
    with pytest.raises(B300SealedGraphFactsError, match="lowercase 64-hex"):
        replace(identity, **{field: "not-a-digest"})


def test_entry_and_registry_reject_untyped_or_incomplete_inputs(
    profiles: tuple[_Profile, _Profile],
) -> None:
    profile, collective = profiles
    entry = _entry(profile, "base")
    collective_entry = _entry(collective, "collective")
    with pytest.raises(B300SealedGraphFactsError, match="target members"):
        replace(
            collective_entry.identity,
            target_members=tuple(reversed(collective_entry.identity.target_members)),
        )
    with pytest.raises(B300SealedGraphFactsError, match="candidate target members"):
        replace(
            entry,
            identity=replace(entry.identity, target_members=("foreign.member",)),
        )
    with pytest.raises(B300SealedGraphFactsError, match="raw_evidence_digest"):
        replace(entry, raw_evidence_digest="bad")
    with pytest.raises(B300SealedGraphFactsError, match="not exactly typed"):
        B300SealedGraphFactsEntry(  # type: ignore[arg-type]
            object(), entry.facts, _h("raw")
        )
    with pytest.raises(B300SealedGraphFactsError, match="not exactly typed"):
        B300SealedGraphFactsEntry(  # type: ignore[arg-type]
            entry.identity, object(), _h("raw")
        )
    with pytest.raises(B300SealedGraphFactsError, match="nonempty exact tuple"):
        B300SealedGraphFactsRegistry(_h("policy"), ())
    with pytest.raises(B300SealedGraphFactsError, match="nonempty exact tuple"):
        B300SealedGraphFactsRegistry(_h("policy"), [entry])  # type: ignore[arg-type]
    with pytest.raises(B300SealedGraphFactsError, match="unsupported"):
        B300SealedGraphFactsRegistry(
            _h("policy"),
            (entry,),
            policy_version="future-policy",
        )
    with pytest.raises(B300SealedGraphFactsError, match="unsupported"):
        B300SealedGraphFactsRegistry(_h("policy"), (entry,), schema_version=2)
    with pytest.raises(B300SealedGraphFactsError, match="verification_policy_digest"):
        B300SealedGraphFactsRegistry("bad", (entry,))


def test_lookup_requires_exact_public_types_and_registered_marginal_arm(
    profiles: tuple[_Profile, _Profile],
) -> None:
    profile, _ = profiles
    registry = B300SealedGraphFactsRegistry(
        _h("policy"),
        (_entry(profile, "base"),),
    )
    with pytest.raises(B300SealedGraphFactsError, match="ArenaCandidateBinding"):
        registry(object(), profile.prepared)  # type: ignore[arg-type]
    with pytest.raises(B300SealedGraphFactsError, match="PreparedCandidateRuntime"):
        registry(profile.candidate, object())  # type: ignore[arg-type]

    forged = object.__new__(PreparedCandidateRuntime)
    object.__setattr__(forged, "arm", object())
    with pytest.raises(B300SealedGraphFactsError, match="marginal arm"):
        registry(profile.candidate, forged)


def test_facts_digest_binds_each_exact_observation_field(
    profiles: tuple[_Profile, _Profile],
) -> None:
    profile, _ = profiles
    baseline = profile.facts
    observed = baseline.observations[0]
    shape = observed.shapes[0]
    changed_shape = replace(shape, replay_count=shape.replay_count + 1)
    changed_observed = replace(
        observed,
        shapes=(changed_shape, *observed.shapes[1:]),
    )
    changed = B300FocusedGraphFacts(
        baseline.expected_graph_replays,
        baseline.variants,
        (changed_observed, *baseline.observations[1:]),
    )

    assert focused_graph_facts_digest(baseline) != focused_graph_facts_digest(changed)
    assert "root" not in B300SealedGraphFactsEntry.seal(
        profile.candidate,
        profile.prepared,
        baseline,
        raw_evidence_digest=_h("raw"),
    ).to_dict()["identity"]


def test_identity_type_is_exact_at_entry_boundary(
    profiles: tuple[_Profile, _Profile],
) -> None:
    entry = _entry(profiles[0], "base")

    class _IdentitySubclass(B300SealedGraphFactsIdentity):
        pass

    subclass = _IdentitySubclass(**entry.identity.__dict__)
    with pytest.raises(B300SealedGraphFactsError, match="not exactly typed"):
        replace(entry, identity=subclass)
