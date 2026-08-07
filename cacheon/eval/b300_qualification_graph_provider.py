"""Commissioned, candidate-dynamic focused-graph facts for B300 qualification.

This module is an authority seam, not a device implementation.  A validator
commissions two candidate-independent callables: one executes or retrieves
correctness-only focused-graph evidence for the exact qualification binding,
and a separate authority reopens the returned content-addressed artifact.
Only canonical, closed, path-free evidence reaches the existing structured
graph-facts converter.

No screen result, timing result, throughput claim, or aggregate verdict is an
input to this builder.  Probe failures and infrastructure-attributed graph
failures raise an authority error; they are never converted into candidate
``FAIL`` facts.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from cacheon._strict import require_digest, require_identifier
from cacheon.arena_service import ArenaCandidateBinding
from cacheon.chain.publication import WorkerBundlePublication
from cacheon.engine_tree import EmittedFile, MaterializedEngineTree
from cacheon.eval.b300_qualification_capabilities import (
    B300QualificationCapabilityError,
    StructuredGraphShapeRecord,
    StructuredGraphVariantRecord,
    structured_focused_graph_facts,
)
from cacheon.eval.b300_registered_qualification_inputs import B300FocusedGraphFacts
from cacheon.eval.b300_qualification_graph_store_io import (
    B300QualificationGraphEvidenceHold,
)
from cacheon.eval.engine_launch import (
    EngineLaunchSpec,
    NativeBuildSpec,
    TrustedLaunchBinding,
    validate_native_build_spec,
)
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.marginal_runtime import MaterializedArmBinding, PreparedCandidateRuntime
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.stack_identity import canonical_digest, canonical_json_bytes
from cacheon.stack_manifest import ProposalContributionRef
from cacheon.stack_plan import MarginalArmPlan


BINDING_SCHEMA = "cacheon.eval.b300-qualification-graph-binding.v1"
ARTIFACT_SCHEMA = "cacheon.eval.b300-qualification-graph-evidence.v1"
BUILDER_SCHEMA = "cacheon.eval.b300-qualification-graph-builder.v1"
ARTIFACT_DOMAIN = "cacheon.eval.b300-qualification-graph"
ARTIFACT_MEDIA_TYPE = "application/json"
TRUSTED_TREE_DOMAIN = "cacheon.eval.b300-qualification-trusted-tree"
BUILDER_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")


class B300QualificationGraphProviderError(RuntimeError):
    """A graph probe, artifact, or lookup differs from commissioned authority."""


def _digest(value: object, field_name: str) -> str:
    return require_digest(
        value,
        field=field_name,
        error=B300QualificationGraphProviderError,
    )


def _identifier(value: object, field_name: str) -> str:
    return require_identifier(
        value,
        field=field_name,
        error=B300QualificationGraphProviderError,
        pattern=_IDENTIFIER,
    )


def _positive(value: object, field_name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise B300QualificationGraphProviderError(
            f"{field_name} must be an integer >= {minimum}"
        )
    return value


def _strict_object(
    value: object,
    fields: frozenset[str],
    field_name: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise B300QualificationGraphProviderError(
            f"{field_name} fields do not match the closed schema"
        )
    return value


def _canonical_object(payload: bytes) -> dict[str, object]:
    def reject_number(value: str) -> object:
        raise B300QualificationGraphProviderError(
            f"graph evidence contains unsupported number {value!r}"
        )

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise B300QualificationGraphProviderError(
                    f"graph evidence repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_float=reject_number,
            parse_constant=reject_number,
            object_pairs_hook=pairs,
        )
    except B300QualificationGraphProviderError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise B300QualificationGraphProviderError(
            f"graph evidence is malformed: {exc}"
        ) from None
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise B300QualificationGraphProviderError(
            "graph evidence is not exact canonical JSON"
        )
    return value


def _tree_identity_digest(tree: MaterializedEngineTree) -> str:
    if type(tree) is not MaterializedEngineTree:
        raise B300QualificationGraphProviderError(
            "prepared materialized tree is not exactly typed"
        )
    stack_digest = _digest(tree.stack_digest, "materialized stack digest")
    tree_digest = _digest(tree.tree_digest, "materialized tree digest")
    files = tree.files
    if (
        type(files) is not tuple
        or not files
        or any(type(row) is not EmittedFile for row in files)
        or tuple(row.path for row in files) != tuple(sorted({row.path for row in files}))
    ):
        raise B300QualificationGraphProviderError(
            "materialized tree file inventory is not exact and canonical"
        )
    identities: list[dict[str, object]] = []
    for row in files:
        path = row.path
        logical = PurePosixPath(path) if type(path) is str else None
        if (
            logical is None
            or not path
            or logical.is_absolute()
            or logical.as_posix() != path
            or any(part in {"", ".", ".."} for part in logical.parts)
            or type(row.mode) is not int
            or row.mode < 0
            or type(row.size) is not int
            or row.size < 0
        ):
            raise B300QualificationGraphProviderError(
                "materialized tree file identity is malformed"
            )
        identities.append(
            {
                "mode": row.mode,
                "path": path,
                "sha256": _digest(row.sha256, "materialized file sha256"),
                "size": row.size,
            }
        )
    manifest = tree.runtime_manifest
    if manifest is not None:
        logical_manifest = PurePosixPath(manifest) if type(manifest) is str else None
        if (
            logical_manifest is None
            or not manifest
            or logical_manifest.is_absolute()
            or logical_manifest.as_posix() != manifest
            or any(part in {"", ".", ".."} for part in logical_manifest.parts)
        ):
            raise B300QualificationGraphProviderError(
                "materialized runtime manifest identity is malformed"
            )
    return canonical_digest(
        TRUSTED_TREE_DOMAIN,
        {
            "files": identities,
            "runtime_manifest": manifest,
            "stack_digest": stack_digest,
            "tree_digest": tree_digest,
        },
    )


@dataclass(frozen=True)
class B300QualificationGraphBinding:
    """Exact path-free identity of one candidate and prepared qualification arm."""

    reservation_digest: str
    reservation_identity_digest: str
    candidate_binding_digest: str
    screen_attempt: int
    target_id: str
    target_members: tuple[str, ...]
    target_spec_digest: str
    selected_delta_digest: str
    publication_content_hash: str
    publication_address_digest: str
    publication_digest: str
    publication_receipt_digest: str
    prepared_arm_digest: str
    prepared_contribution_digest: str
    prepared_launch_digest: str
    materialized_stack_digest: str
    materialized_tree_digest: str
    trusted_tree_identity_digest: str
    native_build_spec_digest: str

    def __post_init__(self) -> None:
        for name in (
            "reservation_digest",
            "reservation_identity_digest",
            "candidate_binding_digest",
            "target_spec_digest",
            "selected_delta_digest",
            "publication_content_hash",
            "publication_address_digest",
            "publication_digest",
            "publication_receipt_digest",
            "prepared_arm_digest",
            "prepared_contribution_digest",
            "prepared_launch_digest",
            "materialized_stack_digest",
            "materialized_tree_digest",
            "trusted_tree_identity_digest",
            "native_build_spec_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self, "screen_attempt", _positive(self.screen_attempt, "screen attempt")
        )
        object.__setattr__(self, "target_id", _identifier(self.target_id, "target ID"))
        members = self.target_members
        if (
            type(members) is not tuple
            or not members
            or members != tuple(sorted(set(members)))
            or any(_identifier(row, "target member") != row for row in members)
        ):
            raise B300QualificationGraphProviderError(
                "target members must be one nonempty canonical tuple"
            )

    @classmethod
    def derive(
        cls,
        candidate: ArenaCandidateBinding,
        prepared: PreparedCandidateRuntime,
    ) -> "B300QualificationGraphBinding":
        """Derive and cross-check every correctness-relevant portable identity."""

        if type(candidate) is not ArenaCandidateBinding:
            raise B300QualificationGraphProviderError(
                "graph candidate must be an exact ArenaCandidateBinding"
            )
        if type(prepared) is not PreparedCandidateRuntime:
            raise B300QualificationGraphProviderError(
                "graph runtime must be an exact PreparedCandidateRuntime"
            )
        reservation = candidate.reservation
        publication = candidate.publication
        arm = prepared.arm
        binding = prepared.binding
        launch = prepared.launch
        session = prepared.session_plan
        if (
            type(reservation) is not QualificationReservation
            or type(publication) is not WorkerBundlePublication
            or type(arm) is not MarginalArmPlan
            or type(binding) is not MaterializedArmBinding
            or type(launch) is not EngineLaunchSpec
            or type(session) is not SessionExecutionPlan
        ):
            raise B300QualificationGraphProviderError(
                "candidate and prepared graph inputs are not exact qualification types"
            )
        tree = binding.tree
        trusted = binding.launch_binding
        if type(trusted) is not TrustedLaunchBinding:
            raise B300QualificationGraphProviderError(
                "prepared trusted launch binding is not exactly typed"
            )
        native = trusted.native_build_spec
        replacement = arm.transition.replacement
        if type(native) is not NativeBuildSpec or type(replacement) is not ProposalContributionRef:
            raise B300QualificationGraphProviderError(
                "prepared graph arm lacks an exact proposal or native-build identity"
            )
        trusted_tree_identity = _tree_identity_digest(tree)
        try:
            validate_native_build_spec(launch, native)
            trusted.physical_hardware.validate_against(launch.hardware)
        except (TypeError, ValueError) as exc:
            raise B300QualificationGraphProviderError(
                f"prepared launch/native identity is inconsistent: {exc}"
            ) from None
        if (
            reservation.submission_digest != publication.digest
            or arm.transition.target_id != reservation.target_id
            or replacement.target_id != reservation.target_id
            or arm.transition.target_spec_digest != replacement.target_spec_digest
            or arm.selected_delta_digest != reservation.selected_delta_digest
            or replacement.selected_delta_digest != reservation.selected_delta_digest
            or replacement.artifact_digest != publication.content_hash
            or arm.contribution_digest != replacement.digest
            or arm.challenger.stack_digest != arm.candidate.digest
            or arm.challenger.stack_digest != tree.stack_digest
            or arm.challenger.tree_digest != tree.tree_digest
            or launch.stack_digest != tree.stack_digest
            or launch.tree_digest != tree.tree_digest
            or trusted.materialized_tree_root != tree.root
            or trusted.controller_distribution_digest
            != launch.controller_distribution_digest
            or native.tree_digest != tree.tree_digest
            or native.digest != launch.native_build_spec_digest
            or session.launch_digest != launch.digest
        ):
            raise B300QualificationGraphProviderError(
                "candidate and prepared runtime do not form one exact graph binding"
            )
        return cls(
            reservation_digest=reservation.reservation_digest,
            reservation_identity_digest=canonical_digest(
                "cacheon.eval.b300-qualification-reservation-identity",
                reservation.to_dict(),
            ),
            candidate_binding_digest=candidate.digest,
            screen_attempt=candidate.screen_attempt,
            target_id=reservation.target_id,
            target_members=reservation.target_members,
            target_spec_digest=arm.transition.target_spec_digest,
            selected_delta_digest=reservation.selected_delta_digest,
            publication_content_hash=publication.content_hash,
            publication_address_digest=publication.address_digest,
            publication_digest=publication.digest,
            publication_receipt_digest=publication.publication_digest,
            prepared_arm_digest=arm.digest,
            prepared_contribution_digest=replacement.digest,
            prepared_launch_digest=launch.digest,
            materialized_stack_digest=tree.stack_digest,
            materialized_tree_digest=tree.tree_digest,
            trusted_tree_identity_digest=trusted_tree_identity,
            native_build_spec_digest=native.digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_binding_digest": self.candidate_binding_digest,
            "materialized_stack_digest": self.materialized_stack_digest,
            "materialized_tree_digest": self.materialized_tree_digest,
            "native_build_spec_digest": self.native_build_spec_digest,
            "prepared_arm_digest": self.prepared_arm_digest,
            "prepared_contribution_digest": self.prepared_contribution_digest,
            "prepared_launch_digest": self.prepared_launch_digest,
            "publication_address_digest": self.publication_address_digest,
            "publication_content_hash": self.publication_content_hash,
            "publication_digest": self.publication_digest,
            "publication_receipt_digest": self.publication_receipt_digest,
            "reservation_digest": self.reservation_digest,
            "reservation_identity_digest": self.reservation_identity_digest,
            "screen_attempt": self.screen_attempt,
            "selected_delta_digest": self.selected_delta_digest,
            "target_id": self.target_id,
            "target_members": list(self.target_members),
            "target_spec_digest": self.target_spec_digest,
            "trusted_tree_identity_digest": self.trusted_tree_identity_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "B300QualificationGraphBinding":
        fields = frozenset(
            {
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
        )
        row = _strict_object(value, fields, "graph binding")
        if type(row["target_members"]) is not list:
            raise B300QualificationGraphProviderError(
                "graph binding target members are not an exact array"
            )
        return cls(
            **{**row, "target_members": tuple(row["target_members"])}  # type: ignore[arg-type]
        )

    @property
    def digest(self) -> str:
        return canonical_digest(BINDING_SCHEMA, self.to_dict())


def _shape_to_dict(row: StructuredGraphShapeRecord) -> dict[str, object]:
    return {
        "applicable": row.applicable,
        "capture_succeeded": row.capture_succeeded,
        "descriptor_digest": row.descriptor_digest,
        "eager_passed": row.eager_passed,
        "failure_is_candidate_attributable": row.failure_is_candidate_attributable,
        "observation_complete": row.observation_complete,
        "replay_count": row.replay_count,
        "replay_passed": row.replay_passed,
    }


def _variant_to_dict(row: StructuredGraphVariantRecord) -> dict[str, object]:
    return {
        "context_applicable": row.context_applicable,
        "domain_coverage_complete": row.domain_coverage_complete,
        "shapes": [_shape_to_dict(shape) for shape in row.shapes],
        "slot_id": row.slot_id,
        "variant_id": row.variant_id,
    }


_SHAPE_FIELDS = frozenset(
    {
        "applicable",
        "capture_succeeded",
        "descriptor_digest",
        "eager_passed",
        "failure_is_candidate_attributable",
        "observation_complete",
        "replay_count",
        "replay_passed",
    }
)
_VARIANT_FIELDS = frozenset(
    {
        "context_applicable",
        "domain_coverage_complete",
        "shapes",
        "slot_id",
        "variant_id",
    }
)


def _shape_from_dict(value: object) -> StructuredGraphShapeRecord:
    row = _strict_object(value, _SHAPE_FIELDS, "graph shape")
    try:
        return StructuredGraphShapeRecord(**row)  # type: ignore[arg-type]
    except (B300QualificationCapabilityError, TypeError, ValueError) as exc:
        raise B300QualificationGraphProviderError(
            f"graph shape is not exact structured correctness evidence: {exc}"
        ) from None


def _variant_from_dict(value: object) -> StructuredGraphVariantRecord:
    row = _strict_object(value, _VARIANT_FIELDS, "graph variant")
    shapes = row["shapes"]
    if type(shapes) is not list:
        raise B300QualificationGraphProviderError(
            "graph variant shapes are not an exact array"
        )
    try:
        return StructuredGraphVariantRecord(
            slot_id=row["slot_id"],  # type: ignore[arg-type]
            variant_id=row["variant_id"],  # type: ignore[arg-type]
            context_applicable=row["context_applicable"],  # type: ignore[arg-type]
            domain_coverage_complete=row["domain_coverage_complete"],  # type: ignore[arg-type]
            shapes=tuple(_shape_from_dict(shape) for shape in shapes),
        )
    except (B300QualificationCapabilityError, TypeError, ValueError) as exc:
        raise B300QualificationGraphProviderError(
            f"graph variant is not exact structured correctness evidence: {exc}"
        ) from None


@dataclass(frozen=True)
class B300QualificationGraphArtifact:
    """Closed canonical correctness artifact returned by a commissioned probe."""

    binding: B300QualificationGraphBinding
    verification_policy_digest: str
    expected_graph_replays: int
    variants: tuple[StructuredGraphVariantRecord, ...]

    def __post_init__(self) -> None:
        if type(self.binding) is not B300QualificationGraphBinding:
            raise B300QualificationGraphProviderError(
                "graph artifact binding is not exactly typed"
            )
        object.__setattr__(
            self,
            "verification_policy_digest",
            _digest(self.verification_policy_digest, "verification policy digest"),
        )
        object.__setattr__(
            self,
            "expected_graph_replays",
            _positive(
                self.expected_graph_replays,
                "expected graph replays",
                minimum=2,
            ),
        )
        variants = self.variants
        if (
            type(variants) is not tuple
            or not variants
            or any(type(row) is not StructuredGraphVariantRecord for row in variants)
        ):
            raise B300QualificationGraphProviderError(
                "graph artifact variants must be a nonempty exact tuple"
            )
        observed_members = tuple(sorted({row.slot_id for row in variants}))
        if observed_members != self.binding.target_members:
            raise B300QualificationGraphProviderError(
                "graph artifact does not cover the exact target member domain"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "binding": self.binding.to_dict(),
            "binding_digest": self.binding.digest,
            "expected_graph_replays": self.expected_graph_replays,
            "schema": ARTIFACT_SCHEMA,
            "variants": [_variant_to_dict(row) for row in self.variants],
            "verification_policy_digest": self.verification_policy_digest,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "B300QualificationGraphArtifact":
        if type(payload) is not bytes or not payload:
            raise B300QualificationGraphProviderError(
                "graph evidence bytes must be exact and nonempty"
            )
        row = _strict_object(
            _canonical_object(payload),
            frozenset(
                {
                    "binding",
                    "binding_digest",
                    "expected_graph_replays",
                    "schema",
                    "variants",
                    "verification_policy_digest",
                }
            ),
            "graph artifact",
        )
        if row["schema"] != ARTIFACT_SCHEMA:
            raise B300QualificationGraphProviderError(
                "graph artifact schema is unsupported"
            )
        variants = row["variants"]
        if type(variants) is not list:
            raise B300QualificationGraphProviderError(
                "graph artifact variants are not an exact array"
            )
        binding = B300QualificationGraphBinding.from_dict(row["binding"])
        if row["binding_digest"] != binding.digest:
            raise B300QualificationGraphProviderError(
                "graph artifact binding digest is mismatched"
            )
        artifact = cls(
            binding=binding,
            verification_policy_digest=row["verification_policy_digest"],  # type: ignore[arg-type]
            expected_graph_replays=row["expected_graph_replays"],  # type: ignore[arg-type]
            variants=tuple(_variant_from_dict(value) for value in variants),
        )
        if artifact.canonical_bytes != payload:
            raise B300QualificationGraphProviderError(
                "graph artifact bytes changed during exact typed parsing"
            )
        return artifact


CommissionedGraphProbe = Callable[
    [
        B300QualificationGraphBinding,
        ArenaCandidateBinding,
        PreparedCandidateRuntime,
    ],
    EvidenceArtifactRef,
]
CommissionedEvidenceReopener = Callable[[EvidenceArtifactRef], bytes]


@dataclass(frozen=True)
class _AcceptedGraphEvidence:
    binding: B300QualificationGraphBinding
    reference: EvidenceArtifactRef
    payload: bytes
    artifact: B300QualificationGraphArtifact
    facts: B300FocusedGraphFacts


@dataclass(frozen=True)
class B300QualificationGraphFactsBuilder:
    """Stable GraphFactsBuilder backed by per-candidate commissioned evidence."""

    verification_policy_digest: str
    probe_authority_digest: str
    evidence_reopener_authority_digest: str
    commissioned_probe: CommissionedGraphProbe = field(repr=False, compare=False)
    evidence_reopener: CommissionedEvidenceReopener = field(repr=False, compare=False)
    schema_version: int = BUILDER_SCHEMA_VERSION
    _accepted: dict[str, _AcceptedGraphEvidence] = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=dict,
    )
    _lock: threading.Lock = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=threading.Lock,
    )

    def __post_init__(self) -> None:
        for name in (
            "verification_policy_digest",
            "probe_authority_digest",
            "evidence_reopener_authority_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if (
            type(self.schema_version) is not int
            or self.schema_version != BUILDER_SCHEMA_VERSION
        ):
            raise B300QualificationGraphProviderError(
                "graph builder schema version is unsupported"
            )
        if not callable(self.commissioned_probe) or not callable(self.evidence_reopener):
            raise B300QualificationGraphProviderError(
                "graph probe and evidence reopener must be commissioned callables"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_domain": ARTIFACT_DOMAIN,
            "artifact_media_type": ARTIFACT_MEDIA_TYPE,
            "artifact_schema": ARTIFACT_SCHEMA,
            "binding_schema": BINDING_SCHEMA,
            "builder_schema": BUILDER_SCHEMA,
            "evidence_reopener_authority_digest": (
                self.evidence_reopener_authority_digest
            ),
            "probe_authority_digest": self.probe_authority_digest,
            "schema_version": self.schema_version,
            "verification_policy_digest": self.verification_policy_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(BUILDER_SCHEMA, self.to_dict())

    @staticmethod
    def _require_reference(value: object) -> EvidenceArtifactRef:
        if type(value) is not EvidenceArtifactRef:
            raise B300QualificationGraphProviderError(
                "commissioned graph probe did not return an exact EvidenceArtifactRef"
            )
        if (
            value.domain != ARTIFACT_DOMAIN
            or value.media_type != ARTIFACT_MEDIA_TYPE
            or value.schema != ARTIFACT_SCHEMA
            or value.size < 1
        ):
            raise B300QualificationGraphProviderError(
                "commissioned graph evidence reference differs from the closed schema"
            )
        return value

    def __call__(
        self,
        candidate: ArenaCandidateBinding,
        prepared: PreparedCandidateRuntime,
    ) -> B300FocusedGraphFacts:
        binding = B300QualificationGraphBinding.derive(candidate, prepared)
        key = binding.digest
        # The commissioned callbacks are serialized with acceptance.  This
        # makes a same-binding race observe one ordered evidence history and
        # prevents two concurrent artifacts from both becoming authoritative.
        with self._lock:
            prior = self._accepted.get(key)
            if prior is not None and prior.binding != binding:
                raise B300QualificationGraphProviderError(
                    "graph binding digest is ambiguous"
                )
            try:
                reference = self._require_reference(
                    self.commissioned_probe(binding, candidate, prepared)
                )
            except B300QualificationGraphEvidenceHold:
                raise
            except B300QualificationGraphProviderError:
                raise
            except Exception as exc:
                raise B300QualificationGraphProviderError(
                    f"commissioned graph probe failed without candidate evidence: {exc}"
                ) from None
            if prior is not None and reference != prior.reference:
                raise B300QualificationGraphProviderError(
                    "one graph binding produced ambiguous evidence references"
                )
            try:
                payload = self.evidence_reopener(reference)
            except B300QualificationGraphEvidenceHold:
                raise
            except B300QualificationGraphProviderError:
                raise
            except Exception as exc:
                raise B300QualificationGraphProviderError(
                    f"commissioned graph evidence reopen failed: {exc}"
                ) from None
            if type(payload) is not bytes or not payload:
                raise B300QualificationGraphProviderError(
                    "graph evidence reopener did not return exact canonical bytes"
                )
            if (
                len(payload) != reference.size
                or hashlib.sha256(payload).hexdigest() != reference.sha256
            ):
                raise B300QualificationGraphProviderError(
                    "graph evidence bytes differ from their content-addressed reference"
                )
            artifact = B300QualificationGraphArtifact.from_canonical_bytes(payload)
            if artifact.binding != binding:
                raise B300QualificationGraphProviderError(
                    "graph artifact binding differs from the exact candidate runtime"
                )
            if artifact.verification_policy_digest != self.verification_policy_digest:
                raise B300QualificationGraphProviderError(
                    "graph artifact verification policy differs from commissioned policy"
                )
            try:
                # This is intentionally the only conversion into focused facts.
                facts = structured_focused_graph_facts(
                    artifact.expected_graph_replays,
                    artifact.variants,
                )
            except B300QualificationCapabilityError as exc:
                raise B300QualificationGraphProviderError(
                    f"structured graph correctness evidence failed closed: {exc}"
                ) from None
            observed = _AcceptedGraphEvidence(
                binding,
                reference,
                payload,
                artifact,
                facts,
            )
            if prior is not None:
                if observed != prior:
                    raise B300QualificationGraphProviderError(
                        "one graph binding produced ambiguous reopened evidence"
                    )
                return prior.facts
            self._accepted[key] = observed
            return facts


# The explicit alias makes the product usable at the registered qualification
# ``GraphFactsBuilder`` seam without conflating it with a device provider.
GraphFactsBuilder = B300QualificationGraphFactsBuilder


__all__ = [
    "ARTIFACT_DOMAIN",
    "ARTIFACT_MEDIA_TYPE",
    "ARTIFACT_SCHEMA",
    "BINDING_SCHEMA",
    "BUILDER_SCHEMA",
    "BUILDER_SCHEMA_VERSION",
    "B300QualificationGraphArtifact",
    "B300QualificationGraphBinding",
    "B300QualificationGraphFactsBuilder",
    "B300QualificationGraphProviderError",
    "CommissionedEvidenceReopener",
    "CommissionedGraphProbe",
    "GraphFactsBuilder",
]
