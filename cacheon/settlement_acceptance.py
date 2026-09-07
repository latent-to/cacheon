"""Accepted qualification candidates and the retained evidence used to settle them."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.stack_identity import canonical_digest
from cacheon.stack_plan import StackArmIdentity

@dataclass(frozen=True)
class SettlementCandidate:
    """An accepted qualification, retaining historical reproduced pairs unchanged."""

    primary: SettlementQualification
    reproduction: SettlementQualification | None = None

    def __post_init__(self) -> None:
        if type(self.primary) is not SettlementQualification:
            raise SettlementError("settlement candidate requires an exact qualification")
        if self.reproduction is None:
            if self.primary.audit_policy is None:
                raise SettlementError("single-run settlement requires an audited qualification")
            return
        if type(self.reproduction) is not SettlementQualification:
            raise SettlementError("retained reproduction is not an exact qualification")
        if self.primary.reproduction_identity != self.reproduction.reproduction_identity:
            raise SettlementError(
                "independent reproduction differs from the primary reproduction identity"
            )
        common = (
            "lane", "arena_digest", "reservation_digest", "finalized_block",
            "event_index", "event_subindex", "hotkey", "target_id", "members",
            "selected_delta_digest", "arm_digest", "incumbent_stack_digest",
            "incumbent_tree_digest", "candidate_stack_digest", "candidate_tree_digest",
            "incumbent_manifest", "proposal_digest", "candidate_manifest",
            "speed_evidence_policy_digest",
            "audit_control_digest",
        )
        if any(
            getattr(self.primary, field) != getattr(self.reproduction, field)
            for field in common
        ):
            raise SettlementError(
                "independent reproduction differs from the primary contribution identity"
            )
        distinct = (
            "qualification_authority_digest", "qualification_plan_digest",
            "qualification_attempt_digest", "qualification_report_digest",
            "selection_commitment_digest", "selection_secret_commitment_digest",
            "selection_evidence_digest",
        )
        if any(
            getattr(self.primary, field) == getattr(self.reproduction, field)
            for field in distinct
        ):
            raise SettlementError(
                "independent reproduction reuses primary authority or evidence"
            )
        primary_audit = self.primary.audit_policy
        reproduction_audit = self.reproduction.audit_policy
        if (primary_audit is None) != (reproduction_audit is None):
            raise SettlementError(
                "independent reproduction mixes audited and auditless qualifications"
            )
        if primary_audit is not None and reproduction_audit is not None:
            if (
                primary_audit.control != reproduction_audit.control
                or primary_audit.validator_seed == reproduction_audit.validator_seed
                or self.primary.audit_evidence_digest
                == self.reproduction.audit_evidence_digest
            ):
                raise SettlementError(
                    "independent reproduction reuses or changes slot-audit authority"
                )
        primary_orientation = self.primary.resident_lane_orientation
        reproduction_orientation = self.reproduction.resident_lane_orientation
        if (primary_orientation is None) != (reproduction_orientation is None):
            raise SettlementError(
                "independent reproduction has incomplete resident lane orientation"
            )
        if (
            primary_orientation is not None
            and reproduction_orientation is not None
            and not reproduction_orientation.is_exact_swap_of(primary_orientation)
        ):
            raise SettlementError(
                "independent reproduction did not swap physical TP lane orientation"
            )

    @classmethod
    def from_qualification(cls, qualification: SettlementQualification) -> "SettlementCandidate":
        """Accept the complete audited PASS without scheduling another evaluator run."""

        return cls(qualification)

    @property
    def qualifications(self) -> tuple[SettlementQualification, ...]:
        """Return exactly the independently retained attempts behind this candidate."""

        return (self.primary,) if self.reproduction is None else (self.primary, self.reproduction)

    @classmethod
    def from_reproductions(
        cls,
        primary: SettlementQualification,
        reproduction: SettlementQualification,
    ) -> "SettlementCandidate":
        """Reconstruct a historical independently bound pair."""

        candidate = cls(primary, reproduction)
        # ``__post_init__`` already refuses a mixed audited/auditless pair and
        # requires an exact physical lane swap whenever orientation is present.
        # A resident acceptance pair (v6 speed PASS on both lane orientations)
        # carries no audit witness; the enforced swap is its reproduction
        # defense.  Any other auditless pair is legacy history and cannot
        # become a new candidate.
        if (
            candidate.primary.audit_policy is None
            or candidate.reproduction.audit_policy is None
        ) and (
            candidate.primary.resident_lane_orientation is None
            or candidate.reproduction.resident_lane_orientation is None
        ):
            raise SettlementError(
                "new settlement candidate requires two audited qualifications"
            )
        return candidate

    def __getattr__(self, field: str):
        if field in SettlementQualification.__dataclass_fields__:
            return getattr(self.primary, field)
        raise AttributeError(field)

    @property
    def speedup(self) -> str:
        """Preserve historical pair scores; new candidates use their accepted run."""

        return min((value.speedup for value in self.qualifications), key=Decimal)

    @property
    def finalized_order(self) -> tuple[int, int, int, str]:
        return self.primary.finalized_order

    @property
    def incumbent(self) -> StackArmIdentity:
        return self.primary.incumbent

    @property
    def challenger(self) -> StackArmIdentity:
        return self.primary.challenger

    @property
    def reproduction_identity(self) -> SettlementReproductionIdentity:
        return self.primary.reproduction_identity

    def to_dict(self) -> dict[str, object]:
        """Preserve pair bytes and omit the absent second attempt for new candidates."""

        value = {"primary": self.primary.to_dict()}
        if self.reproduction is not None:
            value["reproduction"] = self.reproduction.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: object) -> "SettlementCandidate":
        """Read current single-run candidates and unchanged historical pair rows."""

        if type(value) is not dict or set(value) not in ({"primary"}, {"primary", "reproduction"}):
            raise SettlementError("settlement candidate fields do not match")
        return cls(
            SettlementQualification.from_dict(value["primary"]),
            SettlementQualification.from_dict(value["reproduction"])
            if "reproduction" in value else None,
        )

    @property
    def digest(self) -> str:
        domain = (
            "cacheon.settlement.candidate.v5"
            if self.reproduction is None
            else "cacheon.settlement.candidate.v4"
            if self.primary.resident_lane_orientation is not None
            else "cacheon.settlement.candidate.v2"
            if self.primary.audit_policy is None
            else "cacheon.settlement.candidate.v3"
        )
        return canonical_digest(domain, self.to_dict())


@dataclass(frozen=True)
class SettlementEvidence:
    """Receipt that the candidate’s retained qualification artifacts were reopened."""

    candidate_digest: str
    reservation_digest: str
    primary_authority_digest: str
    primary_attempt_ref: EvidenceArtifactRef
    primary_report_digest: str
    primary_selection_evidence_digest: str
    reproduction_authority_digest: str = ""
    reproduction_attempt_ref: EvidenceArtifactRef | None = None
    reproduction_report_digest: str = ""
    reproduction_selection_evidence_digest: str = ""

    def __post_init__(self) -> None:
        for field in (
            "candidate_digest", "reservation_digest", "primary_authority_digest",
            "primary_report_digest", "primary_selection_evidence_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if type(self.primary_attempt_ref) is not EvidenceArtifactRef:
            raise SettlementError("settlement attempt reference is not exactly typed")
        if self.reproduction_attempt_ref is None:
            if any((self.reproduction_authority_digest, self.reproduction_report_digest,
                    self.reproduction_selection_evidence_digest)):
                raise SettlementError("settlement evidence has an incomplete reproduction")
            return
        if type(self.reproduction_attempt_ref) is not EvidenceArtifactRef:
            raise SettlementError("retained reproduction reference is not exactly typed")
        for field in ("reproduction_authority_digest", "reproduction_report_digest",
                      "reproduction_selection_evidence_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if any(
            left == right
            for left, right in (
                (self.primary_authority_digest, self.reproduction_authority_digest),
                (self.primary_attempt_ref.sha256, self.reproduction_attempt_ref.sha256),
                (self.primary_report_digest, self.reproduction_report_digest),
                (self.primary_selection_evidence_digest, self.reproduction_selection_evidence_digest),
            )
        ):
            raise SettlementError("settlement evidence does not contain a reproduction")

    @classmethod
    def bind(
        cls,
        candidate: SettlementCandidate,
        *,
        primary_attempt_ref: EvidenceArtifactRef,
        reproduction_attempt_ref: EvidenceArtifactRef | None = None,
    ) -> "SettlementEvidence":
        """Bind exactly the retained attempts represented by the accepted candidate."""

        if type(candidate) is not SettlementCandidate:
            raise SettlementError("settlement evidence candidate is not exactly typed")
        if (type(primary_attempt_ref) is not EvidenceArtifactRef
                or primary_attempt_ref.sha256 != candidate.primary.qualification_attempt_digest):
            raise SettlementError("settlement attempt references differ from the candidate")
        values = (
            candidate.digest, candidate.reservation_digest,
            candidate.primary.qualification_authority_digest, primary_attempt_ref,
            candidate.primary.qualification_report_digest,
            candidate.primary.selection_evidence_digest,
        )
        if candidate.reproduction is None:
            if reproduction_attempt_ref is not None:
                raise SettlementError("single-run evidence contains an extra attempt")
            return cls(*values)
        if (type(reproduction_attempt_ref) is not EvidenceArtifactRef
                or reproduction_attempt_ref.sha256
                != candidate.reproduction.qualification_attempt_digest):
            raise SettlementError("settlement attempt references differ from the candidate")
        return cls(
            *values, candidate.reproduction.qualification_authority_digest,
            reproduction_attempt_ref, candidate.reproduction.qualification_report_digest,
            candidate.reproduction.selection_evidence_digest,
        )

    def to_dict(self) -> dict[str, object]:
        """Encode only the attempts actually backing the settlement."""

        value: dict[str, object] = {
            "candidate_digest": self.candidate_digest,
            "primary_attempt_ref": self.primary_attempt_ref.to_dict(),
            "primary_authority_digest": self.primary_authority_digest,
            "primary_report_digest": self.primary_report_digest,
            "primary_selection_evidence_digest": self.primary_selection_evidence_digest,
            "reservation_digest": self.reservation_digest,
        }
        if self.reproduction_attempt_ref is not None:
            value.update(
                reproduction_attempt_ref=self.reproduction_attempt_ref.to_dict(),
                reproduction_authority_digest=self.reproduction_authority_digest,
                reproduction_report_digest=self.reproduction_report_digest,
                reproduction_selection_evidence_digest=self.reproduction_selection_evidence_digest,
            )
        return value

    @classmethod
    def from_dict(cls, value: object) -> "SettlementEvidence":
        """Reopen either retained evidence shape without inventing a second attempt."""

        fields = set(cls.__dataclass_fields__)
        primary_fields = {field for field in fields if not field.startswith("reproduction_")}
        if type(value) is not dict or set(value) not in (fields, primary_fields):
            raise SettlementError("settlement evidence fields do not match")
        return cls(
            value["candidate_digest"],  # type: ignore[arg-type]
            value["reservation_digest"],  # type: ignore[arg-type]
            value["primary_authority_digest"],  # type: ignore[arg-type]
            EvidenceArtifactRef.from_dict(value["primary_attempt_ref"]),
            value["primary_report_digest"],  # type: ignore[arg-type]
            value["primary_selection_evidence_digest"],  # type: ignore[arg-type]
            value.get("reproduction_authority_digest", ""),  # type: ignore[arg-type]
            EvidenceArtifactRef.from_dict(value["reproduction_attempt_ref"])
            if "reproduction_attempt_ref" in value else None,
            value.get("reproduction_report_digest", ""),  # type: ignore[arg-type]
            value.get("reproduction_selection_evidence_digest", ""),  # type: ignore[arg-type]
        )

    @property
    def digest(self) -> str:
        domain = ("cacheon.settlement.evidence.v2" if self.reproduction_attempt_ref is None
                  else "cacheon.settlement.evidence")
        return canonical_digest(domain, self.to_dict())


# Both import orders share the facade's exact qualification and error types.
# These imports follow the classes so importing this module directly is safe too.
from cacheon.settlement import (
    SettlementError, SettlementQualification, SettlementReproductionIdentity, _digest,
)
