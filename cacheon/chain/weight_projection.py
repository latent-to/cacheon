"""The existing publication projection and its optional static-allocation evidence."""

from dataclasses import dataclass

from cacheon.chain.weight_publication_record import WeightPublicationError
from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.stack_identity import canonical_digest, require_sha256_hex

WEIGHT_PARTS = 1_000_000
_WEIGHT_PROJECTION_DOMAIN = "cacheon.chain.weight-projection"


@dataclass(frozen=True)
class WeightProjection:
    """Exact settlement output accepted by the single control-plane signer."""

    chain_scope_digest: str
    netuid: int
    validator_hotkey: str
    policy_digest: str
    settlement_state_digest: str
    evaluation_state_digest: str
    metagraph_digest: str
    arena_state_digests: tuple[str, ...]
    stack_generation: int
    effective_block: int
    crown_count: int
    evidence_digests: tuple[str, ...]
    weights_ppm: tuple[tuple[str, int], ...]
    allocation_evidence: EvidenceArtifactRef | None = None
    rewarded_evidence_digests: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for field in (
            "chain_scope_digest",
            "policy_digest",
            "settlement_state_digest",
            "evaluation_state_digest",
            "metagraph_digest",
        ):
            object.__setattr__(
                self, field, require_sha256_hex(getattr(self, field), field=field)
            )
        if (
            type(self.netuid) is not int
            or self.netuid < 0
            or not isinstance(self.validator_hotkey, str)
            or not self.validator_hotkey
            or self.validator_hotkey.strip() != self.validator_hotkey
            or len(self.validator_hotkey) > 256
        ):
            raise WeightPublicationError("projection chain/signer identity is malformed")
        for field in ("stack_generation", "effective_block", "crown_count"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise WeightPublicationError(f"projection {field} is malformed")
        evidence = tuple(self.evidence_digests)
        arenas = tuple(self.arena_state_digests)
        if (
            evidence != tuple(sorted(set(evidence)))
            or any(require_sha256_hex(value, field="evidence_digest") != value for value in evidence)
            or self.crown_count > len(evidence)
            or not arenas
            or arenas != tuple(sorted(set(arenas)))
            or any(require_sha256_hex(value, field="arena_state_digest") != value for value in arenas)
        ):
            raise WeightPublicationError("projection evidence inventory is malformed")
        object.__setattr__(self, "evidence_digests", evidence)
        object.__setattr__(self, "arena_state_digests", arenas)
        if self.allocation_evidence is not None:
            rewarded = self.rewarded_evidence_digests
            if (type(self.allocation_evidence) is not EvidenceArtifactRef
                    or self.allocation_evidence.sha256 not in evidence
                    or type(rewarded) is not tuple or rewarded != tuple(sorted(set(rewarded)))
                    or not set(rewarded) <= set(evidence)):
                raise WeightPublicationError("allocation evidence or earning inventory is malformed")
        elif self.rewarded_evidence_digests is not None:
            raise WeightPublicationError("earning inventory requires allocation evidence")
        raw_rows = tuple(self.weights_ppm)
        if any(type(row) is not tuple or len(row) != 2 for row in raw_rows):
            raise WeightPublicationError("projection weights are not canonical ppm")
        rows = tuple((row[0], row[1]) for row in raw_rows)
        if (
            not rows
            or tuple(hotkey for hotkey, _ppm in rows)
            != tuple(sorted({hotkey for hotkey, _ppm in rows}))
            or any(
                not isinstance(hotkey, str)
                or not hotkey
                or hotkey.strip() != hotkey
                or len(hotkey) > 256
                or type(ppm) is not int
                or ppm <= 0
                for hotkey, ppm in rows
            )
            or sum(ppm for _hotkey, ppm in rows) != WEIGHT_PARTS
        ):
            raise WeightPublicationError("projection weights are not canonical ppm")
        object.__setattr__(self, "weights_ppm", rows)

    @property
    def weights(self) -> dict[str, float]:
        return {hotkey: ppm / WEIGHT_PARTS for hotkey, ppm in self.weights_ppm}

    def to_dict(self) -> dict[str, object]:
        row = {
            "chain_scope_digest": self.chain_scope_digest,
            "arena_state_digests": list(self.arena_state_digests),
            "crown_count": self.crown_count,
            "effective_block": self.effective_block,
            "evaluation_state_digest": self.evaluation_state_digest,
            "evidence_digests": list(self.evidence_digests),
            "netuid": self.netuid,
            "metagraph_digest": self.metagraph_digest,
            "policy_digest": self.policy_digest,
            "settlement_state_digest": self.settlement_state_digest,
            "stack_generation": self.stack_generation,
            "validator_hotkey": self.validator_hotkey,
            "weights_ppm": [list(row) for row in self.weights_ppm],
        }

        if self.allocation_evidence is not None:
            row["allocation_evidence"] = self.allocation_evidence.to_dict()
            row["rewarded_evidence_digests"] = list(self.rewarded_evidence_digests)
        return row

    @classmethod
    def from_dict(cls, value: object) -> "WeightProjection":
        optional = {"allocation_evidence", "rewarded_evidence_digests"}
        fields = set(cls.__dataclass_fields__) - optional
        if type(value) is dict and optional.intersection(value):
            fields |= optional
        if type(value) is not dict or set(value) != fields:
            raise WeightPublicationError("weight projection fields do not match")
        if (
            type(value["evidence_digests"]) is not list
            or type(value["arena_state_digests"]) is not list
            or type(value["weights_ppm"]) is not list
        ):
            raise WeightPublicationError("weight projection arrays are malformed")
        value = dict(value)
        if "allocation_evidence" in value:
            if type(value["rewarded_evidence_digests"]) is not list:
                raise WeightPublicationError("earning inventory is malformed")
            value["allocation_evidence"] = EvidenceArtifactRef.from_dict(value["allocation_evidence"])
            value["rewarded_evidence_digests"] = tuple(value["rewarded_evidence_digests"])
        rows = value["weights_ppm"]
        if any(type(row) is not list or len(row) != 2 for row in rows):
            raise WeightPublicationError("weight projection rows are malformed")
        return cls(
            **{
                **value,
                "evidence_digests": tuple(value["evidence_digests"]),
                "arena_state_digests": tuple(value["arena_state_digests"]),
                "weights_ppm": tuple(tuple(row) for row in rows),
            }
        )  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(_WEIGHT_PROJECTION_DOMAIN, self.to_dict())
