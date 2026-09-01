"""Pure, content-addressed emissions projection.

This module owns no chain client, wallet, database, or settlement transition.  It
accepts only crowns whose retained evidence has already been reopened by the
settlement authority and either returns one complete integer weight projection or
fails without a partial vector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Context, Decimal, localcontext
from typing import Iterable, Mapping

from cacheon.stack_identity import canonical_digest
from cacheon.stack_manifest import EvaluationStackManifest
from cacheon.target_catalog import TargetCatalog, TargetResolutionError
from cacheon._strict import require_digest, require_exact_fields, require_int


POLICY_SCHEMA_VERSION = 1
POLICY_VERSION = "cacheon.emissions.v1.3"
WEIGHT_PPM = 1_000_000
# Internal integer scale for log-speedup × stall-multiplier × exponential decay.
# Consensus-critical: changing this constant changes every published vector.
CREDIT_SCALE = 1_000_000_000_000
_MATH_CONTEXT = Context(prec=50, rounding=ROUND_FLOOR)
_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}\Z")
_HOTKEY = re.compile(r"[^\s]{1,256}\Z")


class EconomicsError(ValueError):
    """The policy, authority, or complete reward projection is invalid."""


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=EconomicsError)


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    return require_int(value, field=field, error=EconomicsError, minimum=minimum)


def _hotkey(value: object, field: str = "hotkey") -> str:
    if not isinstance(value, str) or _HOTKEY.fullmatch(value) is None:
        raise EconomicsError(f"{field} is malformed")
    return value


def _canonical_hotkeys(value: object) -> tuple[str, ...]:
    if value is None:
        value = ()
    if type(value) not in (tuple, list):
        raise EconomicsError("excluded_hotkeys must be a list")
    hotkeys = tuple(_hotkey(row, "excluded_hotkeys") for row in value)
    if len(set(hotkeys)) != len(hotkeys):
        raise EconomicsError("excluded_hotkeys are duplicated")
    return tuple(sorted(hotkeys))


def _canonical_claim_digests(value: object) -> tuple[str, ...]:
    if value is None:
        value = ()
    if type(value) not in (tuple, list):
        raise EconomicsError("excluded_claim_digests must be a list")
    digests = tuple(_digest(row, "excluded_claim_digests") for row in value)
    if len(set(digests)) != len(digests):
        raise EconomicsError("excluded_claim_digests are duplicated")
    return tuple(sorted(digests))


def _strict(value: object, fields: set[str], name: str) -> dict[str, object]:
    return require_exact_fields(
        value, fields=frozenset(fields), label=name, error=EconomicsError, exact_dict=True
    )


def _decimal_floor_int(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def improvement_units(speedup_ppm: int) -> Decimal:
    """Log-relative speedup units: ln(speedup). Path-independent across compounds."""

    _integer(speedup_ppm, "speedup_ppm", minimum=WEIGHT_PPM + 1)
    with localcontext(_MATH_CONTEXT):
        units = (Decimal(speedup_ppm) / Decimal(WEIGHT_PPM)).ln()
    if units <= 0:
        raise EconomicsError("log-relative improvement is not positive")
    return units


def time_multiplier(elapsed_blocks: int, scale_blocks: int) -> Decimal:
    """Uncapped reverse-dutch stall bonus: 1 + sqrt(elapsed / scale)."""

    _integer(elapsed_blocks, "elapsed_blocks")
    _integer(scale_blocks, "scale_blocks", minimum=1)
    with localcontext(_MATH_CONTEXT):
        return Decimal(1) + (Decimal(elapsed_blocks) / Decimal(scale_blocks)).sqrt()


def decay_factor(age_blocks: int, half_life_blocks: int) -> Decimal:
    """Exponential remaining share: 2^(-age / half_life)."""

    _integer(age_blocks, "age_blocks")
    _integer(half_life_blocks, "half_life_blocks", minimum=1)
    with localcontext(_MATH_CONTEXT):
        return Decimal(2) ** (-Decimal(age_blocks) / Decimal(half_life_blocks))


@dataclass(frozen=True)
class EmissionsPolicyManifest:
    """Validator-consensus parameters for one deterministic reward policy."""

    half_life_blocks: int
    discovery_lifetime_blocks: int
    discovery_pool_ppm: int
    time_multiplier_scale_blocks: int
    excluded_hotkeys: tuple[str, ...] = ()
    excluded_claim_digests: tuple[str, ...] = ()
    schema_version: int = POLICY_SCHEMA_VERSION
    policy_version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        _integer(self.half_life_blocks, "half_life_blocks", minimum=1)
        _integer(
            self.discovery_lifetime_blocks,
            "discovery_lifetime_blocks",
            minimum=1,
        )
        _integer(self.discovery_pool_ppm, "discovery_pool_ppm")
        _integer(
            self.time_multiplier_scale_blocks,
            "time_multiplier_scale_blocks",
            minimum=1,
        )
        object.__setattr__(
            self, "excluded_hotkeys", _canonical_hotkeys(self.excluded_hotkeys)
        )
        object.__setattr__(
            self,
            "excluded_claim_digests",
            _canonical_claim_digests(self.excluded_claim_digests),
        )
        if self.discovery_pool_ppm >= WEIGHT_PPM:
            raise EconomicsError("discovery_pool_ppm must leave standing reward capacity")
        if self.schema_version != POLICY_SCHEMA_VERSION:
            raise EconomicsError("emissions policy schema_version is unsupported")
        if self.policy_version != POLICY_VERSION:
            raise EconomicsError("emissions policy_version is unsupported")

    def excludes_accepted(self, claim: "AcceptedRewardClaim") -> bool:
        return (
            claim.hotkey in self.excluded_hotkeys
            or claim.digest in self.excluded_claim_digests
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "discovery_lifetime_blocks": self.discovery_lifetime_blocks,
            "discovery_pool_ppm": self.discovery_pool_ppm,
            "excluded_claim_digests": list(self.excluded_claim_digests),
            "excluded_hotkeys": list(self.excluded_hotkeys),
            "half_life_blocks": self.half_life_blocks,
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
            "time_multiplier_scale_blocks": self.time_multiplier_scale_blocks,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EmissionsPolicyManifest":
        row = _strict(
            value,
            {
                "discovery_lifetime_blocks",
                "discovery_pool_ppm",
                "excluded_claim_digests",
                "excluded_hotkeys",
                "half_life_blocks",
                "policy_version",
                "schema_version",
                "time_multiplier_scale_blocks",
            },
            "emissions policy",
        )
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.economics.policy", self.to_dict())


@dataclass(frozen=True)
class MetagraphMember:
    uid: int
    hotkey: str

    def __post_init__(self) -> None:
        _integer(self.uid, "metagraph uid")
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey, "metagraph hotkey"))

    def to_dict(self) -> dict[str, object]:
        return {"hotkey": self.hotkey, "uid": self.uid}


@dataclass(frozen=True)
class GlobalRewardProjectionContext:
    """Chain authority shared by every arena in one global projection."""

    chain_scope_digest: str
    validator_hotkey: str
    current_block: int
    current_block_hash: str
    metagraph_members: tuple[MetagraphMember, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chain_scope_digest",
            _digest(self.chain_scope_digest, "chain_scope_digest"),
        )
        object.__setattr__(self, "validator_hotkey", _hotkey(self.validator_hotkey))
        _integer(self.current_block, "current_block")
        if (
            not isinstance(self.current_block_hash, str)
            or _BLOCK_HASH.fullmatch(self.current_block_hash) is None
        ):
            raise EconomicsError("current_block_hash is malformed")
        members = tuple(self.metagraph_members)
        if any(type(row) is not MetagraphMember for row in members):
            raise EconomicsError("metagraph_members must be exactly typed")
        members = tuple(sorted(members, key=lambda row: (row.uid, row.hotkey)))
        if (
            not members
            or len({row.uid for row in members}) != len(members)
            or len({row.hotkey for row in members}) != len(members)
        ):
            raise EconomicsError("metagraph membership is empty or duplicated")
        if self.validator_hotkey not in {row.hotkey for row in members}:
            raise EconomicsError("validator is absent from the current metagraph")
        object.__setattr__(self, "metagraph_members", members)

    @property
    def eligible_hotkeys(self) -> frozenset[str]:
        return frozenset(row.hotkey for row in self.metagraph_members)

    @property
    def metagraph_digest(self) -> str:
        return canonical_digest(
            "cacheon.economics.metagraph-membership",
            {
                "block": self.current_block,
                "block_hash": self.current_block_hash,
                "chain_scope_digest": self.chain_scope_digest,
                "members": [row.to_dict() for row in self.metagraph_members],
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "chain_scope_digest": self.chain_scope_digest,
            "current_block": self.current_block,
            "current_block_hash": self.current_block_hash,
            "metagraph_digest": self.metagraph_digest,
            "metagraph_members": [row.to_dict() for row in self.metagraph_members],
            "validator_hotkey": self.validator_hotkey,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.economics.global-context", self.to_dict())


@dataclass(frozen=True)
class StandingRewardClaim:
    """One reopened crown for one currently active registered target."""

    arena_digest: str
    target_id: str
    target_spec_digest: str
    contribution_digest: str
    hotkey: str
    speedup_ppm: int
    crowned_block: int
    retained_evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id:
            raise EconomicsError("standing target_id is malformed")
        for field in (
            "arena_digest",
            "target_spec_digest",
            "contribution_digest",
            "retained_evidence_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        _integer(self.speedup_ppm, "speedup_ppm", minimum=WEIGHT_PPM + 1)
        _integer(self.crowned_block, "crowned_block")

    @property
    def family_id(self) -> str:
        return canonical_digest(
            "cacheon.economics.standing-family",
            {
                "arena_digest": self.arena_digest,
                "target_id": self.target_id,
                "target_spec_digest": self.target_spec_digest,
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "contribution_digest": self.contribution_digest,
            "crowned_block": self.crowned_block,
            "hotkey": self.hotkey,
            "retained_evidence_digest": self.retained_evidence_digest,
            "speedup_ppm": self.speedup_ppm,
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "StandingRewardClaim":
        row = _strict(value, set(cls.__dataclass_fields__), "standing reward claim")
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.economics.standing-claim", self.to_dict())


@dataclass(frozen=True)
class AcceptedRewardClaim:
    """One accepted CROWN that continues earning after later incumbents replace it."""

    arena_digest: str
    target_id: str
    target_spec_digest: str
    contribution_digest: str
    hotkey: str
    speedup_ppm: int
    crowned_block: int
    predecessor_block: int
    retained_evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id:
            raise EconomicsError("accepted target_id is malformed")
        for field in (
            "arena_digest",
            "target_spec_digest",
            "contribution_digest",
            "retained_evidence_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        _integer(self.speedup_ppm, "speedup_ppm", minimum=WEIGHT_PPM + 1)
        _integer(self.crowned_block, "crowned_block")
        _integer(self.predecessor_block, "predecessor_block")
        if self.predecessor_block > self.crowned_block:
            raise EconomicsError("accepted crown predates its arena stall clock")

    @property
    def family_id(self) -> str:
        return canonical_digest(
            "cacheon.economics.standing-family",
            {
                "arena_digest": self.arena_digest,
                "target_id": self.target_id,
                "target_spec_digest": self.target_spec_digest,
            },
        )

    def elapsed_blocks(self) -> int:
        return self.crowned_block - self.predecessor_block

    def credit_at(self, block: int, policy: EmissionsPolicyManifest) -> int:
        _integer(block, "credit block")
        if block < self.crowned_block:
            raise EconomicsError("crown is newer than projection authority")
        if type(policy) is not EmissionsPolicyManifest:
            raise EconomicsError("policy is not exactly typed")
        age = block - self.crowned_block
        with localcontext(_MATH_CONTEXT):
            raw = (
                improvement_units(self.speedup_ppm)
                * time_multiplier(
                    self.elapsed_blocks(), policy.time_multiplier_scale_blocks
                )
                * decay_factor(age, policy.half_life_blocks)
                * Decimal(CREDIT_SCALE)
            )
            return _decimal_floor_int(raw)

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "contribution_digest": self.contribution_digest,
            "crowned_block": self.crowned_block,
            "hotkey": self.hotkey,
            "predecessor_block": self.predecessor_block,
            "retained_evidence_digest": self.retained_evidence_digest,
            "speedup_ppm": self.speedup_ppm,
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "AcceptedRewardClaim":
        row = _strict(value, set(cls.__dataclass_fields__), "accepted reward claim")
        return cls(**row)  # type: ignore[arg-type]

    @classmethod
    def from_standing(
        cls,
        claim: StandingRewardClaim,
        predecessor_block: int | None = None,
    ) -> "AcceptedRewardClaim":
        if type(claim) is not StandingRewardClaim:
            raise EconomicsError("accepted claim standing authority is not exactly typed")
        pred = claim.crowned_block if predecessor_block is None else predecessor_block
        return cls(
            claim.arena_digest,
            claim.target_id,
            claim.target_spec_digest,
            claim.contribution_digest,
            claim.hotkey,
            claim.speedup_ppm,
            claim.crowned_block,
            pred,
            claim.retained_evidence_digest,
        )

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.economics.accepted-claim", self.to_dict())


@dataclass(frozen=True)
class DiscoveryBountyClaim:
    """One non-renewable, expiring discovery reward claim."""

    proposal_digest: str
    retained_evidence_digest: str
    hotkey: str
    bounty_units: int
    awarded_block: int

    def __post_init__(self) -> None:
        for field in ("proposal_digest", "retained_evidence_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        _integer(self.bounty_units, "bounty_units", minimum=1)
        _integer(self.awarded_block, "awarded_block")

    def live_at(self, block: int, policy: EmissionsPolicyManifest) -> bool:
        _integer(block, "bounty block")
        if block < self.awarded_block:
            raise EconomicsError("discovery claim is newer than projection authority")
        return block - self.awarded_block < policy.discovery_lifetime_blocks

    def to_dict(self) -> dict[str, object]:
        return {
            "awarded_block": self.awarded_block,
            "bounty_units": self.bounty_units,
            "hotkey": self.hotkey,
            "proposal_digest": self.proposal_digest,
            "retained_evidence_digest": self.retained_evidence_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryBountyClaim":
        row = _strict(value, set(cls.__dataclass_fields__), "discovery bounty claim")
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.economics.discovery-claim", self.to_dict())


@dataclass(frozen=True)
class ArenaRewardAuthority:
    """One arena's complete active stack generation and reopened crowns."""

    catalog: TargetCatalog
    stack: EvaluationStackManifest
    stack_generation: int
    standing_claims: tuple[StandingRewardClaim, ...]

    def __post_init__(self) -> None:
        if type(self.catalog) is not TargetCatalog:
            raise EconomicsError("arena reward catalog is not exactly typed")
        if type(self.stack) is not EvaluationStackManifest:
            raise EconomicsError("arena reward stack is not exactly typed")
        _integer(self.stack_generation, "arena stack_generation")
        claims = tuple(self.standing_claims)
        if any(type(row) is not StandingRewardClaim for row in claims):
            raise EconomicsError("arena standing claims are not exactly typed")
        object.__setattr__(self, "standing_claims", claims)

    @property
    def arena_digest(self) -> str:
        return self.stack.arena_digest

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "catalog_digest": self.catalog.digest,
            "stack_digest": self.stack.digest,
            "stack_generation": self.stack_generation,
            "standing_claims": [
                row.digest for row in sorted(self.standing_claims, key=lambda row: row.target_id)
            ],
        }

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.economics.arena-authority", self.to_dict())


@dataclass(frozen=True)
class StandingFamilyCredit:
    arena_digest: str
    family_id: str
    target_id: str
    claim_digest: str
    hotkey: str
    credit: int

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "claim_digest": self.claim_digest,
            "credit": self.credit,
            "family_id": self.family_id,
            "hotkey": self.hotkey,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class DiscoveryBountyCredit:
    claim_digest: str
    hotkey: str
    credit: int

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_digest": self.claim_digest,
            "credit": self.credit,
            "hotkey": self.hotkey,
        }


@dataclass(frozen=True)
class HotkeyWeight:
    hotkey: str
    weight_ppm: int

    def to_dict(self) -> dict[str, object]:
        return {"hotkey": self.hotkey, "weight_ppm": self.weight_ppm}


@dataclass(frozen=True)
class GlobalRewardProjection:
    policy_digest: str
    context: GlobalRewardProjectionContext
    arena_authority_digests: tuple[str, ...]
    standing: tuple[StandingFamilyCredit, ...]
    discovery: tuple[DiscoveryBountyCredit, ...]
    expired_discovery_claims: tuple[str, ...]
    weights: tuple[HotkeyWeight, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_digest", _digest(self.policy_digest, "policy_digest"))
        if type(self.context) is not GlobalRewardProjectionContext:
            raise EconomicsError("global reward context is not exactly typed")
        for digest in self.arena_authority_digests:
            _digest(digest, "arena authority digest")
        if tuple(self.arena_authority_digests) != tuple(sorted(self.arena_authority_digests)):
            raise EconomicsError("arena authorities are not canonical")
        if tuple(
            (row.arena_digest, row.target_id, row.claim_digest) for row in self.standing
        ) != tuple(
            sorted(
                (row.arena_digest, row.target_id, row.claim_digest)
                for row in self.standing
            )
        ):
            raise EconomicsError("global standing families are not canonical")
        if tuple(row.hotkey for row in self.weights) != tuple(sorted(row.hotkey for row in self.weights)):
            raise EconomicsError("global weights are not canonical")
        if len({row.hotkey for row in self.weights}) != len(self.weights) or sum(
            row.weight_ppm for row in self.weights
        ) != WEIGHT_PPM:
            raise EconomicsError("global weights are not exactly normalized")

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_authority_digests": list(self.arena_authority_digests),
            "context": self.context.to_dict(),
            "discovery": [row.to_dict() for row in self.discovery],
            "expired_discovery_claims": list(self.expired_discovery_claims),
            "policy_digest": self.policy_digest,
            "standing": [row.to_dict() for row in self.standing],
            "weights": [row.to_dict() for row in self.weights],
        }

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.economics.global-reward-projection", self.to_dict())

    @property
    def weights_by_hotkey(self) -> Mapping[str, int]:
        return {row.hotkey: row.weight_ppm for row in self.weights}


def _allocate_pool(credits: Mapping[str, int], pool: int) -> dict[str, int]:
    positive = {hotkey: credit for hotkey, credit in credits.items() if credit > 0}
    total = sum(positive.values())
    if total <= 0:
        raise EconomicsError("a non-empty reward pool has no positive credit")
    result: dict[str, int] = {}
    remainders = []
    for hotkey in sorted(positive):
        quotient, remainder = divmod(positive[hotkey] * pool, total)
        result[hotkey] = quotient
        remainders.append((remainder, hotkey))
    missing = pool - sum(result.values())
    for _remainder, hotkey in sorted(remainders, key=lambda row: (-row[0], row[1]))[
        :missing
    ]:
        result[hotkey] += 1
    return result


def project_global_rewards(
    policy: EmissionsPolicyManifest,
    context: GlobalRewardProjectionContext,
    arenas: Iterable[ArenaRewardAuthority],
    earning_claims: Iterable[AcceptedRewardClaim],
    discovery_claims: Iterable[DiscoveryBountyClaim] = (),
) -> GlobalRewardProjection:
    """Pool every retained accepted contribution before one indivisible vector.

    Each accepted CROWN freezes log-relative speedup units and the arena stall
    multiplier at the proposal's finalized submission block. Credit then decays
    exponentially with age. Discovery bounties stay in a separate pool and do
    not reset the arena clock. Absent-hotkey share burns to the validator.
    """

    if type(policy) is not EmissionsPolicyManifest:
        raise EconomicsError("policy is not exactly typed")
    if type(context) is not GlobalRewardProjectionContext:
        raise EconomicsError("global projection context is not exactly typed")
    authorities = tuple(arenas)
    if not authorities or any(type(row) is not ArenaRewardAuthority for row in authorities):
        raise EconomicsError("global projection requires typed arena authorities")
    if len({row.arena_digest for row in authorities}) != len(authorities):
        raise EconomicsError("global projection contains duplicate arenas")
    authorities = tuple(sorted(authorities, key=lambda row: row.arena_digest))
    earning = tuple(earning_claims)
    if any(type(row) is not AcceptedRewardClaim for row in earning):
        raise EconomicsError("accepted reward claims are not exactly typed")
    if len({row.digest for row in earning}) != len(earning):
        raise EconomicsError("accepted reward claims are duplicated")
    by_arena = {row.arena_digest: row for row in authorities}
    if {row.arena_digest for row in earning} - set(by_arena):
        raise EconomicsError("accepted reward claim belongs to an absent arena")
    eligible = context.eligible_hotkeys
    standing_index: dict[tuple[str, str, str], AcceptedRewardClaim] = {}
    for claim in earning:
        key = (claim.arena_digest, claim.target_id, claim.contribution_digest)
        if key in standing_index:
            raise EconomicsError("accepted reward claims reuse a contribution")
        standing_index[key] = claim
    for authority in authorities:
        catalog, stack = authority.catalog, authority.stack
        if stack.catalog_digest != catalog.digest or stack.catalog_snapshot != catalog.snapshot():
            raise EconomicsError("evaluation stack and reward catalog differ")
        try:
            active_targets = catalog.validate_active_targets(stack.entries)
        except TargetResolutionError as exc:
            raise EconomicsError(
                f"active reward families overlap or are incomplete: {exc}"
            ) from None
        if not active_targets:
            raise EconomicsError("every registered arena requires an active crown")
        by_target = {row.target_id: row for row in authority.standing_claims}
        if len(by_target) != len(authority.standing_claims) or set(by_target) != set(active_targets):
            raise EconomicsError("every active target requires exactly one standing claim")
        for target_id in active_targets:
            claim = by_target[target_id]
            contribution = stack.entries[target_id]
            if claim.arena_digest != stack.arena_digest:
                raise EconomicsError(f"standing claim for {target_id!r} names another arena")
            if (
                claim.target_spec_digest != catalog.target_spec_digest(target_id)
                or claim.target_spec_digest != contribution.target_spec_digest
                or claim.contribution_digest != contribution.digest
            ):
                raise EconomicsError(f"standing claim for {target_id!r} is stale or incompatible")
            history = standing_index.get(
                (claim.arena_digest, claim.target_id, claim.contribution_digest)
            )
            if history is None:
                raise EconomicsError(
                    f"standing claim for {target_id!r} has no accepted history"
                )
            if (
                history.hotkey != claim.hotkey
                or history.speedup_ppm != claim.speedup_ppm
                or history.crowned_block != claim.crowned_block
                or history.target_spec_digest != claim.target_spec_digest
                or history.retained_evidence_digest != claim.retained_evidence_digest
            ):
                raise EconomicsError(
                    f"standing claim for {target_id!r} differs from accepted history"
                )
    unknown_exclusions = set(policy.excluded_claim_digests) - {
        row.digest for row in earning
    }
    if unknown_exclusions:
        raise EconomicsError(
            "excluded claim digest is not a retained accepted crown"
        )
    family_credits: list[StandingFamilyCredit] = []
    standing_by_hotkey: dict[str, int] = {}
    for claim in sorted(
        earning, key=lambda row: (row.arena_digest, row.target_id, row.digest)
    ):
        if policy.excludes_accepted(claim):
            continue
        credit = claim.credit_at(context.current_block, policy)
        family_credits.append(
            StandingFamilyCredit(
                claim.arena_digest,
                claim.family_id,
                claim.target_id,
                claim.digest,
                claim.hotkey,
                credit,
            )
        )
        # Keep the claim and family audit on the original miner. If they
        # left the metagraph, burn this tick's allocated share to the
        # validator rather than hold the vector or re-slice other families.
        recipient = (
            claim.hotkey
            if claim.hotkey in eligible
            else context.validator_hotkey
        )
        standing_by_hotkey[recipient] = (
            standing_by_hotkey.get(recipient, 0) + credit
        )
    if not any(standing_by_hotkey.values()):
        raise EconomicsError(
            "all standing crown credit has decayed to zero or been excluded"
        )

    discoveries = tuple(discovery_claims)
    if any(type(row) is not DiscoveryBountyClaim for row in discoveries):
        raise EconomicsError("discovery claims are not exactly typed")
    if (
        len({row.proposal_digest for row in discoveries}) != len(discoveries)
        or len({row.retained_evidence_digest for row in discoveries}) != len(discoveries)
        or len({row.digest for row in discoveries}) != len(discoveries)
    ):
        raise EconomicsError("discovery claims are renewed or duplicated")
    live = []
    expired = []
    discovery_by_hotkey: dict[str, int] = {}
    for claim in sorted(discoveries, key=lambda row: row.digest):
        if claim.hotkey in policy.excluded_hotkeys:
            continue
        if claim.live_at(context.current_block, policy):
            live.append(
                DiscoveryBountyCredit(claim.digest, claim.hotkey, claim.bounty_units)
            )
            recipient = (
                claim.hotkey
                if claim.hotkey in eligible
                else context.validator_hotkey
            )
            discovery_by_hotkey[recipient] = (
                discovery_by_hotkey.get(recipient, 0) + claim.bounty_units
            )
        else:
            expired.append(claim.digest)

    discovery_pool = policy.discovery_pool_ppm if live else 0
    if live and discovery_pool == 0:
        raise EconomicsError("live discovery claims exist while bounties are disabled")
    standing_pool = WEIGHT_PPM - discovery_pool
    combined = _allocate_pool(standing_by_hotkey, standing_pool)
    if live:
        for hotkey, value in _allocate_pool(discovery_by_hotkey, discovery_pool).items():
            combined[hotkey] = combined.get(hotkey, 0) + value
    weights = tuple(
        HotkeyWeight(hotkey, combined[hotkey]) for hotkey in sorted(combined)
    )
    return GlobalRewardProjection(
        policy.digest,
        context,
        tuple(sorted(row.digest for row in authorities)),
        tuple(family_credits),
        tuple(live),
        tuple(expired),
        weights,
    )


__all__ = [
    "AcceptedRewardClaim",
    "ArenaRewardAuthority",
    "CREDIT_SCALE",
    "DiscoveryBountyClaim",
    "EconomicsError",
    "EmissionsPolicyManifest",
    "GlobalRewardProjection",
    "GlobalRewardProjectionContext",
    "HotkeyWeight",
    "MetagraphMember",
    "POLICY_SCHEMA_VERSION",
    "POLICY_VERSION",
    "StandingRewardClaim",
    "WEIGHT_PPM",
    "decay_factor",
    "improvement_units",
    "project_global_rewards",
    "time_multiplier",
]
