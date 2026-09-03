"""Pure, content-addressed emissions projection over reopened PASS evidence."""

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
POLICY_VERSION = "cacheon.emissions.v1.5"
WEIGHT_PPM = 1_000_000
CREDIT_SCALE = 1_000_000_000_000
STALL_SCALE_BLOCKS = 1_800
_MATH_CONTEXT = Context(prec=50, rounding=ROUND_FLOOR)
_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}\Z")
_HOTKEY = re.compile(r"[^\s]{1,256}\Z")


class EconomicsError(ValueError):
    pass
def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=EconomicsError)


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    return require_int(value, field=field, error=EconomicsError, minimum=minimum)


def _hotkey(value: object, field: str = "hotkey") -> str:
    if not isinstance(value, str) or _HOTKEY.fullmatch(value) is None:
        raise EconomicsError(f"{field} is malformed")
    return value


def _strict(value: object, fields: set[str], name: str) -> dict[str, object]:
    return require_exact_fields(
        value, fields=frozenset(fields), label=name, error=EconomicsError, exact_dict=True
    )


@dataclass(frozen=True)
class EmissionsPolicyManifest:
    half_life_blocks: int
    discovery_lifetime_blocks: int
    discovery_pool_ppm: int
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
        if self.discovery_pool_ppm >= WEIGHT_PPM:
            raise EconomicsError("discovery_pool_ppm must leave standing reward capacity")
        if self.schema_version != POLICY_SCHEMA_VERSION:
            raise EconomicsError("emissions policy schema_version is unsupported")
        if self.policy_version != POLICY_VERSION:
            raise EconomicsError("emissions policy_version is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "discovery_lifetime_blocks": self.discovery_lifetime_blocks,
            "discovery_pool_ppm": self.discovery_pool_ppm,
            "half_life_blocks": self.half_life_blocks,
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EmissionsPolicyManifest":
        row = _strict(
            value,
            {
                "discovery_lifetime_blocks",
                "discovery_pool_ppm",
                "half_life_blocks",
                "policy_version",
                "schema_version",
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

    def credit_at(
        self,
        block: int,
        policy: EmissionsPolicyManifest,
        *,
        predecessor_block: int | None = None,
    ) -> int:
        predecessor = self.crowned_block if predecessor_block is None else predecessor_block
        _integer(block, "credit block", minimum=self.crowned_block)
        _integer(predecessor, "predecessor_block")
        if predecessor > self.crowned_block:
            raise EconomicsError("reward predates its arena stall clock")
        with localcontext(_MATH_CONTEXT):
            credit = (
                (Decimal(self.speedup_ppm) / WEIGHT_PPM).ln()
                * (Decimal(1) + (Decimal(self.crowned_block - predecessor) / STALL_SCALE_BLOCKS).sqrt())
                * Decimal(2) ** (-Decimal(block - self.crowned_block) / policy.half_life_blocks)
                * CREDIT_SCALE
            )
        return int(credit.to_integral_value(rounding=ROUND_FLOOR))

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
        if (
            len({row.hotkey for row in self.weights}) != len(self.weights)
            or any(row.weight_ppm <= 0 for row in self.weights)
            or sum(row.weight_ppm for row in self.weights) != WEIGHT_PPM
        ):
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
    earning_claims: Iterable[StandingRewardClaim],
    discovery_claims: Iterable[DiscoveryBountyClaim] = (),
) -> GlobalRewardProjection:
    """Pool every retained two-PASS contribution before one indivisible vector."""

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
    if any(type(row) is not StandingRewardClaim for row in earning):
        raise EconomicsError("PASS reward claims are not exactly typed")
    if len({row.digest for row in earning}) != len(earning):
        raise EconomicsError("PASS reward claims are duplicated")
    eligible = context.eligible_hotkeys
    standing_index: dict[tuple[str, str, str], StandingRewardClaim] = {}
    for claim in earning:
        key = (claim.arena_digest, claim.target_id, claim.contribution_digest)
        if key in standing_index:
            raise EconomicsError("PASS reward claims reuse a contribution")
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
                raise EconomicsError(f"standing claim for {target_id!r} has no PASS")
            if (
                history.hotkey != claim.hotkey
                or history.speedup_ppm != claim.speedup_ppm
                or history.target_spec_digest != claim.target_spec_digest
            ):
                raise EconomicsError(f"standing claim for {target_id!r} differs from PASS")
    family_credits: list[StandingFamilyCredit] = []
    standing_by_hotkey: dict[str, int] = {}
    previous: dict[str, int] = {}
    for claim in sorted(
        earning, key=lambda row: (row.arena_digest, row.crowned_block, row.digest)
    ):
        predecessor = previous.get(claim.arena_digest, claim.crowned_block)
        credit = claim.credit_at(
            context.current_block, policy, predecessor_block=predecessor
        )
        previous[claim.arena_digest] = claim.crowned_block
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
        recipient = (
            claim.hotkey
            if claim.hotkey in eligible
            else context.validator_hotkey
        )
        standing_by_hotkey[recipient] = (
            standing_by_hotkey.get(recipient, 0) + credit
        )
    if not any(standing_by_hotkey.values()):
        raise EconomicsError("all PASS credit has decayed to zero")

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
        HotkeyWeight(hotkey, combined[hotkey])
        for hotkey in sorted(combined)
        if combined[hotkey] > 0
    )
    return GlobalRewardProjection(
        policy.digest,
        context,
        tuple(sorted(row.digest for row in authorities)),
        tuple(
            sorted(
                family_credits,
                key=lambda row: (row.arena_digest, row.target_id, row.claim_digest),
            )
        ),
        tuple(live),
        tuple(expired),
        weights,
    )


__all__ = [
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
    "STALL_SCALE_BLOCKS",
    "StandingRewardClaim",
    "WEIGHT_PPM",
    "project_global_rewards",
]
