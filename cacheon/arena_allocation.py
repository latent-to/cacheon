"""Manual arena settings and immutable submission-time reward terms."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import re

from cacheon.stack_identity import canonical_digest

PPM = 1_000_000


def normalize_weights(weights: dict[str, int]) -> dict[str, int]:
    """Scale down only overcommitted settings, with deterministic ppm rounding."""
    from cacheon.economics import _allocate_pool

    if not weights or any(type(v) is not int or v < 0 for v in weights.values()):
        raise ValueError("arena weights must be nonnegative integer ppm")
    if sum(weights.values()) <= PPM:
        return dict(weights)
    scaled = _allocate_pool(weights, PPM)
    return {key: scaled.get(key, 0) for key in weights}


@dataclass(frozen=True)
class ArenaAllocation:
    """An append-only schedule; source names group historical model generations."""

    activation_block: int
    burn_hotkey: str
    sources: tuple[tuple[str, str], ...]
    history: tuple[tuple[int, tuple[tuple[str, int], ...], int], ...]

    @classmethod
    def from_dict(cls, value: object) -> "ArenaAllocation":
        """Read explicit source authorities and full historical settings."""
        if type(value) is not dict or set(value) != {
            "activation_block", "burn_hotkey", "sources", "history",
        }:
            raise ValueError("arena allocation fields do not match")
        activation, burn, sources, history = (value[k] for k in (
            "activation_block", "burn_hotkey", "sources", "history"))
        if type(activation) is not int or activation <= 0:
            raise ValueError("allocation activation block must be positive")
        if type(burn) is not str or not burn or any(c.isspace() for c in burn):
            raise ValueError("allocation burn hotkey is malformed")
        if type(sources) is not dict or not sources:
            raise ValueError("allocation requires explicit sources")
        for key, path in sources.items():
            if (type(key) is not str or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", key)
                    or type(path) is not str or not Path(path).is_absolute()):
                raise ValueError("allocation source name or config path is malformed")
        if type(history) is not list or not history:
            raise ValueError("allocation history is empty")
        rows = []
        for row in history:
            if (type(row) is not dict or set(row) not in (
                    {"from_block", "weights_ppm"}, {"from_block", "weights_ppm", "stall_bonus_ppm"})):
                raise ValueError("allocation history fields do not match")
            block, weights = row["from_block"], row["weights_ppm"]
            if (type(block) is not int or block < 0 or type(weights) is not dict
                    or set(weights) != set(sources)):
                raise ValueError("allocation version must name every source")
            normalize_weights(weights)
            bonus = row.get("stall_bonus_ppm", PPM)
            if type(bonus) is not int or not 0 <= bonus <= PPM or (block < activation and bonus != PPM):
                raise ValueError("stall bonus must be integer ppm, with full strength before activation")
            rows.append((block, tuple(sorted(weights.items())), bonus))
        blocks = [block for block, _, _ in rows]
        if blocks != sorted(set(blocks)) or blocks[0] != 0 or activation not in blocks:
            raise ValueError("allocation history needs baseline zero and ordered activation")
        return cls(activation, burn, tuple(sorted(sources.items())), tuple(rows))

    def to_dict(self) -> dict:
        """Preserve raw settings; their normalized terms are derived deterministically."""
        return {"activation_block": self.activation_block, "burn_hotkey": self.burn_hotkey,
                "sources": dict(self.sources), "history": [
                    {"from_block": block, "weights_ppm": dict(weights),
                     **({"stall_bonus_ppm": bonus} if bonus != PPM else {})}
                    for block, weights, bonus in self.history]}

    def terms_at(self, submission_block: int) -> dict[str, int]:
        """Use finalized arrival, never qualification time or the current settings."""
        return normalize_weights(dict(self._version_at(submission_block)[1]))

    def stall_bonus_at(self, submission_block: int) -> int:
        """Freeze the waiting-bonus strength at the same finalized arrival boundary."""
        return self._version_at(submission_block)[2]

    def _version_at(self, submission_block):
        if type(submission_block) is not int or submission_block < 0:
            raise ValueError("submission block is malformed")
        return next(row for row in reversed(self.history) if row[0] <= submission_block)

    @property
    def digest(self) -> str:
        return canonical_digest("cacheon.arena-allocation.v1", self.to_dict())


def arena_base_credits(claims, policy, context, decay_start_blocks):
    """Keep speed and publication decay, excluding the arena's stall bonus."""
    return {claim.digest: claim.credit_at(
        context.current_block, policy, predecessor_block=claim.crowned_block,
        decay_start_block=(context.current_block if decay_start_blocks.get(claim.digest, -1) is None
                           else decay_start_blocks.get(claim.digest)),
    ) for claim in claims}


def allocate_submission_weights(credits, terms, context, burn_hotkey, *, base_credits):
    """Price arena pools without stall, then split each pool with stall credit.

    Zero-offer submissions neither earn nor dilute later paid submissions.
    Frozen terms survive new-winner dilution in the existing shared weight pool.
    """
    from cacheon.economics import EconomicsError, _allocate_pool

    if (set(terms) != {row.claim_digest for row in credits} or set(base_credits) != set(terms)
            or burn_hotkey not in context.eligible_hotkeys):
        raise EconomicsError("allocation terms or burn registration are incomplete")
    totals, priced, internal = {}, {}, {}
    for row in credits:
        source, ppm = terms[row.claim_digest]
        base = base_credits[row.claim_digest]
        if (type(source) is not str or not source or type(ppm) is not int or not 0 <= ppm <= PPM
                or type(base) is not int or base < 0):
            raise EconomicsError("submission allocation terms are malformed")
        if ppm:
            totals[source] = totals.get(source, 0) + base
            priced[source] = priced.get(source, 0) + ppm * base
            internal[source] = internal.get(source, 0) + ppm * row.credit
    requested, recipients, owners = {}, {}, {}
    for row in credits:
        source, ppm = terms[row.claim_digest]
        if not ppm or not row.credit or not totals[source]:
            continue
        pool = Fraction(priced[source], totals[source])
        requested[row.claim_digest] = pool * Fraction(ppm * row.credit, internal[source])
        recipients[row.claim_digest] = (row.hotkey if row.hotkey in context.eligible_hotkeys
                                       else context.validator_hotkey)
        owners[row.claim_digest] = source
    requested[""] = max(Fraction(0), PPM - sum(requested.values()))
    recipients[""] = burn_hotkey
    by_hotkey = {}
    for key, share in requested.items():
        hotkey = recipients[key]
        by_hotkey[hotkey] = by_hotkey.get(hotkey, Fraction(0)) + share
    weights = _allocate_pool(by_hotkey, PPM)
    arena_shares, rewarded, burned = {}, [], 0
    for hotkey, weight in weights.items():
        parts = {key: share for key, share in requested.items() if recipients[key] == hotkey}
        for key, ppm in _allocate_pool(parts, weight).items():
            if key:
                source = owners[key]
                arena_shares[source] = arena_shares.get(source, 0) + ppm
                if ppm:
                    rewarded.append(key)
            else:
                burned += ppm
    return weights, arena_shares, tuple(sorted(rewarded)), burned
