"""Control-plane reconciliation for one content-addressed global weight vector.

This module owns no database and imports no evaluation runtime. The settlement
store supplies a compare-and-swap journal implementation; only the control-plane
signer supplies a wallet. An SDK return value is never confirmation authority.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, replace
from typing import Protocol

from optima import chain
from optima.stack_identity import canonical_digest, require_sha256_hex


WEIGHT_PARTS = 1_000_000
PUBLICATION_STATUSES = frozenset({"intent", "pending", "held", "confirmed"})


class WeightPublicationError(RuntimeError):
    """A projection, journal transition, or signer identity is unsafe."""

    validator_fault = True
    retryable = False


@dataclass(frozen=True)
class WeightProjection:
    """Exact settlement output accepted by the single control-plane signer."""

    chain_scope_digest: str
    netuid: int
    validator_hotkey: str
    policy_digest: str
    settlement_state_digest: str
    evaluation_state_digest: str
    stack_generation: int
    effective_block: int
    crown_count: int
    evidence_digests: tuple[str, ...]
    weights_ppm: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for field in (
            "chain_scope_digest",
            "policy_digest",
            "settlement_state_digest",
            "evaluation_state_digest",
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
        if (
            evidence != tuple(sorted(set(evidence)))
            or any(require_sha256_hex(value, field="evidence_digest") != value for value in evidence)
            or self.crown_count > len(evidence)
        ):
            raise WeightPublicationError("projection evidence inventory is malformed")
        object.__setattr__(self, "evidence_digests", evidence)
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
        return {
            "chain_scope_digest": self.chain_scope_digest,
            "crown_count": self.crown_count,
            "effective_block": self.effective_block,
            "evaluation_state_digest": self.evaluation_state_digest,
            "evidence_digests": list(self.evidence_digests),
            "netuid": self.netuid,
            "policy_digest": self.policy_digest,
            "settlement_state_digest": self.settlement_state_digest,
            "stack_generation": self.stack_generation,
            "validator_hotkey": self.validator_hotkey,
            "weights_ppm": [list(row) for row in self.weights_ppm],
        }

    @classmethod
    def from_dict(cls, value: object) -> "WeightProjection":
        fields = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != fields:
            raise WeightPublicationError("weight projection fields do not match")
        if type(value["evidence_digests"]) is not list or type(value["weights_ppm"]) is not list:
            raise WeightPublicationError("weight projection arrays are malformed")
        rows = value["weights_ppm"]
        if any(type(row) is not list or len(row) != 2 for row in rows):
            raise WeightPublicationError("weight projection rows are malformed")
        return cls(
            **{
                **value,
                "evidence_digests": tuple(value["evidence_digests"]),
                "weights_ppm": tuple(tuple(row) for row in rows),
            }
        )  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.chain.weight-projection", self.to_dict())


@dataclass(frozen=True)
class WeightPublicationRecord:
    """One immutable event in the injected publication journal."""

    projection_digest: str
    status: str
    prior_record_digest: str | None = None
    submit_block: int = 0
    retry_after_block: int = 0
    reveal_round: int = 0
    confirmed_block: int = 0
    confirmed_last_update: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "projection_digest",
            require_sha256_hex(self.projection_digest, field="projection_digest"),
        )
        if self.prior_record_digest is not None:
            object.__setattr__(
                self,
                "prior_record_digest",
                require_sha256_hex(
                    self.prior_record_digest, field="prior_record_digest"
                ),
            )
        if self.status not in PUBLICATION_STATUSES:
            raise WeightPublicationError("publication status is unsupported")
        for field in (
            "submit_block",
            "retry_after_block",
            "reveal_round",
            "confirmed_block",
            "confirmed_last_update",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise WeightPublicationError(f"publication {field} is malformed")
        if (
            not isinstance(self.reason, str)
            or len(self.reason) > 2_048
            or any(char in self.reason for char in "\x00\r\n")
        ):
            raise WeightPublicationError("publication reason is malformed")
        if self.status in {"intent", "pending"} and (
            self.submit_block <= 0 or self.retry_after_block < self.submit_block
        ):
            raise WeightPublicationError("in-flight publication bounds are malformed")
        if self.status == "held" and not (
            (self.submit_block == 0 and self.retry_after_block == 0)
            or (
                self.submit_block > 0
                and self.retry_after_block >= self.submit_block
            )
        ):
            raise WeightPublicationError("held publication bounds are malformed")
        if self.status == "confirmed" and (
            self.confirmed_block < self.confirmed_last_update
            or (
                self.submit_block > 0
                and self.confirmed_last_update < self.submit_block
            )
        ):
            raise WeightPublicationError("confirmation chronology is malformed")

    def to_dict(self) -> dict[str, object]:
        return {
            "confirmed_block": self.confirmed_block,
            "confirmed_last_update": self.confirmed_last_update,
            "prior_record_digest": self.prior_record_digest,
            "projection_digest": self.projection_digest,
            "reason": self.reason,
            "retry_after_block": self.retry_after_block,
            "reveal_round": self.reveal_round,
            "status": self.status,
            "submit_block": self.submit_block,
        }

    @classmethod
    def from_dict(cls, value: object) -> "WeightPublicationRecord":
        fields = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != fields:
            raise WeightPublicationError("publication record fields do not match")
        return cls(**value)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.chain.weight-publication", self.to_dict())


class WeightPublicationJournal(Protocol):
    """Transactional storage seam implemented by the settlement authority."""

    def load(self) -> WeightPublicationRecord | None: ...

    def compare_and_swap(
        self,
        expected_record_digest: str | None,
        replacement: WeightPublicationRecord,
    ) -> None: ...


@dataclass(frozen=True)
class WeightPublicationResult:
    projection_digest: str
    status: str
    record: WeightPublicationRecord | None
    chain_matches: bool
    submitted: bool
    dry_run: bool
    observed_block: int

    def __post_init__(self) -> None:
        require_sha256_hex(self.projection_digest, field="projection_digest")
        if (
            self.status not in PUBLICATION_STATUSES | {"dry_run"}
            or (self.record is not None and type(self.record) is not WeightPublicationRecord)
            or any(type(value) is not bool for value in (self.chain_matches, self.submitted, self.dry_run))
            or type(self.observed_block) is not int
            or self.observed_block < 0
            or self.dry_run != (self.status == "dry_run")
            or (self.status == "dry_run") != (self.record is None)
            or (
                self.record is not None
                and self.status != self.record.status
            )
        ):
            raise WeightPublicationError("publication result status is unsupported")


def _block(subtensor) -> int:
    try:
        value = operator.index(subtensor.get_current_block())
    except (AttributeError, TypeError, OverflowError, ValueError) as exc:
        raise chain.ChainWeightStateError(f"cannot read current chain block: {exc}") from None
    if value < 0:
        raise chain.ChainWeightStateError("current chain block is negative")
    return value


def _matches(snapshot: chain.ValidatorWeightSnapshot, projection: WeightProjection) -> bool:
    expected = projection.weights
    return set(snapshot.weights) == set(expected) and all(
        math.isclose(
            float(snapshot.weights[hotkey]), expected[hotkey],
            rel_tol=2e-5, abs_tol=2e-5,
        )
        for hotkey in expected
    )


def _reveal_round(result: object) -> int:
    if not isinstance(result, dict):
        return 0
    response = result.get("result")
    data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
    value = data.get("reveal_round") if isinstance(data, dict) else None
    return value if type(value) is int and value >= 0 else 0


def _advance(
    journal: WeightPublicationJournal,
    current: WeightPublicationRecord | None,
    replacement: WeightPublicationRecord,
) -> WeightPublicationRecord:
    expected = current.digest if current is not None else None
    if replacement.prior_record_digest != expected:
        replacement = replace(replacement, prior_record_digest=expected)
    journal.compare_and_swap(expected, replacement)
    return replacement


def _held(
    journal: WeightPublicationJournal,
    current: WeightPublicationRecord,
    reason: str,
) -> WeightPublicationRecord:
    return _advance(
        journal,
        current,
        WeightPublicationRecord(
            projection_digest=current.projection_digest,
            status="held",
            submit_block=current.submit_block,
            retry_after_block=current.retry_after_block,
            reveal_round=current.reveal_round,
            confirmed_block=current.confirmed_block,
            confirmed_last_update=current.confirmed_last_update,
            reason=reason,
        ),
    )


def reconcile_weight_publication(
    subtensor,
    signer_wallet,
    projection: WeightProjection,
    journal: WeightPublicationJournal,
    *,
    refresh_blocks: int,
    dry_run: bool = False,
) -> WeightPublicationResult:
    """Reconcile and optionally publish one exact projection.

    The live sparse row is read before any journal decision and after every SDK
    attempt. Real publication persists ``intent`` before the signer is called;
    exact readback plus a sufficiently new ``last_update`` is the only path to
    ``confirmed``.
    """

    if type(projection) is not WeightProjection:
        raise WeightPublicationError("weight projection is not exactly typed")
    if type(refresh_blocks) is not int or refresh_blocks <= 0:
        raise WeightPublicationError("weight refresh cadence is malformed")
    pre = chain.read_validator_weight_snapshot(
        subtensor, projection.netuid, projection.validator_hotkey
    )
    observed_block = _block(subtensor)
    if pre.last_update_block > observed_block or projection.effective_block > observed_block:
        raise chain.ChainWeightStateError("weight authority chronology is inconsistent")
    matches = _matches(pre, projection)

    if dry_run:
        chain.set_weights(
            subtensor, None, projection.netuid, projection.weights, dry_run=True
        )
        return WeightPublicationResult(
            projection.digest, "dry_run", None, matches, False, True, observed_block
        )

    current = journal.load()
    if current is not None and type(current) is not WeightPublicationRecord:
        raise WeightPublicationError("journal returned an untyped publication record")
    if current is not None and current.status in {"intent", "pending", "held"}:
        if current.projection_digest != projection.digest:
            current = _held(journal, current, "projection_changed_while_unresolved")
            return WeightPublicationResult(
                projection.digest, "held", current, matches, False, False, observed_block
            )
        if current.status == "held":
            return WeightPublicationResult(
                projection.digest, "held", current, matches, False, False, observed_block
            )
        if matches and pre.last_update_block >= current.submit_block:
            current = _advance(
                journal,
                current,
                WeightPublicationRecord(
                    projection.digest,
                    "confirmed",
                    submit_block=current.submit_block,
                    retry_after_block=current.retry_after_block,
                    reveal_round=current.reveal_round,
                    confirmed_block=observed_block,
                    confirmed_last_update=pre.last_update_block,
                    reason="authoritative_readback",
                ),
            )
            return WeightPublicationResult(
                projection.digest, "confirmed", current, True, False, False, observed_block
            )
        if observed_block < current.retry_after_block:
            return WeightPublicationResult(
                projection.digest, current.status, current, False, False, False, observed_block
            )
        current = _held(journal, current, "publication_readback_deadline_expired")
        return WeightPublicationResult(
            projection.digest, "held", current, False, False, False, observed_block
        )

    if current is not None and current.projection_digest == projection.digest:
        if not matches:
            current = _held(journal, current, "confirmed_vector_changed_on_chain")
            return WeightPublicationResult(
                projection.digest, "held", current, False, False, False, observed_block
            )
        if observed_block - pre.last_update_block < refresh_blocks:
            return WeightPublicationResult(
                projection.digest, "confirmed", current, True, False, False, observed_block
            )

    if current is None and matches:
        current = _advance(
            journal,
            None,
            WeightPublicationRecord(
                projection.digest,
                "confirmed",
                confirmed_block=observed_block,
                confirmed_last_update=pre.last_update_block,
                reason="preexisting_authoritative_readback",
            ),
        )
        return WeightPublicationResult(
            projection.digest, "confirmed", current, True, False, False, observed_block
        )

    if projection.crown_count <= 0:
        raise WeightPublicationError("real weight submission requires a current crown")
    try:
        wallet_hotkey = signer_wallet.hotkey.ss58_address
    except AttributeError:
        raise WeightPublicationError("real weight submission requires a signer wallet") from None
    if wallet_hotkey != projection.validator_hotkey:
        raise WeightPublicationError("signer wallet differs from projection authority")
    if observed_block <= 0:
        raise chain.ChainWeightStateError("real publication requires a positive chain block")

    intent = _advance(
        journal,
        current,
        WeightPublicationRecord(
            projection.digest,
            "intent",
            submit_block=observed_block,
            retry_after_block=observed_block + refresh_blocks,
            reason="before_sdk_submission",
        ),
    )
    submitted = False
    response: object = None
    sdk_reason = "sdk_result_unconfirmed"
    try:
        response = chain.set_weights(
            subtensor,
            signer_wallet,
            projection.netuid,
            projection.weights,
            dry_run=False,
        )
        submitted = bool(response.get("submitted")) if isinstance(response, dict) else False
    except Exception as exc:
        sdk_reason = f"sdk_exception:{type(exc).__name__}"
    pending = _advance(
        journal,
        intent,
        WeightPublicationRecord(
            projection.digest,
            "pending",
            submit_block=observed_block,
            retry_after_block=observed_block + refresh_blocks,
            reveal_round=_reveal_round(response),
            reason=sdk_reason,
        ),
    )
    try:
        post = chain.read_validator_weight_snapshot(
            subtensor, projection.netuid, projection.validator_hotkey
        )
        post_block = _block(subtensor)
        if post.last_update_block > post_block or post_block < observed_block:
            raise chain.ChainWeightStateError("post-submit chronology is inconsistent")
    except chain.ChainWeightStateError:
        held = _held(journal, pending, "post_submit_authority_unavailable")
        return WeightPublicationResult(
            projection.digest, "held", held, False, submitted, False, observed_block
        )
    post_matches = _matches(post, projection)
    if post_matches and post.last_update_block >= observed_block:
        confirmed = _advance(
            journal,
            pending,
            WeightPublicationRecord(
                projection.digest,
                "confirmed",
                submit_block=observed_block,
                retry_after_block=observed_block + refresh_blocks,
                reveal_round=pending.reveal_round,
                confirmed_block=post_block,
                confirmed_last_update=post.last_update_block,
                reason="post_submit_authoritative_readback",
            ),
        )
        return WeightPublicationResult(
            projection.digest, "confirmed", confirmed, True, submitted, False, post_block
        )
    return WeightPublicationResult(
        projection.digest, "pending", pending, False, submitted, False, post_block
    )


__all__ = [
    "PUBLICATION_STATUSES",
    "WEIGHT_PARTS",
    "WeightProjection",
    "WeightPublicationError",
    "WeightPublicationJournal",
    "WeightPublicationRecord",
    "WeightPublicationResult",
    "reconcile_weight_publication",
]
