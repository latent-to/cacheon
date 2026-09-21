"""The existing weight journal wire record, shared by its producer and follower."""

from __future__ import annotations

from dataclasses import dataclass

from cacheon.stack_identity import canonical_digest, require_sha256_hex

_WEIGHT_PUBLICATION_DOMAIN = "cacheon.chain.weight-publication"
PUBLICATION_STATUSES = frozenset(
    {"intent", "pending", "held", "confirmed", "released"}
)


class WeightPublicationError(RuntimeError):
    """A projection, journal transition, or signer identity is unsafe."""

    validator_fault = True
    retryable = False


@dataclass(frozen=True)
class WeightPublicationRecord:
    """One immutable event in the injected publication journal.

    Included submissions may precede active-weight readback. Reopen the follower's
    block_inclusion records without treating last_update=0 as corrupt chronology.
    """

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
            or self.confirmed_block < self.submit_block
            or (
                self.submit_block > 0
                and self.reason != "block_inclusion"
                and self.confirmed_last_update < self.submit_block
            )
        ):
            raise WeightPublicationError("confirmation chronology is malformed")
        if self.status == "released" and not self.reason:
            raise WeightPublicationError("publication release requires an audit reason")

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
        return canonical_digest(_WEIGHT_PUBLICATION_DOMAIN, self.to_dict())
