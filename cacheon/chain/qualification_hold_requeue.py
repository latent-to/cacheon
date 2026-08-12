"""Bounded automatic re-entry for remote qualification HOLDs.

A remote qualification HOLD is not a candidate verdict: it records that the
stage could not produce PASS/FAIL evidence (worker infrastructure loss,
missing graph evidence, transport interruption).  Before 2026-08-12 those rows
parked in ``held`` until an operator released them by hand — on an autonomous
validator that converts every novel infrastructure fault into a permanent
wedge.  This policy demotes ``held`` from an operator-touch state to a
bounded retry state:

- every committed remote HOLD releases its reservations straight back to FIFO
  (fresh claim, fresh request, real re-execution) until one reservation has
  burned ``max_attempts`` total attempts; then it stays held and alarms;
- ``breaker_threshold`` consecutive holds with an identical reason and no
  PASS/FAIL terminal between them open a circuit breaker: fresh qualification
  claims stop and every suppression alarms, because N identical
  infrastructure holds in a row are one systemic fault burning the whole
  queue, not N candidate problems.

All state is in-memory by design.  A supervisor restart closes the breaker and
forgets attempt counts — restarting is the operator's explicit "cause fixed"
signal, and the error direction is re-running paid work at most once more per
restart, never wedging the queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BREAKER_THRESHOLD = 5
_SUPPRESSED_CLAIM_REMINDER_EVERY = 60


class QualificationHoldRequeueError(RuntimeError):
    """The hold-requeue policy is malformed."""


class _ReleasesHolds(Protocol):
    def release_hold(self, reservation_id: str, *, reason: str) -> object: ...


def _alarm(line: str) -> None:
    import sys

    print(line, flush=True, file=sys.stdout)


@dataclass
class QualificationHoldRequeuePolicy:
    """In-memory bounded re-entry + same-reason circuit breaker."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD
    _attempts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _streak_reason: str | None = field(default=None, init=False, repr=False)
    _streak: int = field(default=0, init=False, repr=False)
    _suppressed_claims: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("max_attempts", "breaker_threshold"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 1:
                raise QualificationHoldRequeueError(
                    f"{name} must be a positive integer"
                )

    @property
    def breaker_open(self) -> bool:
        return self._streak >= self.breaker_threshold

    def note_terminal(self) -> None:
        """A PASS/FAIL terminal proves the stage works; forget the streak."""

        self._streak_reason = None
        self._streak = 0
        self._suppressed_claims = 0

    def refuse_fresh_claim(self) -> bool:
        """True when the breaker is open and fresh claims must not start."""

        if not self.breaker_open:
            return False
        self._suppressed_claims += 1
        if (
            self._suppressed_claims == 1
            or self._suppressed_claims % _SUPPRESSED_CLAIM_REMINDER_EVERY == 0
        ):
            _alarm(
                "QUALIFICATION-HOLD-BREAKER-OPEN: "
                f"{self._streak} consecutive holds with reason "
                f"{self._streak_reason!r}; refusing fresh qualification claims "
                f"(suppressed={self._suppressed_claims}). Fix the cause, then "
                "restart the standing supervisor to close the breaker."
            )
        return True

    def after_hold(
        self,
        store: _ReleasesHolds,
        *,
        reservation_ids: tuple[str, ...],
        reason: str,
    ) -> tuple[str, ...]:
        """Release the held reservations back to FIFO within their budget.

        Called with the same open store session that just committed the hold,
        immediately after the commit.  A release failure is alarmed and leaves
        that row held — failing toward held never loses evidence or work.
        """

        if reason == self._streak_reason:
            self._streak += 1
        else:
            self._streak_reason = reason
            self._streak = 1
        if self.breaker_open:
            _alarm(
                "QUALIFICATION-HOLD-BREAKER-OPEN: "
                f"{self._streak} consecutive holds with reason {reason!r}; "
                f"leaving {len(reservation_ids)} reservation(s) held. Fix the "
                "cause, release held rows, then restart the standing "
                "supervisor to close the breaker."
            )
            return ()
        released: list[str] = []
        for reservation_id in reservation_ids:
            burned = self._attempts.get(reservation_id, 0) + 1
            self._attempts[reservation_id] = burned
            if burned >= self.max_attempts:
                _alarm(
                    "QUALIFICATION-HOLD-CAP: reservation "
                    f"{reservation_id} burned {burned} attempts "
                    f"(cap {self.max_attempts}), latest reason {reason!r}; "
                    "staying held for operator review."
                )
                continue
            release_reason = (
                f"auto_requeue_attempt_{burned + 1}_of_{self.max_attempts}:{reason}"
            )
            try:
                store.release_hold(reservation_id, reason=release_reason)
            except Exception as exc:  # noqa: BLE001 - alarmed, held is safe
                _alarm(
                    "QUALIFICATION-AUTO-REQUEUE-RELEASE-FAILED: reservation "
                    f"{reservation_id} stays held: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            released.append(reservation_id)
            _alarm(
                "QUALIFICATION-AUTO-REQUEUE: reservation "
                f"{reservation_id} back to FIFO for attempt {burned + 1} of "
                f"{self.max_attempts} after {reason!r}."
            )
        return tuple(released)
