"""Bounded retry policy for remote qualification HOLDs.

Only retryable infrastructure failures re-enter FIFO. Deterministic failures
such as GPU OOM stay held after one attempt. Distinct reservations exhausting
the same systemic reason open the circuit breaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BREAKER_THRESHOLD = 5
_SUPPRESSED_CLAIM_REMINDER_EVERY = 60

# Bundle-class limitations must not stop unrelated swappable submissions.
NON_SYSTEMIC_HOLD_REASONS = frozenset(
    {"remote_qualification_hold:legacy_no_decision"}
)


class QualificationHoldRequeueError(RuntimeError):
    """The hold-requeue policy is malformed."""


class _ReleasesHolds(Protocol):
    def release_hold(self, reservation_id: str, *, reason: str) -> object: ...
    def mark_hold_retry_exhausted(self, reservation_id: str) -> object: ...


class _ListsParkedHolds(_ReleasesHolds, Protocol):
    def auto_requeueable_holds(self, *, limit: int = ...) -> tuple[str, ...]: ...


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
    _exhausted: set[str] = field(default_factory=set, init=False, repr=False)
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
        return len(self._exhausted) >= self.breaker_threshold

    def note_terminal(self) -> None:
        """A PASS/FAIL terminal proves the stage works; forget the streak."""

        self._streak_reason = None
        self._exhausted.clear()
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
                f"{len(self._exhausted)} reservations exhausted their retry "
                f"budget with reason {self._streak_reason!r}; refusing fresh "
                f"qualification claims (suppressed={self._suppressed_claims}). "
                "Fix the cause, then restart the standing supervisor to close "
                "the breaker."
            )
        return True

    def _note_exhausted(
        self,
        store: _ReleasesHolds,
        reservation_id: str,
        reason: str,
        burned: int,
    ) -> None:
        """Record one reservation as stuck: durably, and once, toward the breaker.

        Durably, because attempt counts live in memory and a restart forgets
        them; without the mark, start-time reconciliation rehydrates a spent
        budget on every respawn.  Once, because a single reservation's own
        retries are not independent evidence of a systemic fault.
        """

        systemic = reason not in NON_SYSTEMIC_HOLD_REASONS
        if systemic:
            if reason != self._streak_reason:
                self._streak_reason = reason
                self._exhausted.clear()
            self._exhausted.add(reservation_id)
        try:
            store.mark_hold_retry_exhausted(reservation_id)
        except Exception as exc:  # noqa: BLE001 - alarmed, held is safe
            _alarm(
                "QUALIFICATION-HOLD-EXHAUSTED-MARK-FAILED: reservation "
                f"{reservation_id} may be retried again after a restart: "
                f"{type(exc).__name__}: {exc}"
            )
        _alarm(
            "QUALIFICATION-HOLD-CAP: reservation "
            f"{reservation_id} burned {burned} attempts "
            f"(cap {self.max_attempts}), latest reason {reason!r}; "
            "staying held for operator review"
            + ("." if systemic else "; not counted toward the systemic breaker.")
        )
        if self.breaker_open:
            _alarm(
                "QUALIFICATION-HOLD-BREAKER-OPEN: "
                f"{len(self._exhausted)} reservations exhausted their budget "
                f"with reason {reason!r}; refusing fresh qualification claims. "
                "Fix the cause, then restart the standing supervisor."
            )

    def reconcile_parked(self, store: _ListsParkedHolds) -> tuple[str, ...]:
        """Release evaluation holds parked before this lifetime back into FIFO.

        ``after_hold`` only reaches rows this process itself parked, so a hold
        committed by an earlier lifetime -- or by a lifetime that died between
        the commit and the release -- stays held forever and the queue never
        drains to zero.  A supervisor start is already this policy's "cause
        fixed" signal (it closes the breaker and forgets attempt counts), so it
        is also the right moment to reopen those rows on a fresh budget.  Each
        released row re-enters FIFO for a real re-execution; a row that parks
        again lands back here at the next restart, never silently.
        """

        parked = store.auto_requeueable_holds()
        if type(parked) is not tuple:
            raise QualificationHoldRequeueError(
                "parked hold listing is not exactly typed"
            )
        released: list[str] = []
        for reservation_id in parked:
            self._attempts.pop(reservation_id, None)
            try:
                store.release_hold(
                    reservation_id, reason="auto_requeue_reconciled_at_start"
                )
            except Exception as exc:  # noqa: BLE001 - alarmed, held is safe
                _alarm(
                    "QUALIFICATION-HOLD-RECONCILE-FAILED: reservation "
                    f"{reservation_id} stays held: {type(exc).__name__}: {exc}"
                )
                continue
            released.append(reservation_id)
        if released:
            _alarm(
                "QUALIFICATION-HOLD-RECONCILED: released "
                f"{len(released)} parked reservation(s) back to FIFO "
                f"of {len(parked)} eligible."
            )
        return tuple(released)

    def after_hold(
        self,
        store: _ReleasesHolds,
        *,
        reservation_ids: tuple[str, ...],
        reason: str,
        retryable: bool = True,
    ) -> tuple[str, ...]:
        """Release the held reservations back to FIFO within their budget.

        Called with the same open store session that just committed the hold,
        immediately after the commit.  A release failure is alarmed and leaves
        that row held — failing toward held never loses evidence or work.
        """

        if self.breaker_open:
            _alarm(
                "QUALIFICATION-HOLD-BREAKER-OPEN: "
                f"{len(self._exhausted)} reservations exhausted their budget "
                f"with reason {self._streak_reason!r}; leaving "
                f"{len(reservation_ids)} reservation(s) held. Fix the cause, "
                "then restart the standing supervisor to close the breaker."
            )
            return ()
        released: list[str] = []
        for reservation_id in reservation_ids:
            burned = self._attempts.get(reservation_id, 0) + 1
            self._attempts[reservation_id] = burned
            if not retryable:
                self._note_exhausted(store, reservation_id, reason, burned)
                continue
            if burned >= self.max_attempts:
                self._note_exhausted(store, reservation_id, reason, burned)
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
