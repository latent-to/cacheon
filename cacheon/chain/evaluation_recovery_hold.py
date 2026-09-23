"""Complete authenticated worker HOLDs and free their physical evaluation lane."""

from cacheon.chain.evaluation_leases import EvaluationLease
from cacheon.chain.evaluation_recovery import (
    EvaluationRecovery, RecoveryResolution, RecoveryPhase, RecoveryEventType,
)
from cacheon.stack_identity import require_sha256_hex


def _intake_error(message: str) -> RuntimeError:
    from cacheon.chain.intake import IntakeError
    return IntakeError(message)


class CompletedQualificationHoldMixin:
    """Preserve a request-bound HOLD while releasing only its exact worker lease."""

    def commit_remote_qualification_hold(
        self,
        recovery: EvaluationRecovery,
        *,
        current_block: int,
        result_digest: str,
        reason: str,
        reservation_ids: tuple[str, ...],
        lease_blocks: int = 30,
    ) -> EvaluationLease:
        """Terminalize one authenticated remote HOLD without blocking FIFO.

        The worker has returned a closed, request-bound HOLD product.  This is
        neither a candidate failure nor retry authority: retain it as a blank-
        decision reservation HOLD, complete its exact lease/recovery, and make
        the next promoted cohort claimable.  ``HELD`` input migrates products
        retained by the former campaign-halting behavior.
        """

        require_sha256_hex(result_digest, field="remote qualification HOLD digest")
        if (
            type(recovery) is not EvaluationRecovery
            or recovery.resolution is not RecoveryResolution.UNRESOLVED
            or recovery.phase not in {RecoveryPhase.RESULT_READY, RecoveryPhase.HELD}
            or not isinstance(reason, str)
            or not reason.startswith("remote_qualification_hold:")
            or reason.strip() != reason
            or len(reason) > 2_048
            or type(reservation_ids) is not tuple
            or reservation_ids != recovery.lease.reservation_ids
            or type(lease_blocks) is not int
            or lease_blocks <= 0
            or lease_blocks > self.policy.expiry_blocks
        ):
            raise _intake_error("remote qualification HOLD completion is malformed")
        self._require_evaluation_clock(current_block)
        with self._transaction():
            current = self._active_qualification_recovery(recovery.lease)
            if current != recovery or (
                current.phase is RecoveryPhase.HELD
                and current.reason
                not in {reason, "post_publication_no_decision"}
            ):
                raise _intake_error("remote qualification HOLD recovery changed")
            reservations = tuple(
                self.get(member.reservation_id) for member in current.lease.members
            )
            if any(
                row.status != member.prior_status
                for row, member in zip(
                    reservations, current.lease.members, strict=True
                )
            ):
                raise _intake_error("remote qualification HOLD changed its cohort")
            with self._evaluation_recovery_mutation(current.lease.lease_id):
                if current.phase is RecoveryPhase.HELD:
                    current = self._transition_evaluation_recovery_locked(
                        current,
                        phase=RecoveryPhase.RESULT_READY,
                        resolution=RecoveryResolution.UNRESOLVED,
                        event_type=RecoveryEventType.RESULT_READY,
                        current_block=current_block,
                    )
                if current_block >= current.lease.expires_block:
                    expires = current_block + lease_blocks
                    renewed_lease = EvaluationLease(
                        current.lease.lease_id,
                        current.lease.generation,
                        current.lease.stage,
                        current.lease.owner,
                        current.lease.members,
                        current.lease.claimed_block,
                        current.lease.initial_expires_block,
                        expires,
                    )
                    previous_expiry = current.lease.expires_block
                    current = self._transition_evaluation_recovery_locked(
                        current,
                        phase=current.phase,
                        resolution=RecoveryResolution.UNRESOLVED,
                        event_type=RecoveryEventType.RENEWED,
                        current_block=current_block,
                        lease=renewed_lease,
                    )
                    cursor = self._db.execute(
                        "UPDATE evaluation_leases SET expires_block=? WHERE lease_id=? "
                        "AND state='active' AND expires_block=?",
                        (expires, current.lease.lease_id, previous_expiry),
                    )
                    if cursor.rowcount != 1:
                        raise _intake_error(
                            "remote qualification HOLD lease changed during renewal"
                        )
                if self._evaluation_mutation_authority:
                    raise _intake_error("nested evaluation mutation authority is forbidden")
                authorized = set(current.lease.reservation_ids)
                self._evaluation_mutation_authority.update(authorized)
                try:
                    for member in current.lease.members:
                        cursor = self._db.execute(
                            "UPDATE reservations SET status='held',decision='',reason=?,"
                            "qualification_evidence_digest=?,retry_group_digest='',"
                            "retry_position=0,qualification_authority_digest='',"
                            "qualification_authority_json='' WHERE reservation_id=? "
                            "AND status=?",
                            (
                                reason,
                                result_digest,
                                member.reservation_id,
                                member.prior_status,
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise _intake_error(
                                "remote qualification HOLD disposition changed"
                            )
                finally:
                    self._evaluation_mutation_authority.difference_update(authorized)
                cursor = self._db.execute(
                    "UPDATE evaluation_leases SET state='completed',completed_block=?,"
                    "result_digest=?,reason='' WHERE lease_id=? AND state='active' "
                    "AND expires_block=?",
                    (
                        current_block,
                        result_digest,
                        current.lease.lease_id,
                        current.lease.expires_block,
                    ),
                )
                if cursor.rowcount != 1:
                    raise _intake_error("remote qualification HOLD lease changed")
                members = self._db.execute(
                    "UPDATE evaluation_lease_members SET active=0 WHERE lease_id=? "
                    "AND active=1",
                    (current.lease.lease_id,),
                )
                if members.rowcount != len(current.lease.members):
                    raise _intake_error("remote qualification HOLD members changed")
                self._append_evaluation_lease_event(
                    current.lease,
                    "completed",
                    finalized_block=current_block,
                    result_digest=result_digest,
                )
                self._complete_evaluation_recovery_locked(
                    current, current_block=current_block
                )
        return current.lease
