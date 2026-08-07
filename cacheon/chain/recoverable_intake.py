"""Opt-in durable qualification recovery over the finalized intake store.

The base :class:`FinalizedIntakeStore` remains the version-1 authority for
databases without recovery schema. Once recovery triggers are commissioned,
every mutating evaluator connection—including screen dispatch—must use this
exact store type so the connection-local SQL capabilities and recovery mixin
reopen together.
"""

from __future__ import annotations

from cacheon.chain.evaluation_recovery_store import (
    EvaluationRecoveryStoreError,
    EvaluationRecoveryStoreMixin,
    configure_evaluation_recovery_connection,
    ensure_evaluation_recovery_schema,
)
from cacheon.chain.intake import FinalizedIntakeStore, IntakeError


class RecoverableFinalizedIntakeStore(
    EvaluationRecoveryStoreMixin, FinalizedIntakeStore
):
    """Finalized intake authority with fail-closed qualification recovery."""

    _evaluation_recovery_enabled = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._evaluation_recovery_mutation_authority: set[str] = set()
        super().__init__(*args, **kwargs)

    def _create_schema(self) -> None:
        configure_evaluation_recovery_connection(
            self._db, self._evaluation_recovery_mutation_authority
        )
        super()._create_schema()
        try:
            ensure_evaluation_recovery_schema(self._db)
        except EvaluationRecoveryStoreError as exc:
            raise IntakeError(f"evaluation recovery schema cannot open: {exc}") from None


__all__ = ["RecoverableFinalizedIntakeStore"]
