"""Recover why a screen stage graded a candidate the way it did.

``_stage_result`` hashes ``{authority, candidate, facts, grade, publication,
reason, screen_attempt, service, stage}`` into one ``evidence_digest`` and
returns only the digest. Nothing stores the payload, so the durable receipt
records that ``static`` failed and never records why. A rejected contributor is
told the screen has five gates and left to guess which one, and so is the
operator: the reason existed for the length of one function call and then only
its hash survived.

It survives well enough. Every field except ``reason`` and ``facts`` is already
in the receipt or the reservation row, and the validator draws both of those
from vocabularies it wrote itself. So the payload can be rebuilt and rehashed
until it matches the digest already on record. That reads sealed bytes without
changing them, needs no schema version, and works on every receipt ever
written, including the ones already in the database.

Recovery is exact or absent: a match is the payload, because finding a second
preimage of SHA-256 is the alternative. It resolves *failure* reasons, whose
facts are drawn from small closed sets. Success reasons carry digests of things
the stage built, which cannot be enumerated -- and are not the interesting case.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from cacheon.stack_identity import canonical_digest

SCREEN_EVIDENCE_SCHEMA = "cacheon.eval.b300-screen-stage-evidence.v1"

# Read off the ``reason=`` argument of every ``_stage_result`` call site. A
# reason absent here recovers as unknown rather than wrong, and the test walks
# the source to catch a stage that adds one without telling this table.
STAGE_REASONS: dict[str, tuple[str, ...]] = {
    "static": (
        "static_policy",
        "static_infrastructure",
        "static_authority_changed",
        "static_runtime_quant_mismatch",
        "static_verified",
    ),
    "build": ("build_infrastructure", "native_build_reopened"),
    "abi": ("abi_infrastructure", "eager_abi_failed", "resident_abi_deferred"),
    "graph": (
        "graph_infrastructure",
        "graph_session_failed",
        "resident_graph_deferred",
    ),
}

# ``facts={"exception_type": type(exc).__name__}`` accompanies every
# ``*_infrastructure`` and ``static_policy`` result. The deterministic names are
# the isinstance tuple those stages test; the rest are what an infrastructure
# fault raises in practice.
_EXCEPTION_TYPES: tuple[str, ...] = (
    "_CandidateStaticFailure",
    "ManifestError",
    "EngineTreeError",
    "RebuildError",
    "TargetCatalogError",
    "TargetResolutionError",
    "ArenaServiceError",
    "OSError",
    "FileNotFoundError",
    "PermissionError",
    "TimeoutError",
    "MemoryError",
    "ValueError",
    "RuntimeError",
    "KeyError",
    "TypeError",
)

_QUANTS: tuple[str, ...] = ("nvfp4", "modelopt_fp4", "fp8", "bf16")


def _fact_candidates(reason: str, slots: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    if reason.endswith("_infrastructure") or reason == "static_policy":
        return tuple({"exception_type": name} for name in _EXCEPTION_TYPES)
    if reason == "static_runtime_quant_mismatch":
        # Both values are the validator's, taken from the sealed deployment's
        # slot/quant requirements -- never from the candidate.
        return tuple(
            {"required_quant": quant, "slot": slot}
            for quant, slot in product(_QUANTS, slots)
        )
    return ({},)


@dataclass(frozen=True)
class RecoveredScreenReason:
    """One stage's plaintext reason, proved against its recorded digest."""

    stage: str
    grade: str
    reason: str
    facts: tuple[tuple[str, str], ...]

    def sentence(self) -> str:
        """One line naming the gate and what it decided, for a person."""

        detail = ", ".join(f"{key} {value}" for key, value in self.facts)
        head = f"{self.stage}: {self.reason}"
        return f"{head} ({detail})" if detail else head


def recover_screen_reason(
    *,
    stage: str,
    grade: str,
    evidence_digest: str,
    authority_digest: str,
    candidate_digest: str,
    publication_digest: str,
    service_digest: str,
    screen_attempt: int,
    slots: tuple[str, ...] = (),
) -> RecoveredScreenReason | None:
    """Return the reason whose payload hashes to ``evidence_digest``, or None.

    ``authority_digest`` is the stage adapter's own identity, which the caller
    holds: it is the one field that is neither in the receipt nor guessable.
    """

    for reason in STAGE_REASONS.get(stage, ()):
        for facts in _fact_candidates(reason, slots):
            probe = canonical_digest(
                SCREEN_EVIDENCE_SCHEMA,
                {
                    "authority_digest": authority_digest,
                    "candidate_digest": candidate_digest,
                    "facts": dict(sorted(facts.items())),
                    "grade": grade,
                    "publication_digest": publication_digest,
                    "reason": reason,
                    "screen_attempt": screen_attempt,
                    "service_digest": service_digest,
                    "stage": stage,
                },
            )
            if probe == evidence_digest:
                return RecoveredScreenReason(
                    stage, grade, reason, tuple(sorted(facts.items()))
                )
    return None


__all__ = [
    "STAGE_REASONS",
    "RecoveredScreenReason",
    "SCREEN_EVIDENCE_SCHEMA",
    "recover_screen_reason",
]
