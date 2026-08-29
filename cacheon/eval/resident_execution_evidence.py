"""Generation-scoped proof that a resident candidate really executed.

The closing swap carries each rank's receipts for the generation it ends.
``UNOBSERVED`` means broken evidence; observed short coverage means the
candidate did not run cleanly everywhere; only exact rank coverage can
authorize speed grading or screen promotion.

The rows themselves travel with the count. The count says whether every rank
ran; the rows say WHAT ran on each rank, whether it was inside the captured
graph the scored windows replay, whether it raised, and why anything else
routed to stock. A miner's report needs the rows, and a gate that only counted
receipt files could pass a candidate that sat outside the graph.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from cacheon.eval.continuation_codec import ContinuationCodec

# Sentinel for "not observable", distinct from an observed count of zero. A
# generation is non-negative and a rank count is non-negative, so a negative
# value cannot collide with a real observation.
UNOBSERVED = -1

MAX_EXECUTION_TEXT = 256
MAX_SKIPPED_REASONS = 4
_SLOT = re.compile(r"[A-Za-z0-9_.\-]{1,128}\Z")


def _text(value: object, field: str) -> str:
    if type(value) is not str or len(value) > MAX_EXECUTION_TEXT or not value.isprintable():
        raise ValueError(f"resident execution {field} is not bounded printable text")
    return value


@dataclass(frozen=True)
class SlotExecution:
    """What one rank did with one registered slot under one generation."""

    slot: str
    calls: int  # invocations of the candidate entry; UNOBSERVED when unrecorded
    captured: bool | None  # inside a CUDA-graph capture; None when unrecorded
    error: str = ""  # ``Type: message`` when the entry raised
    skipped: tuple[str, ...] = ()  # why live calls routed to stock instead

    def __post_init__(self) -> None:
        if type(self.slot) is not str or _SLOT.fullmatch(self.slot) is None:
            raise ValueError("resident execution slot is invalid")
        if type(self.calls) is not int or self.calls < UNOBSERVED:
            raise ValueError("resident execution calls is invalid")
        if self.captured is not None and type(self.captured) is not bool:
            raise ValueError("resident execution captured is invalid")
        _text(self.error, "error")
        if (
            type(self.skipped) is not tuple
            or len(self.skipped) > MAX_SKIPPED_REASONS
            or len(set(self.skipped)) != len(self.skipped)
        ):
            raise ValueError("resident execution skipped reasons are invalid")
        for reason in self.skipped:
            _text(reason, "skipped reason")


@dataclass(frozen=True)
class RankExecution:
    """One rank's receipts for one closed generation, reduced."""

    rank: int
    loaded: bool  # the bundle loaded and the registry was enabled on this rank
    load_error: str = ""  # the load was attempted and fell back to stock
    slots: tuple[SlotExecution, ...] = ()

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("resident execution rank is invalid")
        if type(self.loaded) is not bool:
            raise ValueError("resident execution loaded flag is invalid")
        _text(self.load_error, "load error")
        if type(self.slots) is not tuple or any(
            type(row) is not SlotExecution for row in self.slots
        ):
            raise ValueError("resident execution slots are not exactly typed")
        names = [row.slot for row in self.slots]
        if names != sorted(set(names)):
            raise ValueError("resident execution slots must be sorted and unique")

    def clean(self, *, eager_slots: frozenset[str]) -> bool:
        """True when this rank ran every registered slot, raised nothing, and
        was inside the captured graph wherever serving replays one.

        ``eager_slots`` are the registered slots SGLang serves outside its CUDA
        graph; every other slot must prove capture, and an unrecorded capture
        fails closed. One rank serving stock out of its captured graph makes the
        whole measurement stock, so there is no partial credit.
        """

        if not self.loaded or self.load_error or not self.slots:
            return False
        return all(
            row.calls >= 1
            and not row.error
            and (row.slot in eager_slots or row.captured is True)
            for row in self.slots
        )

    @classmethod
    def from_receipts(cls, rank: int, rows: Mapping[str, object]) -> "RankExecution":
        """Reduce one rank's receipt rows by kind, as ``receipts.rows_for_scope``
        returns them, to the facts a gate and a miner both need.

        A slot the rank registered but never completed is kept with zero calls:
        "loaded and never called" is the phantom-pass shape and must stay visible.
        """

        def kind(name: str) -> list[dict]:
            found = rows.get(name)
            return [row for row in found if isinstance(row, dict)] if isinstance(found, list) else []

        def message(row: dict, *keys: str) -> str:
            text = " ".join(str(row[key]) for key in keys if row.get(key))
            return "".join(ch if ch.isprintable() else " " for ch in text)[:MAX_EXECUTION_TEXT]

        active = kind("active")
        by_slot: dict[str, dict] = {}
        for row in active:
            for slot in row.get("slots") or ():
                by_slot.setdefault(str(slot), {})
        for row in kind("completed"):
            facts = by_slot.setdefault(str(row.get("slot")), {})
            calls = row.get("calls")
            facts["calls"] = calls if type(calls) is int and calls >= 0 else UNOBSERVED
            facts["captured"] = row.get("captured") if type(row.get("captured")) is bool else None
        for row in kind("failed"):
            by_slot.setdefault(str(row.get("slot")), {})["error"] = (
                message(row, "error_type") + ": " + message(row, "error")
            )
        for row in kind("not_selected"):
            facts = by_slot.setdefault(str(row.get("slot")), {})
            facts["skipped"] = tuple(dict.fromkeys(
                message(
                    {"why": f"{reason.get('outcome')} on "
                     f"{', '.join(map(str, reason.get('fields') or ())) or 'unrecorded'}"},
                    "why",
                )
                for reason in row.get("reasons") or ()
                if isinstance(reason, dict)
            ))[:MAX_SKIPPED_REASONS]
        load_failed = kind("load_failed")
        return cls(
            rank,
            bool(active),
            message(load_failed[0], "reason") if load_failed else "",
            tuple(
                SlotExecution(
                    slot,
                    facts.get("calls", 0),
                    facts.get("captured"),
                    facts.get("error", ""),
                    facts.get("skipped", ()),
                )
                for slot, facts in sorted(by_slot.items())
            ),
        )


@dataclass(frozen=True)
class ResidentExecutionEvidence:
    """Execution evidence for the generation a swap closed.

    ``prior_generation`` is the scope the counts describe, not the generation
    being swapped in — a swap reports the receipts of the generation it is
    ending, because that scope is final only once nothing more can run under it.
    ``ranks`` carries the rows the count was reduced from, one per rank, when
    the closing scope was observable.
    """

    prior_generation: int
    prior_execution_ranks: int
    ranks: tuple[RankExecution, ...] = ()

    def __post_init__(self) -> None:
        for field in ("prior_generation", "prior_execution_ranks"):
            value = getattr(self, field)
            if type(value) is not int or value < UNOBSERVED:
                raise ValueError(f"resident execution evidence {field} is invalid")
        if type(self.ranks) is not tuple or any(
            type(row) is not RankExecution for row in self.ranks
        ):
            raise ValueError("resident execution rows are not exactly typed")
        if self.ranks and (
            [row.rank for row in self.ranks] != list(range(len(self.ranks)))
            or not 0 <= self.prior_execution_ranks <= len(self.ranks)
        ):
            raise ValueError("resident execution rows contradict their count")

    @property
    def observed(self) -> bool:
        return (
            self.prior_generation >= 0
            and self.prior_execution_ranks >= 0
        )

    def proves_execution(self, *, generation: int, expected_ranks: int) -> bool:
        """True only when every rank cleanly executed under exactly ``generation``."""

        return (
            self.observed
            and self.prior_generation == generation
            and expected_ranks >= 1
            and self.prior_execution_ranks == expected_ranks
        )

    def faults(self) -> str:
        """Why each unclean rank was not counted, for the hold that names it."""

        eager = eager_slots()
        found = []
        for row in self.ranks:
            if row.clean(eager_slots=eager):
                continue
            if row.load_error:
                found.append(f"rank {row.rank} failed to load: {row.load_error}")
            elif not row.loaded:
                found.append(f"rank {row.rank} never loaded the bundle")
            for slot in row.slots:
                if slot.error:
                    found.append(f"rank {row.rank} {slot.slot} raised {slot.error}")
                elif slot.calls < 1:
                    found.append(f"rank {row.rank} {slot.slot} was never called")
                elif slot.slot not in eager and slot.captured is not True:
                    found.append(
                        f"rank {row.rank} {slot.slot} ran outside the captured graph"
                    )
        return "; " + "; ".join(found) if found else ""


UNOBSERVED_EVIDENCE = ResidentExecutionEvidence(UNOBSERVED, UNOBSERVED)

#: The wire and artifact form of the evidence: closed fields, exact types, and
#: every constructor check above, mechanically from the dataclass definitions.
EXECUTION_CODEC = ContinuationCodec((ResidentExecutionEvidence, RankExecution))


def eager_slots() -> frozenset[str]:
    """Registered slots SGLang serves outside its CUDA graph.

    Resolved from the slot registry when it is importable; a process without it
    treats every slot as graph-served, which is the stricter reading.
    """

    try:
        from cacheon.slots import SLOTS
    except Exception:  # noqa: BLE001 - the registry needs torch; the gate must not
        return frozenset()
    return frozenset(name for name, spec in SLOTS.items() if not spec.serving_graph_captured)


def summarize_rank_acks(
    rows: dict, *, tp_size: int
) -> ResidentExecutionEvidence:
    """Reduce per-rank swap acks to one execution fact; never raises.

    Any rank that cannot be read, disagrees about which generation is closing,
    or reports unobservable receipts collapses the whole summary to UNOBSERVED.
    A partial reading is not evidence — it would let a rank that silently
    stopped reporting look like a rank that cleanly executed.
    """

    if type(tp_size) is not int or tp_size < 1:
        return UNOBSERVED_EVIDENCE
    generations: set[int] = set()
    reduced: list[RankExecution] = []
    observable = True
    for rank in range(tp_size):
        row = rows.get(rank)
        if not isinstance(row, dict):
            observable = False
            continue
        prior = row.get("prior_generation")
        if type(prior) is not int:
            observable = False
            continue
        generations.add(prior)
        if prior < 0:
            # The lane's first swap closes nothing; there is no scope to count.
            continue
        receipts = row.get("prior_rows")
        if not isinstance(receipts, dict):
            observable = False
            continue
        try:
            reduced.append(RankExecution.from_receipts(rank, receipts))
        except ValueError:
            observable = False
    if len(generations) != 1:
        return UNOBSERVED_EVIDENCE
    generation = generations.pop()
    if generation < 0 or not observable:
        return ResidentExecutionEvidence(max(generation, UNOBSERVED), UNOBSERVED)
    eager = eager_slots()
    executed = sum(row.clean(eager_slots=eager) for row in reduced)
    return ResidentExecutionEvidence(generation, executed, tuple(reduced))


__all__ = [
    "EXECUTION_CODEC",
    "UNOBSERVED",
    "UNOBSERVED_EVIDENCE",
    "RankExecution",
    "ResidentExecutionEvidence",
    "SlotExecution",
    "eager_slots",
    "summarize_rank_acks",
]
