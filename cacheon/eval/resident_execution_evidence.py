"""Per-rank kernel execution facts used by qualification feedback.

Rows retain calls, capture coverage and dispatch misses so a miner can see
what ran and why a registered slot was not selected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cacheon._strict import NODE_ADDRESS

# Sentinel for "not observable", distinct from an observed count of zero. A
# generation is non-negative and a rank count is non-negative, so a negative
# value cannot collide with a real observation.
UNOBSERVED = -1

MAX_EXECUTION_TEXT = 256
MAX_SKIPPED_REASONS = 4


def _text(value: object, field: str) -> str:
    if type(value) is not str or len(value) > MAX_EXECUTION_TEXT or not value.isprintable():
        raise ValueError(f"resident execution {field} is not bounded printable text")
    return value


@dataclass(frozen=True)
class SlotExecution:
    """What one rank did with one registered slot."""

    slot: str
    calls: int  # invocations of the candidate entry; UNOBSERVED when unrecorded
    captured: bool | None  # inside a CUDA-graph capture; None when unrecorded
    error: str = ""  # ``Type: message`` when the entry raised
    skipped: tuple[str, ...] = ()  # why live calls routed to stock instead

    def __post_init__(self) -> None:
        if type(self.slot) is not str or NODE_ADDRESS.fullmatch(self.slot) is None:
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
    """One rank's execution receipts, reduced."""

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

    @classmethod
    def from_receipts(cls, rank: int, rows: Mapping[str, object]) -> "RankExecution":
        """Reduce one rank's receipt rows by kind, as ``explain.ranks_from_log``
        groups them, to the facts a gate and a miner both need.

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


def eager_slots() -> frozenset[str]:
    """Registered slots SGLang serves outside its CUDA graph: the scheduler's prefix cache."""

    from cacheon.integrations.sglang_cache import ADDRESS

    return frozenset({ADDRESS})

