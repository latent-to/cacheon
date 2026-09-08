"""Host-observed resident read measurements shared by execution and regrade."""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field

from cacheon.stack_identity import require_sha256_hex


class CrossoverRuntimeError(RuntimeError):
    """Resident execution or retained measurement violates its sealed plan."""


@dataclass(frozen=True)
class TimedWindow:
    """One host-clocked timed batch inside a read: the v3 scoring quantum.

    The trust model only accepts wall-clock spans stamped by the controller at
    batch boundaries (worker-reported timestamps are hostile input), so the
    window IS the timed batch: every field recomputes exactly from the sealed
    batch evidence and nothing here depends on a worker clock. Window seconds
    exclude inter-batch host gaps, so their sum is bounded by — not equal
    to — the read's timed makespan."""

    batch_index: int
    tokens: int
    seconds: float
    input_tokens: int | None = dataclass_field(
        default=None, metadata={"wire_optional": True}
    )
    prompt_latencies: tuple[tuple[float, float], ...] = dataclass_field(
        default=(), metadata={"wire_optional": True}
    )

    def __post_init__(self) -> None:
        if (
            type(self.batch_index) is not int
            or self.batch_index < 0
            or type(self.tokens) is not int
            or self.tokens <= 0
            or type(self.seconds) is not float
            or not math.isfinite(self.seconds)
            or self.seconds <= 0
            or type(self.prompt_latencies) is not tuple
        ):
            raise CrossoverRuntimeError("timed window is malformed")
        if self.prompt_latencies:
            if (
                type(self.input_tokens) is not int
                or self.input_tokens <= 0
                or self.tokens % len(self.prompt_latencies)
                or self.tokens // len(self.prompt_latencies) < 2
            ):
                raise CrossoverRuntimeError("phase window geometry is malformed")
            for pair in self.prompt_latencies:
                if (
                    type(pair) is not tuple
                    or len(pair) != 2
                    or any(type(t) is not float or not math.isfinite(t) for t in pair)
                    or not 0 < pair[0] < pair[1] <= self.seconds
                ):
                    raise CrossoverRuntimeError("phase window host times are malformed")
        elif self.input_tokens is not None:
            raise CrossoverRuntimeError("phase window lacks prompt latencies")

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_index": self.batch_index,
            "seconds": format(self.seconds, ".17g"),
            "tokens": self.tokens,
            **(
                {
                    "input_tokens": self.input_tokens,
                    "prompt_latencies": [
                        [format(first, ".17g"), format(last, ".17g")]
                        for first, last in self.prompt_latencies
                    ],
                }
                if self.prompt_latencies
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> "TimedWindow":
        if type(value) is not dict or set(value) - {
            "input_tokens",
            "prompt_latencies",
        } != {
            "batch_index",
            "seconds",
            "tokens",
        }:
            raise CrossoverRuntimeError("timed window fields differ")
        try:
            result = cls(
                value["batch_index"],  # type: ignore[arg-type]
                value["tokens"],  # type: ignore[arg-type]
                float(value["seconds"]),  # type: ignore[arg-type]
                value.get("input_tokens"),
                tuple(
                    tuple(float(t) for t in pair)
                    for pair in value.get("prompt_latencies", ())
                ),
            )
            if result.to_dict() != value:
                raise CrossoverRuntimeError("timed window is noncanonical")
            return result
        except (TypeError, ValueError, OverflowError) as exc:
            raise CrossoverRuntimeError("timed window is malformed") from exc


@dataclass(frozen=True)
class ResidentReadRate:
    """One resident arm's charged span and independently retained timed windows."""

    role: str
    lane_digest: str
    launch_digest: str
    session_id: str
    first_batch_index: int
    last_batch_index: int
    first_timed_batch_index: int
    last_timed_batch_index: int
    conditioning_tokens: int
    timed_tokens: int
    charged_tokens: int
    conditioning_seconds: float
    timed_seconds: float
    charged_seconds: float
    tokens_per_second: float
    windows: tuple[TimedWindow, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.role not in {"B", "C", "B_prime"}
            or not isinstance(self.session_id, str)
            or len(self.session_id) != 32
            or any(char not in "0123456789abcdef" for char in self.session_id)
            or self.session_id == "0" * 32
            or any(
                type(value) is not int
                for value in (
                    self.first_batch_index,
                    self.last_batch_index,
                    self.first_timed_batch_index,
                    self.last_timed_batch_index,
                )
            )
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.conditioning_tokens,
                    self.timed_tokens,
                    self.charged_tokens,
                )
            )
            or self.charged_tokens != self.conditioning_tokens + self.timed_tokens
            or not (
                0
                <= self.first_batch_index
                <= self.first_timed_batch_index
                <= self.last_timed_batch_index
                <= self.last_batch_index
            )
            or any(
                not math.isfinite(value) or value <= 0
                for value in (
                    self.conditioning_seconds,
                    self.timed_seconds,
                    self.charged_seconds,
                )
            )
            or self.charged_seconds != self.conditioning_seconds + self.timed_seconds
            or self.tokens_per_second != self.charged_tokens / self.charged_seconds
        ):
            raise CrossoverRuntimeError("resident read rate is malformed")
        windows = tuple(self.windows)
        object.__setattr__(self, "windows", windows)
        if windows and (
            any(type(window) is not TimedWindow for window in windows)
            or tuple(window.batch_index for window in windows)
            != tuple(
                range(
                    self.first_timed_batch_index,
                    self.last_timed_batch_index + 1,
                )
            )
            or sum(window.tokens for window in windows) != self.timed_tokens
        ):
            raise CrossoverRuntimeError(
                "resident read windows do not tile the timed span"
            )
        for field in ("lane_digest", "launch_digest"):
            try:
                require_sha256_hex(getattr(self, field), field=field)
            except ValueError as exc:
                raise CrossoverRuntimeError(str(exc)) from None

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "batches": [self.first_batch_index, self.last_batch_index],
            "charged_seconds": format(self.charged_seconds, ".17g"),
            "charged_tokens": self.charged_tokens,
            "conditioning_seconds": format(self.conditioning_seconds, ".17g"),
            "conditioning_tokens": self.conditioning_tokens,
            "lane_digest": self.lane_digest,
            "launch_digest": self.launch_digest,
            "role": self.role,
            "session_id": self.session_id,
            "timed_batches": [
                self.first_timed_batch_index,
                self.last_timed_batch_index,
            ],
            "timed_seconds": format(self.timed_seconds, ".17g"),
            "timed_tokens": self.timed_tokens,
        }
        if self.windows:
            # Per-window retention exists only for v3 reads; earlier sealed
            # rate rows keep their exact historical bytes and digests.
            row["windows"] = [window.to_dict() for window in self.windows]
            if any(window.prompt_latencies for window in self.windows):
                row["cells"] = phase_cells(self.windows)
        return row

    @classmethod
    def from_dict(cls, value: object) -> "ResidentReadRate":
        fields = {
            "batches",
            "charged_seconds",
            "charged_tokens",
            "conditioning_seconds",
            "conditioning_tokens",
            "lane_digest",
            "launch_digest",
            "role",
            "session_id",
            "timed_batches",
            "timed_seconds",
            "timed_tokens",
        }
        if type(value) is not dict:
            raise CrossoverRuntimeError("resident rate fields differ")
        raw_windows = value.get("windows", [])
        if (
            set(value) - {"windows", "cells"} != fields
            or ("windows" in value and type(raw_windows) is not list)
            or type(value["batches"]) is not list
            or len(value["batches"]) != 2
            or type(value["timed_batches"]) is not list
            or len(value["timed_batches"]) != 2
        ):
            raise CrossoverRuntimeError("resident rate fields differ")
        try:
            conditioning_seconds = float(value["conditioning_seconds"])
            timed_seconds = float(value["timed_seconds"])
            charged_seconds = float(value["charged_seconds"])
            conditioning_tokens = value["conditioning_tokens"]
            timed_tokens = value["timed_tokens"]
            charged_tokens = value["charged_tokens"]
            result = cls(
                value["role"],  # type: ignore[arg-type]
                value["lane_digest"],  # type: ignore[arg-type]
                value["launch_digest"],  # type: ignore[arg-type]
                value["session_id"],  # type: ignore[arg-type]
                value["batches"][0],  # type: ignore[index,arg-type]
                value["batches"][1],  # type: ignore[index,arg-type]
                value["timed_batches"][0],  # type: ignore[index,arg-type]
                value["timed_batches"][1],  # type: ignore[index,arg-type]
                conditioning_tokens,  # type: ignore[arg-type]
                timed_tokens,  # type: ignore[arg-type]
                charged_tokens,  # type: ignore[arg-type]
                conditioning_seconds,
                timed_seconds,
                charged_seconds,
                charged_tokens / charged_seconds,  # type: ignore[operator]
                tuple(TimedWindow.from_dict(row) for row in raw_windows),
            )
            if result.to_dict() != value:
                raise CrossoverRuntimeError("resident rate is noncanonical")
            return result
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise CrossoverRuntimeError("resident rate is malformed") from exc


def _timed_windows(timed: tuple) -> tuple[TimedWindow, ...]:
    """Per-batch windows from the same sealed host-clock spans the aggregate
    rate uses; live measurement and regrade share this single derivation."""

    for row in timed:
        if row.prompt_latencies and (
            len(row.prompt_latencies) != len(row.evidence.prompts)
            or len({prompt.prompt_tokens for prompt in row.evidence.prompts}) != 1
        ):
            raise CrossoverRuntimeError(
                "phase timings differ from their prompt geometry"
            )
    return tuple(
        TimedWindow(
            row.batch_index,
            row.token_numerator,
            float(row.response_completed_at - row.request_started_at),
            row.evidence.prompts[0].prompt_tokens if row.prompt_latencies else None,
            row.prompt_latencies,
        )
        for row in timed
    )


def phase_cells(windows: tuple[TimedWindow, ...]) -> list[dict[str, object]]:
    """Report each input/output/concurrency cell without blending unlike workloads."""
    groups: dict[tuple[int, int, int], list[TimedWindow]] = {}
    for window in windows:
        if not window.prompt_latencies:
            raise CrossoverRuntimeError(
                "phase read lacks a timed window's token latencies"
            )
        concurrency = len(window.prompt_latencies)
        key = window.input_tokens, window.tokens // concurrency, concurrency
        groups.setdefault(key, []).append(window)
    rows = []
    for (input_tokens, output_tokens, concurrency), cells in sorted(groups.items()):
        firsts = [first for cell in cells for first, _ in cell.prompt_latencies]
        intervals = [
            last - first for cell in cells for first, last in cell.prompt_latencies
        ]
        rows.append(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "concurrency": concurrency,
                "timed_batches": len(cells),
                "mean_ttft_seconds": format(sum(firsts) / len(firsts), ".17g"),
                "mean_tpot_seconds": format(
                    sum(intervals) / (len(intervals) * (output_tokens - 1)), ".17g"
                ),
                "end_to_end_output_tokens_per_second": format(
                    sum(cell.tokens for cell in cells)
                    / sum(cell.seconds for cell in cells),
                    ".17g",
                ),
            }
        )
    return rows


__all__ = ["CrossoverRuntimeError", "ResidentReadRate", "TimedWindow", "phase_cells"]

# The extraction must not rename types in existing continuation identities.
for _type in (CrossoverRuntimeError, ResidentReadRate, TimedWindow):
    _type.__module__ = "cacheon.eval.crossover_runtime"
