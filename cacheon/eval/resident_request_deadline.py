"""Pure validation and composition for resident request wall bounds."""

from __future__ import annotations

import math
from typing import TypeVar


class ResidentRequestDeadlineError(ValueError):
    """A caller supplied an invalid or already-expired request wall."""


DeadlineErrorT = TypeVar("DeadlineErrorT", bound=Exception)


def _exact_finite(value: object, message: str, error_type: type[DeadlineErrorT]) -> float:
    if type(value) not in (int, float):
        raise error_type(message)
    try:
        exact = float(value)
    except (OverflowError, ValueError):
        raise error_type(message) from None
    if not math.isfinite(exact):
        raise error_type(message)
    return exact


def require_resident_request_deadline(
    deadline: object,
    *,
    now: object,
    error_type: type[DeadlineErrorT] = ResidentRequestDeadlineError,
) -> float:
    """Return one exact future absolute monotonic deadline."""

    current = _exact_finite(now, "request clock is invalid", error_type)
    absolute = _exact_finite(deadline, "request deadline is invalid", error_type)
    if absolute <= current:
        raise error_type("request deadline has expired")
    return absolute


def resolve_resident_request_deadline(
    now: object,
    configured_timeout_s: object,
    deadline: object | None,
    *,
    error_type: type[DeadlineErrorT] = ResidentRequestDeadlineError,
) -> float:
    """Intersect an optional outer wall with a configured relative timeout."""

    current = _exact_finite(now, "request clock is invalid", error_type)
    timeout = _exact_finite(
        configured_timeout_s, "request timeout is invalid", error_type
    )
    if timeout <= 0:
        raise error_type("request timeout is invalid")
    configured = current + timeout
    if not math.isfinite(configured) or configured <= current:
        raise error_type("request timeout is invalid")
    if deadline is None:
        return configured
    outer = require_resident_request_deadline(
        deadline, now=current, error_type=error_type
    )
    return min(configured, outer)


__all__ = [
    "ResidentRequestDeadlineError",
    "require_resident_request_deadline",
    "resolve_resident_request_deadline",
]
