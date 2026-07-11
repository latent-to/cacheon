"""Bounded, non-executable wire format for untrusted worker results.

The candidate launch and collective-verification workers execute miner-controlled
code.  Their result files must therefore be treated as hostile bytes.  In
particular, pickle is not an IPC format here: unpickling lets the writer select an
arbitrary Python callable for the trusted parent to execute.

This module deliberately implements a very small JSON protocol.  It has no object
hooks, dynamic imports, or generic ``module + class`` reconstruction.  Only the
few dataclasses that actually cross Optima's worker boundary are allowlisted, and
their field sets must match exactly.  Container size, semantic depth, JSON nesting,
and total item count are all bounded before a value reaches trusted code.
"""

from __future__ import annotations

import json
import hashlib
import hmac
import math
import os
import stat
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


WIRE_SCHEMA = "optima-worker-result-v1"
AUTH_HEADER = b"optima-worker-auth-v1"

# The documented reasoning-model benchmark regime can return 64 prompts × up to
# 4096 positions × top-20 logprobs.  That is legitimately hundreds of MiB in JSON
# and tens of millions of bounded scalar/container items.  These ceilings preserve
# that regime while still preventing an unbounded hostile file/container walk.
MAX_WIRE_BYTES = 512 * 1024 * 1024
MAX_DEPTH = 32
MAX_JSON_NESTING = 96
MAX_ITEMS = 32_000_000
MAX_STRING_CHARS = 1_000_000
MAX_INT_ABS = (1 << 63) - 1


class WireError(ValueError):
    """The worker result is unsupported, malformed, or exceeds a wire bound."""


@dataclass(frozen=True)
class CollectiveVerdict:
    """Internal result written by one distributed-verification rank."""

    rank: int
    passed: bool
    score: float
    max_abs: float
    detail: str
    metric: str
    err: str | None
    graph_replays: int


@dataclass(frozen=True)
class LaunchOutcome:
    """Internal success/error envelope written by a candidate launch child."""

    value: Any
    error: str | None


# Exact class identities accepted by the encoder.  Strings are constants owned by
# the validator, never module/class names copied from a payload.
_DATACLASS_TAGS = {
    ("optima.eval.throughput_kl", "ModeResult"): "mode_result",
    ("optima.verify", "ShapeResult"): "shape_result",
    ("optima.verify", "VerifyResult"): "verify_result",
    ("optima.ipc", "CollectiveVerdict"): "collective_verdict",
    ("optima.ipc", "LaunchOutcome"): "launch_outcome",
}


@dataclass
class _Budget:
    remaining: int = field(default_factory=lambda: MAX_ITEMS)

    def take(self, count: int = 1) -> None:
        if count < 0 or self.remaining < count:
            raise WireError(f"worker result exceeds {MAX_ITEMS} items")
        self.remaining -= count


def _checked_string(value: str, *, what: str = "string") -> str:
    if len(value) > MAX_STRING_CHARS:
        raise WireError(f"{what} exceeds {MAX_STRING_CHARS} characters")
    return value


def _encode(value: Any, *, depth: int, budget: _Budget) -> Any:
    if depth > MAX_DEPTH:
        raise WireError(f"worker result exceeds depth {MAX_DEPTH}")
    budget.take()

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_INT_ABS:
            raise WireError("integer is outside the signed 64-bit wire range")
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        marker = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return {"$": "float", "value": marker}
    if isinstance(value, str):
        return _checked_string(value)
    if isinstance(value, list):
        return [_encode(item, depth=depth + 1, budget=budget) for item in value]
    if isinstance(value, tuple):
        return {
            "$": "tuple",
            "items": [_encode(item, depth=depth + 1, budget=budget) for item in value],
        }
    if isinstance(value, dict):
        encoded_items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise WireError("worker-result mappings require string keys")
            budget.take()
            encoded_items.append([
                _checked_string(key, what="mapping key"),
                _encode(item, depth=depth + 1, budget=budget),
            ])
        return {"$": "dict", "items": encoded_items}

    if is_dataclass(value) and not isinstance(value, type):
        identity = (type(value).__module__, type(value).__qualname__)
        tag = _DATACLASS_TAGS.get(identity)
        if tag is None:
            raise WireError(
                "dataclass type is not allowed on the worker-result wire: "
                f"{identity[0]}.{identity[1]}"
            )
        values = {
            field.name: _encode(
                getattr(value, field.name), depth=depth + 1, budget=budget
            )
            for field in fields(value)
        }
        return {"$": "dataclass", "name": tag, "fields": values}

    # Do not call repr(), reducers, iterators, or user attributes here.  The object
    # may be miner-controlled; merely explaining the failure must not invoke it.
    raise WireError("unsupported worker-result value type")


def dumps(value: Any, *, max_bytes: int = MAX_WIRE_BYTES) -> bytes:
    """Encode one result with hard resource bounds and strict JSON numbers."""

    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    envelope = {
        "schema": WIRE_SCHEMA,
        "payload": _encode(value, depth=0, budget=_Budget()),
    }
    try:
        data = json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise WireError(f"worker result is not JSON encodable: {exc}") from None
    if len(data) > max_bytes:
        raise WireError(f"worker result exceeds {max_bytes} bytes")
    return data


def _auth_inputs(key: bytes, nonce: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("worker-result authentication key must contain at least 32 bytes")
    if not isinstance(nonce, bytes) or len(nonce) < 16:
        raise ValueError("worker-result nonce must contain at least 16 bytes")


def authenticated_dumps(value: Any, *, key: bytes, nonce: bytes,
                        max_bytes: int = MAX_WIRE_BYTES) -> bytes:
    """Encode and authenticate a result written near candidate-controlled processes.

    JSON safety prevents object execution, but does not establish who wrote the bytes.
    Candidate native code shares the worker filesystem and could otherwise replace a
    valid result with invented throughput.  The parent creates a fresh key/nonce and
    passes them only to its trusted timing child (never through the scheduler env).
    """
    _auth_inputs(key, nonce)
    payload = dumps(value, max_bytes=max_bytes)
    mac = hmac.new(key, AUTH_HEADER + nonce + payload, hashlib.sha256).hexdigest()
    header = AUTH_HEADER + b" " + nonce.hex().encode("ascii") + b" " + mac.encode("ascii")
    data = header + b"\n" + payload
    if len(data) > max_bytes + 256:
        raise WireError("authenticated worker result exceeds its wire bound")
    return data


def authenticated_loads(data: bytes, *, key: bytes, nonce: bytes,
                        max_bytes: int = MAX_WIRE_BYTES) -> Any:
    """Verify origin/freshness before parsing any worker-controlled JSON."""
    _auth_inputs(key, nonce)
    if not isinstance(data, bytes) or len(data) > max_bytes + 256:
        raise WireError("authenticated worker result exceeds its wire bound")
    try:
        header, payload = data.split(b"\n", 1)
        marker, encoded_nonce, encoded_mac = header.split(b" ")
    except ValueError:
        raise WireError("worker result is missing its authentication header") from None
    if marker != AUTH_HEADER:
        raise WireError("worker result authentication schema mismatch")
    if not hmac.compare_digest(encoded_nonce, nonce.hex().encode("ascii")):
        raise WireError("worker result nonce mismatch (stale/replayed file)")
    expected = hmac.new(
        key, AUTH_HEADER + nonce + payload, hashlib.sha256
    ).hexdigest().encode("ascii")
    if not hmac.compare_digest(encoded_mac, expected):
        raise WireError("worker result authentication failed")
    return loads(payload, max_bytes=max_bytes)


def _check_json_nesting(text: str) -> None:
    """Reject pathological nesting before handing text to ``json.loads``."""

    depth = 0
    quoted = False
    escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING:
                raise WireError(
                    f"worker-result JSON exceeds nesting {MAX_JSON_NESTING}"
                )
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise WireError("worker-result JSON has unbalanced delimiters")
    if depth != 0 or quoted or escaped:
        raise WireError("worker-result JSON is truncated or unbalanced")


def _dataclass_class(tag: str):
    """Resolve a fixed wire tag to a fixed trusted class.

    These are static imports in validator code.  Payload text is never interpreted
    as a module or attribute name.
    """

    if tag == "mode_result":
        from optima.eval.throughput_kl import ModeResult

        return ModeResult
    if tag == "shape_result":
        from optima.verify import ShapeResult

        return ShapeResult
    if tag == "verify_result":
        from optima.verify import VerifyResult

        return VerifyResult
    if tag == "collective_verdict":
        return CollectiveVerdict
    if tag == "launch_outcome":
        return LaunchOutcome
    raise WireError(f"unknown worker-result dataclass tag: {tag!r}")


def _require_number(value: Any, field: str, *, finite: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WireError(f"{field} must be numeric")
    if finite and not math.isfinite(float(value)):
        raise WireError(f"{field} must be finite")


def _validate_dataclass(tag: str, value: Any) -> None:
    """Validate load-bearing field types after exact-field reconstruction."""

    if tag == "mode_result":
        _require_number(value.tok_per_s, "ModeResult.tok_per_s", finite=True)
        if not isinstance(value.tok_per_s_samples, list):
            raise WireError("ModeResult.tok_per_s_samples must be a list")
        for sample in value.tok_per_s_samples:
            _require_number(sample, "ModeResult throughput sample", finite=True)
        if isinstance(value.tokens, bool) or not isinstance(value.tokens, int):
            raise WireError("ModeResult.tokens must be an integer")
        if not isinstance(value.per_prompt, list):
            raise WireError("ModeResult.per_prompt must be a list")
        return

    if tag == "shape_result":
        if not isinstance(value.shape, dict) or not isinstance(value.dtype, str):
            raise WireError("ShapeResult shape/dtype have invalid types")
        if not isinstance(value.passed, bool) or not isinstance(value.applicable, bool):
            raise WireError("ShapeResult passed/applicable fields must be boolean")
        for name in ("max_abs_err", "max_rel_err", "pass_ratio"):
            _require_number(getattr(value, name), f"ShapeResult.{name}")
        if not isinstance(value.detail, str) or not isinstance(value.metric, str):
            raise WireError("ShapeResult detail/metric must be strings")
        if (isinstance(value.graph_replays, bool)
                or not isinstance(value.graph_replays, int)
                or value.graph_replays < 0):
            raise WireError("ShapeResult.graph_replays must be a nonnegative integer")
        return

    if tag == "verify_result":
        if not isinstance(value.slot, str) or not isinstance(value.dtype, str):
            raise WireError("VerifyResult slot/dtype must be strings")
        if not all(isinstance(v, bool) for v in (
            value.passed, value.graph_required, value.graph_verified
        )):
            raise WireError("VerifyResult verdict fields must be boolean")
        if (isinstance(value.coverage_required, bool)
                or not isinstance(value.coverage_required, int)
                or value.coverage_required < 0):
            raise WireError("VerifyResult.coverage_required must be a nonnegative integer")
        if (not isinstance(value.shape_results, list)
                or not all(type(item).__module__ == "optima.verify"
                           and type(item).__qualname__ == "ShapeResult"
                           for item in value.shape_results)):
            raise WireError("VerifyResult.shape_results must contain ShapeResult values")
        return

    if tag == "collective_verdict":
        if (isinstance(value.rank, bool) or not isinstance(value.rank, int)
                or value.rank < 0):
            raise WireError("CollectiveVerdict.rank must be a nonnegative integer")
        if not isinstance(value.passed, bool):
            raise WireError("CollectiveVerdict.passed must be boolean")
        _require_number(value.score, "CollectiveVerdict.score", finite=True)
        _require_number(value.max_abs, "CollectiveVerdict.max_abs")
        if not isinstance(value.detail, str) or not isinstance(value.metric, str):
            raise WireError("CollectiveVerdict detail/metric must be strings")
        if value.err is not None and not isinstance(value.err, str):
            raise WireError("CollectiveVerdict.err must be a string or null")
        if (isinstance(value.graph_replays, bool)
                or not isinstance(value.graph_replays, int)
                or value.graph_replays < 0):
            raise WireError(
                "CollectiveVerdict.graph_replays must be a nonnegative integer"
            )
        return

    if tag == "launch_outcome":
        if value.error is not None and not isinstance(value.error, str):
            raise WireError("LaunchOutcome.error must be a string or null")
        if value.error is not None and value.value is not None:
            raise WireError("failed LaunchOutcome must not contain a value")
        return

    raise WireError(f"unvalidated worker-result dataclass tag: {tag!r}")


def _decode(value: Any, *, depth: int, budget: _Budget) -> Any:
    if depth > MAX_DEPTH:
        raise WireError(f"worker result exceeds depth {MAX_DEPTH}")
    budget.take()

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_INT_ABS:
            raise WireError("integer is outside the signed 64-bit wire range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WireError("bare nonfinite JSON number is forbidden")
        return value
    if isinstance(value, str):
        return _checked_string(value)
    if isinstance(value, list):
        return [_decode(item, depth=depth + 1, budget=budget) for item in value]
    if not isinstance(value, dict):
        raise WireError("unsupported JSON value on worker-result wire")

    marker = value.get("$")
    if marker == "float":
        if set(value) != {"$", "value"}:
            raise WireError("nonfinite-float marker has unexpected fields")
        names = {"nan": float("nan"), "inf": float("inf"), "-inf": -float("inf")}
        if value["value"] not in names:
            raise WireError("invalid nonfinite-float marker")
        return names[value["value"]]

    if marker == "tuple":
        if set(value) != {"$", "items"} or not isinstance(value["items"], list):
            raise WireError("tuple marker is malformed")
        return tuple(
            _decode(item, depth=depth + 1, budget=budget)
            for item in value["items"]
        )

    if marker == "dict":
        if set(value) != {"$", "items"} or not isinstance(value["items"], list):
            raise WireError("mapping marker is malformed")
        result: dict[str, Any] = {}
        for pair in value["items"]:
            budget.take()
            if (not isinstance(pair, list) or len(pair) != 2
                    or not isinstance(pair[0], str)):
                raise WireError("mapping entry is malformed")
            key = _checked_string(pair[0], what="mapping key")
            if key in result:
                raise WireError(f"duplicate mapping key: {key!r}")
            result[key] = _decode(pair[1], depth=depth + 1, budget=budget)
        return result

    if marker == "dataclass":
        if set(value) != {"$", "name", "fields"}:
            raise WireError("dataclass marker has unexpected fields")
        tag = value["name"]
        raw_fields = value["fields"]
        if not isinstance(tag, str) or not isinstance(raw_fields, dict):
            raise WireError("dataclass marker is malformed")
        cls = _dataclass_class(tag)
        expected = {field.name for field in fields(cls)}
        if set(raw_fields) != expected:
            missing = sorted(expected - set(raw_fields))
            extra = sorted(set(raw_fields) - expected)
            raise WireError(
                f"{tag} fields do not match schema (missing={missing}, extra={extra})"
            )
        decoded_fields = {
            name: _decode(raw_fields[name], depth=depth + 1, budget=budget)
            for name in sorted(expected)
        }
        try:
            result = cls(**decoded_fields)
        except (TypeError, ValueError) as exc:
            raise WireError(f"could not construct {tag}: {exc}") from None
        _validate_dataclass(tag, result)
        return result

    raise WireError("unrecognized mapping marker on worker-result wire")


def loads(data: bytes, *, max_bytes: int = MAX_WIRE_BYTES) -> Any:
    """Decode hostile bytes without executing writer-selected Python code."""

    if not isinstance(data, bytes):
        raise TypeError("worker-result payload must be bytes")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if len(data) > max_bytes:
        raise WireError(f"worker result exceeds {max_bytes} bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WireError(f"worker result is not UTF-8: {exc}") from None
    _check_json_nesting(text)
    try:
        envelope = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise WireError(f"worker result is not valid JSON: {exc}") from None
    if (not isinstance(envelope, dict)
            or set(envelope) != {"schema", "payload"}
            or envelope.get("schema") != WIRE_SCHEMA):
        raise WireError("worker-result envelope/schema is invalid")
    return _decode(envelope["payload"], depth=0, budget=_Budget())


def dump_file(path: str | os.PathLike[str], value: Any) -> None:
    """Write a complete bounded result to an already-created regular file."""

    data = dumps(value)
    flags = os.O_WRONLY | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise WireError("worker-result destination is not a regular file")
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def load_file(path: str | os.PathLike[str]) -> Any:
    """Read at most ``MAX_WIRE_BYTES + 1`` and decode fail-closed."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise WireError("worker-result source is not a regular file")
        if info.st_size > MAX_WIRE_BYTES:
            raise WireError(f"worker result exceeds {MAX_WIRE_BYTES} bytes")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(MAX_WIRE_BYTES + 1)
    finally:
        os.close(fd)
    return loads(data)


def dump_authenticated_file(
    path: str | os.PathLike[str], value: Any, *, key: bytes, nonce: bytes
) -> None:
    """Write a complete result whose origin/freshness the parent can verify."""
    data = authenticated_dumps(value, key=key, nonce=nonce)
    flags = os.O_WRONLY | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise WireError("worker-result destination is not a regular file")
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def load_authenticated_file(
    path: str | os.PathLike[str], *, key: bytes, nonce: bytes
) -> Any:
    """Read, authenticate, freshness-check, then safely decode a launch result."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise WireError("worker-result source is not a regular file")
        if info.st_size > MAX_WIRE_BYTES + 256:
            raise WireError(f"worker result exceeds {MAX_WIRE_BYTES + 256} bytes")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(MAX_WIRE_BYTES + 257)
    finally:
        os.close(fd)
    return authenticated_loads(data, key=key, nonce=nonce)
