"""Seam execution receipts — positive accounting evidence for the referee.

The failure mode this closes: the candidate engine comes up WITHOUT the seam
(missing ``cacheon.pth`` or bad env) and the eval scores stock-vs-stock —
identical logits, KL exactly 0.0, verdict PASS. Candidate activation failures
propagate after writing a receipt; the driver also demands positive evidence.

Evidence lives where the seam lives — in sglang's spawned scheduler ranks — so it
travels by file: the driver sets ``CACHEON_SEAM_RECEIPT_DIR`` for the candidate
launch (the resident lane sets a root and one scope per swap generation), ranks
write receipts there, the driver requires them:

  * ``active``       — bundle loaded + registry enabled in a rank (seam.activate).
  * ``load_failed``  — a rank attempted the bundle load and then failed loudly.
  * ``completed``    — a dispatcher produced the model-facing output after invoking
                       the selected implementation; one file per slot per process,
                       carrying ``calls`` (invocations under this scope) and
                       ``captured`` (at least one happened inside a CUDA-graph
                       capture). ``captured`` is the fact that matters: a scored
                       window replays the graph without re-entering Python, so a
                       candidate absent from it serves stock on every replay while
                       its receipt sits on disk from eager warmup.
  * ``failed``       — the selected implementation raised. The exception still
                       propagates (there is no fallback), but the rank names the
                       slot and the exception before the engine goes down with it.
  * ``not_selected`` — why a live call routed to stock while a candidate was
                       registered, keyed on fields and reasons, never values.

An invoked entry either produces the output or raises; nothing serves stock in a
candidate's name. ``completed`` is diagnostic execution evidence, not hostile-code
proof: candidate Python shares the scheduler process and can forge process-local
state; complete-engine isolation plus external qualification is the crown boundary.

Counting costs one list increment per call; capture detection is delegated to the
dispatch layer and probed only until the first capturing call. Files are rewritten
at scope change and at exit, never per call. No env var set -> every helper is a
silent no-op.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from types import FunctionType
from typing import Optional

logger = logging.getLogger("cacheon.receipts")

_SAFE_RE = re.compile(r"[^0-9A-Za-z._\-]+")
_SAFE_SOURCE_RE = re.compile(r"[^0-9A-Za-z._/\-]+")
_VALIDATOR_WRITABLE_ROOTS = (
    "/usr/local/lib/",
    "/sgl-workspace/sglang/python/",
    "/cacheon/runtime-cache/",
    "/tmp/",
)
# Every kind a rank writes about itself carries the rank's identity. A kind left
# out of this set has no ``pid`` on disk and is dropped by every per-process
# reader — which is how the routing reasons vanished from the resident lane's
# record while the code that wrote them was reported as working.
_IDENTITY_KINDS = frozenset(
    {
        "active",
        "load_failed",
        "completed",
        "failed",
        "not_selected",
        "audit",
        "aot_loaded",
        "aot_invoked",
    }
)
# Include the receipt directory so one long-lived process can participate in
# independent launches without an earlier launch suppressing the later receipt.
_ONCE: set[tuple[str, int, str, str]] = set()
_ONCE_LOCK = threading.Lock()

# Sub-path appended to the receipt root, so "once per slot" means once per
# scope rather than once per process lifetime.
_SCOPE = ""
_SCOPE_LOCK = threading.Lock()

# Invocation accounting for the current scope: slot -> [calls, captured].
# Plain lists mutated in place by the dispatchers, which run single-threaded inside
# one scheduler rank's model forward. A lost increment under an unexpected thread
# would understate a diagnostic; it can never manufacture evidence that a candidate
# ran, so this deliberately takes no lock on the hot path.
_CALLS: dict[str, list[int]] = {}
# CUDA-graph capture detector, installed by the dispatch layer that already owns
# it. None means nothing told us how to detect capture, which is reported as
# unknown rather than as "not captured".
_GRAPH_PROBE = None
# slot -> {(outcome, mismatch detail): payload}. Bounded by the number of DISTINCT
# routing reasons, not by call volume; see ``not_selected``.
_NOT_SELECTED: dict[str, dict[tuple, dict]] = {}
# slot -> {input signature: {kernel name: launches}}. Filled by kernel_trace when
# it is armed, empty otherwise. Bounded by that module's per-slot profile budget.
_KERNELS: dict[str, dict[str, dict]] = {}

# Fallback receipt root for a process the driver launched WITHOUT one. The
# one-shot driver mints a receipt directory only for an engine that is active at
# construction (engine_worker: `mkdtemp() if active else ""`), and a resident
# lane is born stock — it acquires candidates later by hot-swap — so it runs its
# whole life with the environment variable forced empty and cannot emit any
# evidence at all. The swap seam establishes this root instead, at the moment
# the lane stops being stock. The environment still wins where it is set, so the
# one-shot path is untouched.
_ROOT = ""
_ROOT_LOCK = threading.Lock()


def set_root(path: object) -> str:
    """Establish a receipt root for a process launched without one.

    Returns the root actually in force, which is the environment's whenever that
    is set. Never raises: a diagnostic must not be able to kill an engine.
    """

    global _ROOT
    try:
        cleaned = "" if path is None else str(path).strip()
    except Exception:  # noqa: BLE001 - hostile __str__ must not break the engine
        cleaned = ""
    if cleaned and os.path.isabs(cleaned):
        with _ROOT_LOCK:
            _ROOT = cleaned
    return _root()


def _root() -> str:
    """The receipt root in force: the driver's environment, else the seam's."""

    return os.environ.get("CACHEON_SEAM_RECEIPT_DIR", "").strip() or _ROOT


def set_scope(scope: object) -> str:
    """Re-scope subsequent receipts; returns the scope actually applied.

    A resident engine serves many candidates on one process. Execution receipts
    are once-per-slot-per-root, so without a scope the second candidate onward
    would emit nothing at all and the controller would either read the *first*
    candidate's evidence as if it were theirs, or see an empty directory and
    convict an honest bundle. Scoping by the swap generation — which the swap
    ack already carries, so no wire change is needed to find it — keeps "once"
    honest while letting every candidate produce its own receipts.

    Never raises: a diagnostic must not be able to kill an engine.
    """

    global _SCOPE
    try:
        cleaned = "" if scope is None else _SAFE_RE.sub("_", str(scope))[:64]
    except Exception:  # noqa: BLE001 - hostile __str__ must not break the engine
        cleaned = ""
    # `_SAFE_RE` permits '.', so a scope of ".." would resolve to the receipt
    # root's parent. Production passes an integer swap generation, but this
    # runs in the candidate's own process: fail closed to unscoped rather than
    # let a scope escape the root it is supposed to partition.
    if not cleaned or not cleaned[0].isalnum() or ".." in cleaned:
        cleaned = ""
    # The outgoing scope's counts are final the instant nothing more can run under
    # it. Persist them before the root moves, or a resident lane would attribute
    # the closing candidate's invocations to the one arriving. Every per-scope
    # table resets here: a routing reason left over from the previous candidate
    # would suppress the identical reason for the next one.
    flush_calls()
    _CALLS.clear()
    _KERNELS.clear()
    _NOT_SELECTED.clear()
    with _SCOPE_LOCK:
        _SCOPE = cleaned
    # Create the scope eagerly. Receipt files are written lazily, so without
    # this an un-invoked candidate and a broken receipt path are the same
    # observation — an absent directory — and a reader would have to convert an
    # infrastructure fault into a candidate verdict to act on it. With the
    # directory present, "exists but holds no execution receipt" means the
    # candidate did not run, and "absent" means the evidence path itself is
    # unsound. Best effort: never raise into an engine.
    if cleaned:
        raw = _root()
        if raw:
            try:
                _resolved_dir(os.path.join(raw, cleaned)).mkdir(
                    parents=True, exist_ok=True
                )
            except Exception:  # noqa: BLE001 - diagnostics never break an engine
                logger.exception("cacheon: receipt scope mkdir failed (%s)", cleaned)
    return cleaned


def _dir() -> str:
    raw = _root()
    if not raw:
        return ""
    scope = _SCOPE
    return os.path.join(raw, scope) if scope else raw


def _resolved_dir(raw: str) -> Path:
    try:
        return Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return Path(os.path.abspath(os.path.expanduser(raw)))


# Set the first time a real group identity resolves; see ``identity``.
_IDENTITY: Optional[dict] = None


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.isascii() or not raw.isdecimal():
        return None
    return int(raw)


def identity() -> dict:
    """Best-effort scheduler-member identity, always including a stable PID.

    Memoized once it resolves. A process's rank in its group does not change, but
    its ability to READ that rank does: every scheduler rank destroys its process
    group before exit, and the counts receipt is rewritten after that, at
    ``atexit``. Re-detecting there returns the degraded ``-1`` identity, which no
    longer matches the ``active`` receipt written while the group was live — and
    a coverage check that compares the two would call the whole run's execution
    evidence malformed at the last instant. The PID guard keeps a forked child
    from inheriting its parent's rank.
    """
    global _IDENTITY
    if _IDENTITY is not None and _IDENTITY["pid"] == os.getpid():
        return dict(_IDENTITY)
    pid = os.getpid()
    rank: Optional[int] = None
    world_size: Optional[int] = None
    try:
        import torch.distributed as dist  # deferred: CPU tooling need not initialize it

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
    except Exception:  # noqa: BLE001 - receipts must never break model execution
        pass
    if rank is None or world_size is None:
        rank = _env_int("RANK")
        world_size = _env_int("WORLD_SIZE")
    if (
        rank is None
        or world_size is None
        or world_size < 1
        or rank < 0
        or rank >= world_size
    ):
        rank = world_size = -1
    resolved = {"pid": pid, "rank": rank, "world_size": world_size}
    if rank >= 0:
        _IDENTITY = resolved
    return dict(resolved)


def _write_to(root: Path, kind: str, payload: dict, *, tag: str = "") -> bool:
    try:
        body = dict(payload)
        if kind in _IDENTITY_KINDS:
            # Detected identity is authoritative over caller-supplied fields.
            body = {**body, **identity()}
        root.mkdir(parents=True, exist_ok=True)
        suffix = f".{_SAFE_RE.sub('_', tag)}" if tag else ""
        p = root / f"{kind}{suffix}.{os.getpid()}.json"
        # The resident lane rewrites this file at every swap while the host
        # reads it for the crossover proof; a truncating write is observable as
        # an empty file (2026-08-27, 2026-09-03: "invalid receipt ... Expecting
        # value" then HOLD). Replace atomically so a reader sees the previous
        # or the new receipt, never a zero-byte one. The dot prefix keeps the
        # temporary file outside every "<kind>*.json" collector glob.
        tmp = root / f".{p.name}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(body, sort_keys=True))
        os.replace(tmp, p)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("cacheon: receipt write failed (kind=%s)", kind)
        return False


def write(kind: str, payload: dict, *, tag: str = "") -> None:
    """Write one receipt file; never raises (a receipt must not break an engine)."""
    raw = _dir()
    if raw:
        _write_to(_resolved_dir(raw), kind, payload, tag=tag)


def validator_runtime_failure(error_type: object, message: object) -> bool:
    """Whether an installed validator library failed on its runtime files."""

    kind = str(error_type)
    text = str(message)
    permission = (
        kind.endswith("PermissionError")
        and "Permission denied" in text
        and any(root in text.replace("\\", "/") for root in _VALIDATOR_WRITABLE_ROOTS)
    )
    missing_cubin_metadata = kind.endswith("AssertionError") and (
        "Failed to get checksums.txt" in text
    )
    return permission or missing_cubin_metadata


def _write_execution_once(
    kind: str,
    slot: str,
    *,
    error: BaseException | None = None,
    phase: str = "",
    entry: Callable[..., object] | None = None,
) -> None:
    """Write one slot execution receipt without adding hot-path file churn."""
    rdir = _dir()
    if not rdir:
        # Do not consume the guard: a later independently receipted launch in this
        # process must still produce evidence.
        return
    root = _resolved_dir(rdir)
    key = (str(root), os.getpid(), kind, slot)
    with _ONCE_LOCK:
        if key in _ONCE:
            return
        payload = {"slot": slot}
        if error is not None:
            try:
                message = str(error)[:512]
            except Exception:  # noqa: BLE001 - hostile exception formatting is diagnostic
                message = "<unprintable exception>"
            payload.update(error_type=type(error).__name__, error=message)
            if validator_runtime_failure(payload["error_type"], message):
                payload["failure_owner"] = "validator_runtime"
            if phase in {"prepare", "entry"}:
                payload["phase"] = phase
            if type(entry) is FunctionType:
                code = entry.__code__
                cursor = error.__traceback__
                while cursor is not None:
                    if cursor.tb_frame.f_code is code:
                        source = code.co_filename.replace("\\", "/")
                        marker = "/kernels/"
                        if marker in source:
                            source = "kernels/" + source.rsplit(marker, 1)[1]
                        else:
                            source = source.rsplit("/", 1)[-1]
                        source = _SAFE_SOURCE_RE.sub("_", source)[:128]
                        if source:
                            payload.update(source=source, line=cursor.tb_lineno)
                        break
                    cursor = cursor.tb_next
        if _write_to(root, kind, payload, tag=slot):
            _ONCE.add(key)


def set_graph_probe(probe: object) -> None:
    """Install the CUDA-graph capture detector owned by the dispatch layer."""

    global _GRAPH_PROBE
    _GRAPH_PROBE = probe if callable(probe) else None


def _count_call(slot: str) -> None:
    """Tally one invocation of ``slot``. Hot path: keep it cheap.

    The capture probe runs only until this slot has been seen inside a capture.
    One capturing invocation is the whole claim — it puts the candidate in the
    graph the scored windows replay — so continuing to probe would buy nothing.
    """

    entry = _CALLS.get(slot)
    if entry is None:
        entry = [0, 0]
        _CALLS[slot] = entry
    entry[0] += 1
    if not entry[1] and _GRAPH_PROBE is not None:
        try:
            if _GRAPH_PROBE():
                entry[1] = 1
        except Exception:  # noqa: BLE001 - a probe must not break model execution
            pass


def _calls_payload(slot: str) -> dict:
    entry = _CALLS.get(slot)
    if entry is None:
        return {}
    payload: dict = {"calls": entry[0]}
    if _GRAPH_PROBE is not None:
        payload["captured"] = bool(entry[1])
    kernels = _KERNELS.get(slot)
    if kernels:
        payload["kernels"] = kernels
    return payload


def capturing() -> bool:
    """Is a CUDA graph being captured right now? True when nothing can tell.

    Deliberately not tri-state. Its one caller arms a profiler, and profiling
    during capture is the failure it exists to avoid, so "we do not know" and
    "yes" must lead to the same decision. ``_calls_payload`` reports the same
    underlying probe as tri-state because there the honest answer matters.

    The unknown answers used to be ``False``, which is the one thing the
    docstring above says they must not be. It cost nothing while no production
    path could arm the profiler; it stopped being free once one could. Failing
    closed also costs no coverage: ``cacheon.dispatch`` installs the probe when
    it is imported, and the trace arms only after the registry is enabled
    through that same module, so an absent probe means no dispatch is running
    and there is nothing to profile.
    """

    if _GRAPH_PROBE is None:
        return True
    try:
        return bool(_GRAPH_PROBE())
    except Exception:  # noqa: BLE001 - a probe must not break model execution
        return True


def record_kernels(slot: str, signature: str, counts: dict) -> None:
    """Attach one observed launch table to ``slot``'s receipt for this scope.

    Keyed by input signature rather than accumulated, because the question this
    answers is which internal path a bundle took, and a bundle that branches
    takes different paths at different shapes. Summing them would erase exactly
    the distinction being measured.
    """

    if not counts:
        return
    _KERNELS.setdefault(slot, {})[signature] = dict(counts)


def completed(slot: str) -> None:
    """Record successful candidate output production for this slot/process.

    The file is written once — the count it carries is refreshed at every phase
    boundary and at exit, so the hot path never touches the filesystem.
    """
    _count_call(slot)
    _write_execution_once("completed", slot)


def failed(
    slot: str,
    error: BaseException,
    *,
    phase: str = "entry",
    entry: Callable[..., object] | None = None,
) -> None:
    """Record that the selected implementation for ``slot`` raised.

    Written by the dispatcher on its way out, before the exception reaches the
    scheduler. The engine still dies — nothing serves stock in a candidate's
    name — but the receipt outlives the rank, so the closing swap or the session
    worker can report "your kernel raised X" instead of "the validator's lane
    failed". Never raises.
    """

    _write_execution_once(
        "failed", slot, error=error, phase=phase, entry=entry
    )


def invoke(
    slot: str,
    entry: Callable[..., object],
    *args: object,
    phase: str = "entry",
) -> object:
    """Run the selected implementation; a raise is receipted before it propagates.

    There is no fallback: the exception still takes the engine down. The receipt
    is what lets the closing swap or the session worker say "the candidate raised
    <Type> in <slot>" instead of reporting the lane as broken.
    """

    try:
        return entry(*args)
    except BaseException as exc:
        failed(slot, exc, phase=phase, entry=entry)
        raise


def not_selected(slot: str, outcome: str, mismatches: Iterable) -> None:
    """Record why a live call routed to stock while a candidate was registered.

    Without this, "registered but never ran" is one shape on disk covering three
    unrelated causes: the declared domain never matched a live call, the seam
    never fired at all, or the entry was never reached. They need different
    fixes, and the registry already computes which one it is and then discards
    it. One receipt per slot holds every distinct reason seen, so the reader gets
    the whole routing story from one file.

    Keyed on the field names and reasons, never on observed values: a call that
    is out of domain on ``num_tokens`` says so once, not once per token count.
    """

    reasons = _NOT_SELECTED.setdefault(slot, {})
    detail = tuple(
        (str(m.field), str(m.reason), str(m.expected)) for m in mismatches
    )
    key = (outcome, detail)
    if key in reasons:
        return
    reasons[key] = {
        "outcome": outcome,
        "fields": [field for field, _reason, _expected in detail],
        "mismatches": [
            {"field": field, "reason": reason, "expected": expected}
            for field, reason, expected in detail
        ],
    }
    rdir = _dir()
    if not rdir:
        return
    try:
        _write_to(
            _resolved_dir(rdir),
            "not_selected",
            {"slot": slot, "reasons": list(reasons.values())},
            tag=slot,
        )
    except Exception:  # noqa: BLE001 - diagnostics never break an engine
        logger.exception("cacheon: not-selected receipt failed")


def flush_calls() -> None:
    """Persist the current invocation counts into this scope's receipts.

    Rewrites rather than appends: one receipt per slot per process holds the
    running total, so a reader never has to sum files and can never double count.
    Never raises — accounting must not be able to kill an engine.
    """

    rdir = _dir()
    if not rdir or not _CALLS:
        return
    try:
        root = _resolved_dir(rdir)
        for slot in list(_CALLS):
            _write_to(root, "completed", {"slot": slot, **_calls_payload(slot)}, tag=slot)
    except Exception:  # noqa: BLE001 - diagnostics never break an engine
        logger.exception("cacheon: receipt call flush failed")


# A one-shot engine is read after it exits, so its final counts must reach disk
# without anyone asking. The resident lane flushes at every swap and does not
# depend on this.
atexit.register(flush_calls)


class ReceiptFormatError(RuntimeError):
    """A receipt file exists but is unreadable or not a JSON object."""


def collect(rdir: str | Path, kind: str) -> list[dict]:
    """Strictly read all receipts of ``kind``; malformed evidence fails closed."""
    out: list[dict] = []
    root = Path(rdir)
    if not root.is_dir():
        return out
    for p in sorted(root.glob(f"{kind}*.json")):
        try:
            payload = json.loads(p.read_text())
        except (OSError, ValueError) as exc:  # noqa: PERF203
            raise ReceiptFormatError(f"invalid receipt {p}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ReceiptFormatError(f"invalid receipt {p}: expected a JSON object")
        out.append(payload)
    return out


_EXECUTION_KINDS = ("active", "completed", "failed", "load_failed")


def rows_for_scope(scope: object, *, pid: Optional[int] = None) -> Optional[dict]:
    """One scope's receipts by kind, or ``None`` when they are unobservable.

    The tri-state is the whole point. ``None`` means the evidence path itself is
    unusable (no root established, no such scope, or a malformed receipt) and no
    verdict may be drawn from it; an empty dict means the scope existed and
    nothing executed under it, which is a fact about the candidate rather than
    the plumbing.

    Passing ``pid`` restricts the reading to receipts this process wrote itself.
    That is race-free across a TP group sharing one root: a rank finishes its own
    receipts before it acknowledges the swap that closes the scope, whereas a
    global reading could observe a peer mid-write.

    Never raises: a diagnostic must not be able to kill an engine.
    """

    try:
        root = _root()
        if not root:
            return None
        cleaned = "" if scope is None else _SAFE_RE.sub("_", str(scope))[:64]
        if not cleaned or not cleaned[0].isalnum() or ".." in cleaned:
            return None
        if cleaned == _SCOPE and pid == os.getpid():
            # Reading our own live scope: the in-memory tally is ahead of the file.
            flush_calls()
        directory = _resolved_dir(os.path.join(root, cleaned))
        if not directory.is_dir():
            return None
        out: dict = {}
        for kind in (*_EXECUTION_KINDS, "not_selected"):
            rows = collect(directory, kind)
            if pid is not None:
                rows = [row for row in rows if row.get("pid") == pid]
            if rows:
                out[kind] = rows
        return out
    except Exception:  # noqa: BLE001 - unreadable evidence is unobservable, not zero
        logger.exception("cacheon: receipt scope rows failed (%s)", scope)
        return None


def require(rdir: str | Path, kind: str, *, context: str) -> list[dict]:
    """Return receipts of ``kind`` or raise with a diagnosis — the eval-side gate."""
    got = collect(rdir, kind)
    if got:
        return got
    failed = collect(rdir, "load_failed")
    if failed:
        raise RuntimeError(
            f"{context}: seam rank(s) failed bundle activation "
            f"(load_failed receipts: {failed}). The engine stopped; fix the named "
            "cause before another launch."
        )
    raise RuntimeError(
        f"{context}: no '{kind}' seam receipt was written by any engine rank. The candidate "
        "ran WITHOUT the miner kernel (stock-vs-stock) — likely missing cacheon.pth bootstrap "
        "in the engine interpreter, CACHEON env not reaching spawned ranks, or the seamed "
        "module was never imported by this engine config. Refusing to score a phantom."
    )


def _exact_int(value, *, minimum: int = 0) -> Optional[int]:
    if type(value) is not int or value < minimum:  # bool is intentionally invalid
        return None
    return value


def _validated_identity(receipt: dict) -> tuple[int, int, int] | None:
    pid = _exact_int(receipt.get("pid"), minimum=1)
    rank = _exact_int(receipt.get("rank"), minimum=-1)
    world_size = _exact_int(receipt.get("world_size"), minimum=-1)
    if pid is None or rank is None or world_size is None:
        return None
    if (rank, world_size) == (-1, -1):
        return pid, rank, world_size
    if world_size < 1 or rank < 0 or rank >= world_size:
        return None
    return pid, rank, world_size


def _expected_members(
    members: list[dict],
    *,
    expected_member_count: int | None,
) -> tuple[
    list[str],
    list[dict],
    list[dict],
    bool,
    dict[int, tuple[int, int]],
]:
    """Resolve members without allowing observed completions to hide a silent rank.

    The roster comes from the ``active`` receipts and nothing else. Deriving it
    from the completions instead — which this function used to do when handed no
    members — lets a rank that silently stopped reporting shrink the roster to
    exactly the ranks that did report, so short coverage reads as full coverage.
    Every production caller has always supplied the roster; the derivation was
    reachable only from its own tests.
    """
    malformed: list[dict] = []
    duplicates: list[dict] = []
    seen: set[int] = set()
    member_identities: dict[int, tuple[int, int]] = {}
    for receipt in members:
        ident = _validated_identity(receipt)
        if ident is None:
            malformed.append(receipt)
            continue
        pid = ident[0]
        if pid in seen:
            duplicates.append(receipt)
        seen.add(pid)
        member_identities[pid] = (ident[1], ident[2])
    labels = [f"pid:{pid}" for pid in sorted(seen)]
    count_ok = (
        bool(members)
        and (expected_member_count is None or len(labels) == expected_member_count)
    )
    known = [identity for identity in member_identities.values() if identity != (-1, -1)]
    if known and len(known) != len(member_identities):
        malformed.extend(members)
    elif known:
        world_sizes = {world_size for _rank, world_size in known}
        ranks = [rank for rank, _world_size in known]
        if len(world_sizes) != 1:
            malformed.extend(members)
        else:
            world_size = next(iter(world_sizes))
            if (
                world_size != len(member_identities)
                or (expected_member_count is not None
                    and world_size != expected_member_count)
                or set(ranks) != set(range(world_size))
                or len(set(ranks)) != len(ranks)
            ):
                malformed.extend(members)
                count_ok = False
    return labels, malformed, duplicates, count_ok, member_identities


def coverage_matrix(
    observed: Iterable[dict],
    *,
    expected_slots: Iterable[str],
    member_receipts: Iterable[dict],
    expected_member_count: int | None = None,
) -> dict:
    """Build fail-closed per-slot/per-member diagnostic coverage.

    Presence, not volume: one completion per slot per member is the whole gate.
    Invocation counts belong on the receipt (see ``flush_calls``) where they are
    reported to operators and miners; they are deliberately not a threshold here,
    because how many times a captured kernel re-enters Python is a property of
    CUDA graphs rather than of the candidate.
    """
    got = list(observed)
    members = list(member_receipts)
    raw_slots = list(expected_slots)
    if any(not isinstance(slot, str) or not slot for slot in raw_slots):
        raise ValueError("expected_slots must contain non-empty strings")
    slots = sorted(set(raw_slots))
    if expected_member_count is not None and (
        type(expected_member_count) is not int or expected_member_count < 1
    ):
        raise ValueError("expected_member_count must be a positive integer or None")
    (
        expected_members,
        malformed,
        duplicates,
        member_count_ok,
        active_identities,
    ) = _expected_members(members, expected_member_count=expected_member_count)
    expected_pairs = {
        (slot, member) for slot in slots for member in expected_members
    }
    counts: dict[tuple[str, str], int] = {}
    unexpected: list[dict] = []
    for receipt in got:
        slot = receipt.get("slot")
        ident = _validated_identity(receipt)
        if not isinstance(slot, str) or not slot or ident is None:
            malformed.append(receipt)
            continue
        member = f"pid:{ident[0]}"
        active_identity = active_identities.get(ident[0])
        if (
            active_identity is not None
            and active_identity != (-1, -1)
            and active_identity != (ident[1], ident[2])
        ):
            malformed.append(receipt)
            continue
        if slot not in slots or member not in expected_members:
            unexpected.append(receipt)
            continue
        key = (slot, member)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > 1:
            duplicates.append(receipt)

    if active_identities and all(
        identity == (-1, -1) for identity in active_identities.values()
    ):
        # Bootstrap may precede process-group initialization, so early active
        # receipts can be PID-only. For multi-member execution, the later
        # completions must then prove one coherent distributed identity per PID.
        target_count = expected_member_count or len(active_identities)
        derived: dict[int, set[tuple[int, int]]] = {
            pid: set() for pid in active_identities
        }
        for receipt in got:
            ident = _validated_identity(receipt)
            if ident is not None and ident[0] in derived:
                derived[ident[0]].add((ident[1], ident[2]))
        if target_count > 1:
            identities = [
                next(iter(values))
                for values in derived.values()
                if len(values) == 1 and (-1, -1) not in values
            ]
            if (
                len(identities) != len(derived)
                or {world for _rank, world in identities} != {target_count}
                or {rank for rank, _world in identities} != set(range(target_count))
            ):
                malformed.extend(got)
        else:
            known = {
                identity
                for values in derived.values()
                for identity in values
                if identity != (-1, -1)
            }
            if known and known != {(0, 1)}:
                malformed.extend(got)
    present = {pair for pair, count in counts.items() if count >= 1}
    missing = sorted(expected_pairs - present)
    return {
        "ok": (
            bool(slots)
            and bool(expected_members)
            and member_count_ok
            and not missing
            and not malformed
            and not unexpected
            and not duplicates
        ),
        "expected_slots": slots,
        "members": expected_members,
        "expected_member_count": expected_member_count,
        "observed_member_count": len(expected_members),
        "member_count_ok": member_count_ok,
        "expected_pairs": len(expected_pairs),
        "covered_pairs": len(expected_pairs & present),
        "missing": [
            {"slot": slot, "member": member} for slot, member in missing
        ],
        "malformed": malformed,
        "unexpected": unexpected,
        "duplicates": duplicates,
    }


def completed_gate(
    completed_receipts: Iterable[dict],
    *,
    expected_slots: Iterable[str],
    member_receipts: Iterable[dict],
    expected_member_count: int | None = None,
) -> tuple[bool, str]:
    """Require one completion per expected slot and member."""
    complete = list(completed_receipts)
    members = list(member_receipts)
    detail = coverage_matrix(
        complete,
        expected_slots=expected_slots,
        member_receipts=members,
        expected_member_count=expected_member_count,
    )
    ok = detail["ok"]
    desc = (
        f"completed coverage {detail['covered_pairs']}/{detail['expected_pairs']} "
        "slot/member pairs"
    )
    if detail["missing"]:
        desc += f"; missing={detail['missing']}"
    if detail["malformed"]:
        desc += f"; malformed={len(detail['malformed'])}"
    if detail["unexpected"]:
        desc += f"; unexpected={len(detail['unexpected'])}"
    if detail["duplicates"]:
        desc += f"; duplicates={len(detail['duplicates'])}"
    if not detail["member_count_ok"]:
        desc += (
            f"; members={detail['observed_member_count']}"
            f"/{detail['expected_member_count']}"
        )
    return ok, desc
