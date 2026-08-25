"""Turn an evaluation product into the answer a miner asked for.

Everything a miner needs is already produced and retained. It is also base64
inside digest-wrapped envelopes, so in practice a miner is told
``decision=FAIL, reason=candidate_slower`` and nothing else — not whether their
code ran, not whether it was correct, not by how much it lost, and above all not
whether the failure was theirs or ours. That gap is not missing evidence. It is
missing rendering.

Two rules shape this module:

* **It knows no slot.** Slot identity arrives as data on every path — read from
  the product, printed, never branched on. A slot registered tomorrow renders
  today. Same for lane, target, model, and arm count.
* **A partial product still renders.** The products worth explaining are the
  ones from runs that went wrong, and those are exactly the products missing
  sections. Every lookup tolerates absence and says what is absent, because
  "the speed section is not here" is itself the answer to why a bundle has no
  speed verdict.
"""

from __future__ import annotations

import base64
import json
import re
import statistics
from collections.abc import Iterable
from typing import Any

from cacheon.eval.continuation_codec import ContinuationCodecError
from cacheon.eval.resident_execution_evidence import (
    EXECUTION_CODEC,
    RankExecution,
    eager_slots,
)
from cacheon.kernel_trace import format_kernels
from cacheon.receipts import validator_runtime_failure

#: Kept in step with ``engine_worker.EXECUTION_SUMMARY_PREFIX``. Duplicated as a
#: literal rather than imported: this renderer must read a log written by any
#: worker version, including one older than the module it would import from.
_SUMMARY_PREFIX = "CACHEON-EXECUTION-SUMMARY: "
_CONFIG_PREFIX = "CACHEON-ENGINE-CONFIG: "
_TRACE_FRAME = re.compile(
    r'^  File "(?P<path>[^"\n]+)", line (?P<line>[1-9][0-9]*), in (?P<call>[^\n]+)$',
    re.MULTILINE,
)
_CANDIDATE_FRAME = re.compile(
    r"/swap-intake/(?P<bundle>[0-9a-f]{64})/(?P<source>[^\"\n]+)"
)
_TRACE_ERROR = re.compile(
    r"^(?P<kind>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s+(?P<message>[^\n]+)$",
    re.MULTILINE,
)

#: Internally the timed runs are named B, C, B'. A miner has no reason to know
#: that vocabulary, and using it in their report makes the one number they care
#: about — did my kernel help — harder to read, not more precise. Unknown roles
#: print their raw name rather than being dropped.
_ARM_LABEL = {
    "B": "SGLang alone (before)",
    "C": "with your kernel",
    "B_prime": "SGLang alone (after)",
    "C_prime": "with your kernel (repeat)",
    "B_double_prime": "SGLang alone (third time)",
}


def _tagged(stderr: str, prefix: str) -> Iterable[dict]:
    """Every JSON object on a line carrying ``prefix``, corrupt ones skipped.

    Searches within the line rather than anchoring at its start: these are
    written to a stream a container runtime prefixes with its own timestamps,
    and dropping the evidence over a log format is not a trade worth making.
    """

    for line in stderr.splitlines():
        marker = line.find(prefix)
        if marker < 0:
            continue
        try:
            parsed = json.loads(line[marker + len(prefix):])
        except ValueError:
            continue
        if isinstance(parsed, dict):
            yield parsed


def config_lines(stderr: object) -> list[str]:
    """State the engine settings each arm ran under, and any difference between them.

    A speed pair is only a measurement if both arms were built the same way. The
    settings decide which backend serves an unclaimed chokepoint, so a difference
    here is not a detail -- it is the comparison being invalid. When one log holds
    both arms this says so outright; when it holds one, it still puts the settings
    on the record where they can be compared later.
    """

    if not isinstance(stderr, str) or not stderr:
        return []
    by_arm: dict[str, dict] = {}
    for row in _tagged(stderr, _CONFIG_PREFIX):
        engine = row.get("engine")
        if isinstance(engine, dict):
            by_arm[str(row.get("arm") or "unnamed")] = engine
    if not by_arm:
        return []
    label = {"candidate": "with your kernel", "stock": "SGLang alone"}
    lines = []
    for arm, engine in sorted(by_arm.items()):
        settings = ", ".join(f"{k}={v}" for k, v in sorted(engine.items()))
        lines.append(f"  {label.get(arm, arm):<26s} {settings}")
    if len(by_arm) > 1:
        arms = sorted(by_arm)
        keys = {k for engine in by_arm.values() for k in engine}
        differing = sorted(k for k in keys if len({str(by_arm[a].get(k)) for a in arms}) > 1)
        lines.append(
            f"  {'SETTINGS DIFFER':<26s} " + ", ".join(differing)
            + " — the two runs were not set up the same way, so comparing them "
            "proves nothing"
            if differing
            else f"  {'settings match':<26s} both runs were set up identically"
        )
    return lines


def _get(value: object, *path: str, default: Any = None) -> Any:
    """Walk nested dicts, returning ``default`` the moment the path breaks."""

    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _number(value: object) -> float | None:
    """Rates cross the wire as strings to survive float round-tripping."""

    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _decoded_evidence(product: dict) -> list[tuple[str, dict]]:
    """``(domain, payload)`` for every readable evidence row, in product order.

    An undecodable blob is not fatal here. This renders what a run produced; a
    reader that refused to explain the readable half because one blob was
    corrupt would fail exactly when a miner most needs an explanation.
    """

    found: list[tuple[str, dict]] = []
    rows = product.get("evidence")
    for row in rows if isinstance(rows, list) else []:
        domain = _get(row, "reference", "domain")
        raw = row.get("payload_base64") if isinstance(row, dict) else None
        if not isinstance(domain, str) or not isinstance(raw, str):
            continue
        try:
            payload = json.loads(base64.b64decode(raw))
        except Exception:  # noqa: BLE001 - a corrupt blob is reported, not raised
            continue
        if isinstance(payload, dict):
            found.append((domain, payload))
    return found


def _evidence(decoded: list[tuple[str, dict]], suffix: str) -> list[dict]:
    """Every evidence payload whose domain kind is ``suffix``, ignoring the stage.

    Domains are ``<stage>.<kind>`` — ``qualification.stage-exit``,
    ``screen.stage-exit``. Matching the kind alone is what lets one renderer
    serve every stage instead of growing a branch per lane.
    """

    return [
        payload for domain, payload in decoded if domain.split(".", 1)[-1] == suffix
    ]


def _first(decoded: list[tuple[str, dict]], suffix: str) -> dict:
    found = _evidence(decoded, suffix)
    return found[0] if found else {}


def _shape_rows(graph_evidence: dict) -> Iterable[tuple[str, str, dict]]:
    """Yield ``(slot, variant, shape)`` for every verified shape, in order."""

    members = graph_evidence.get("members")
    for member in members if isinstance(members, list) else []:
        variants = member.get("variants") if isinstance(member, dict) else None
        for variant in variants if isinstance(variants, list) else []:
            if not isinstance(variant, dict):
                continue
            slot = str(variant.get("slot_id") or _get(member, "slot_id") or "?")
            name = str(variant.get("variant_id") or "default")
            shapes = variant.get("shapes")
            for shape in shapes if isinstance(shapes, list) else []:
                if isinstance(shape, dict):
                    yield slot, name, shape


def _correctness_lines(graph_evidence: dict) -> list[str]:
    if not graph_evidence:
        return [f"  {'correctness':<26s} not checked — the run stopped before we tested it"]
    rows = list(_shape_rows(graph_evidence))
    if not rows:
        return [f"  {'correctness':<26s} no input sizes were tested"]

    total = len(rows)
    skipped = [r for r in rows if not r[2].get("applicable")]
    eager_ok = [r for r in rows if r[2].get("eager_passed")]
    graph_needed = [r for r in rows if r[2].get("graph_required")]
    graph_ok = [r for r in graph_needed if r[2].get("graph_passed")]
    replays = {int(r[2]["graph_replays"]) for r in graph_ok if isinstance(r[2].get("graph_replays"), int)}

    lines = [
        f"  {'correctness':<26s} {len(eager_ok)}/{total} input sizes matched the"
        f" reference output"
    ]
    if graph_needed:
        detail = f" ({min(replays)} replays each)" if len(replays) == 1 else ""
        lines.append(
            f"  {'CUDA graph':<26s} {len(graph_ok)}/{len(graph_needed)} input sizes"
            f" captured and replayed cleanly{detail}"
        )
    else:
        lines.append(
            f"  {'CUDA graph':<26s} not required — this slot runs outside the graph"
        )

    for slot, variant, shape in rows:
        kind = shape.get("failure_kind")
        if kind and kind != "none":
            lines.append(f"  {'':<26s} FAILED {slot} [{variant}] — {kind}")
    for slot, variant, _shape in skipped:
        # Not a failure, and the most misread outcome there is: the code was
        # never asked to run this shape because the miner's own declared domain
        # excluded it.
        lines.append(
            f"  {'':<26s} not tested {slot} [{variant}] — you said your kernel does not"
            f" handle this input size"
        )
    for member in graph_evidence.get("members") or []:
        for variant in (member.get("variants") if isinstance(member, dict) else None) or []:
            if isinstance(variant, dict) and variant.get("domain_coverage_complete") is False:
                lines.append(
                    f"  {'':<26s} warning {variant.get('slot_id')} [{variant.get('variant_id')}] — "
                    "you said your kernel handles fewer input sizes than the model"
                    " actually asks this slot for"
                )
    return lines


def _speed_lines(stage_exit: dict) -> list[str]:
    rates = _get(stage_exit, "speed_witness", "rates", default=[])
    rates = [r for r in rates if isinstance(r, dict)] if isinstance(rates, list) else []
    if not rates:
        return [f"  {'speed':<26s} not measured — the run stopped before timing"]

    lines: list[str] = []
    measured: dict[str, float] = {}
    for row in rates:
        role = str(row.get("role") or "?")
        seconds = _number(row.get("timed_seconds"))
        tokens = _number(row.get("timed_tokens"))
        if not seconds or not tokens:
            lines.append(f"  {_ARM_LABEL.get(role, role):<26s} recorded, but its timing is unusable")
            continue
        rate = tokens / seconds
        measured[role] = rate
        windows = row.get("windows")
        windows = [w for w in windows if isinstance(w, dict)] if isinstance(windows, list) else []
        spread = _window_spread(windows)
        spread_text = f", varying by {spread * 100:.2f}%" if spread is not None else ""
        detail = f"  ({len(windows)} runs{spread_text})" if windows else ""
        lines.append(
            f"  {_ARM_LABEL.get(role, role):<26s} {rate:10.2f} tokens/sec{detail}"
        )

    # Compared against the SLOWEST SGLang-only run, which is the reading most
    # generous to the miner. Saying which rule was used matters more than the
    # number, because a miner who disagrees can then recompute it themselves.
    sglang = [rate for role, rate in measured.items() if role.startswith("B")]
    with_kernel = measured.get("C")
    if sglang and with_kernel:
        ratio = with_kernel / min(sglang)
        lines.append(
            f"  {'result':<26s} your kernel is {max(ratio, 1 / ratio):.2f}x "
            f"{'FASTER' if ratio > 1 else 'SLOWER'} than SGLang "
            f"({ratio:.4f}x, measured against the slower SGLang-only run)"
        )
        if len(sglang) > 1:
            # The machine's own run-to-run variation. If it is bigger than the
            # difference the kernel showed, the number measured the machine.
            noise = (max(sglang) - min(sglang)) / min(sglang)
            claim = abs(ratio - 1.0)
            lines.append(
                f"  {'machine noise':<26s} SGLang alone measured {noise * 100:.1f}% "
                f"apart on the same hardware"
            )
            if noise > claim:
                lines.append(
                    f"  {'NOT A REAL RESULT':<26s} that noise is larger than the "
                    f"{claim * 100:.1f}% your kernel changed things by, so this "
                    "comparison cannot tell them apart"
                )
    return lines


def _window_spread(windows: list[dict]) -> float | None:
    """How much the timed runs disagreed; ``None`` when it cannot be computed.

    Deliberately the SAME statistic as ``ResidentSpeedPolicy.read_window_scatter``
    — median absolute deviation about the median, relative to the median — and
    not the max-minus-min it used to be. That difference is not cosmetic: on a
    real crowned run the two read 0.57% and 8.22% on identical evidence, so a
    miner reading this page would have concluded their run was twelve times
    noisier than the gate that actually judged it. A report that disagrees with
    the gate is worse than no report, because both look authoritative.
    """

    rates = []
    for window in windows:
        seconds = _number(window.get("seconds"))
        tokens = _number(window.get("tokens"))
        if seconds and tokens:
            rates.append(tokens / seconds)
    if len(rates) < 3:
        return None
    median = statistics.median(rates)
    if median <= 0:
        return None
    return statistics.median([abs(rate - median) for rate in rates]) / median


def _headline(execution: list[str], speed: list[str]) -> str:
    """The one sentence that must be read before any number below it.

    Ordered by what disqualifies what. A kernel that never ran makes the speed
    numbers meaningless, and a machine noisier than the effect makes them
    meaningless too — either way saying so afterwards is too late.

    Matches on the labels this module itself emits rather than re-deriving the
    facts from the evidence, so there is exactly one place that decides whether
    a run counts, and the headline can never disagree with the section below it.
    """

    joined = "\n".join(execution)
    if "NEVER RAN" in joined:
        return (
            "VERDICT  your kernel never ran. Every speed number below was "
            "measured without it — ignore them."
        )
    if "FAILED TO LOAD" in joined:
        return (
            "VERDICT  your kernel failed to load on at least one GPU, so part "
            "of this run was SGLang, not you."
        )
    if "VALIDATOR RUNTIME FAILURE" in joined:
        return (
            "VERDICT  an installed validator library failed on its runtime files. "
            "This is a validator setup failure, not a bundle failure."
        )
    if "RAISED" in joined:
        return (
            "VERDICT  your kernel raised an exception and the engine went down "
            "with it. This is attributed to the bundle, not to the validator."
        )
    if any("NOT A REAL RESULT" in line for line in speed):
        return (
            "VERDICT  this run cannot tell your kernel apart from normal "
            "machine variation. The speedup below is not evidence."
        )
    return ""


def explain(product: object, *, stderr: object = None) -> list[str]:
    """Render one evaluation product as the story of what happened to a bundle.

    What the ranks did comes from the product's own ``execution`` evidence when
    the run published it. ``stderr`` is the retained worker log, for runs that
    did not: it carries the same per-rank record as one printed line per rank.
    """

    if not isinstance(product, dict):
        return ["This is not an evaluation product."]
    decoded = _decoded_evidence(product)
    stage_exit = _first(decoded, "stage-exit")

    # Identity only as a heading. Which reservation, hotkey, and verdict this is
    # belongs to the durable record that ``chain.miner_feedback`` already reports;
    # restating it here would create a second answer that can disagree.
    targets = sorted(
        {
            str(row.get("target_id"))
            for row in _get(product, "authority_manifest", "reservations", default=[]) or []
            if isinstance(row, dict) and row.get("target_id")
        }
    )
    lines = [f"evidence for {', '.join(targets) or 'an unnamed target'}"]
    execution: list[str] = []
    for payload in _evidence(decoded, "execution"):
        for swap in payload.get("swaps") or []:
            if not isinstance(swap, dict):
                continue
            ranks = _typed_ranks(swap.get("ranks"))
            execution.append(
                f"  {'generation':<26s} {swap.get('generation')} on lane "
                f"{swap.get('lane_id')}: {swap.get('executed_ranks')} of "
                f"{swap.get('expected_ranks')} GPU(s) ran your kernel cleanly"
            )
            execution.extend(execution_lines(ranks))
    if not execution and stderr is not None:
        ranks = ranks_from_log(stderr)
        execution = ([] if not ranks else execution_lines(ranks))
        execution += failure_lines(stderr) + _path_lines(_log_rows(stderr))
        if not execution:
            execution = execution_lines(())
    # The headline exists because the sections below can be read in the wrong
    # order. A run where the kernel never executed still produces three speed
    # numbers and a ratio, and a miner who reads that first walks away believing
    # they were 1.37x faster. Say the disqualifying fact before the numbers, or
    # the numbers do the talking.
    headline = _headline(execution, _speed_lines(stage_exit))
    if headline:
        lines.append("")
        lines.append(headline)
    if execution:
        lines.append("")
        lines.append("did your kernel run")
        lines.extend(execution)
    lines.append("")
    lines.append("did it work")
    lines.extend(_correctness_lines(_first(decoded, "graph-verification")))
    lines.append("")
    lines.append("was it faster")
    lines.extend(_speed_lines(stage_exit))
    settings = config_lines(stderr)
    if settings:
        lines.append("")
        lines.append("what we ran you under")
        lines.extend(settings)
    if not decoded:
        lines.append("")
        lines.append(
            "note           this product carries no readable evidence at all; the run"
            " ended before it produced any."
        )
    return lines


def _trace_path(path: str, bundle: str) -> str:
    candidate = f"/swap-intake/{bundle}/"
    if candidate in path:
        return path.split(candidate, 1)[1]
    for marker in ("/site-packages/", "/dist-packages/", "/sglang/python/"):
        if marker in path:
            return path.split(marker, 1)[1]
    return path.rsplit("/", 1)[-1]


def failure_lines(stderr: object) -> list[str]:
    """Extract candidate-owned tracebacks from a retained worker stream."""

    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")
    if not isinstance(stderr, str) or not stderr:
        return []
    failures: dict[tuple[str, str, int, str, str, str], dict] = {}
    for frame in _TRACE_FRAME.finditer(stderr):
        candidate = _CANDIDATE_FRAME.search(frame.group("path"))
        if candidate is None:
            continue
        boundary = stderr.find("\n\n", frame.end())
        segment_end = len(stderr) if boundary < 0 else boundary
        error = _TRACE_ERROR.search(stderr, frame.end(), segment_end)
        if error is None:
            continue
        message = error.group("message")
        summary = message.split(" GPU ", 1)[0]
        key = (
            candidate.group("bundle"),
            candidate.group("source"),
            int(frame.group("line")),
            frame.group("call").strip(),
            error.group("kind"),
            summary,
        )
        row = failures.setdefault(
            key, {"devices": set(), "chain": [], "occurrences": 0}
        )
        row["occurrences"] += 1
        device = re.search(r"\bGPU ([0-9]+)\b", message)
        if device:
            row["devices"].add(int(device.group(1)))
        chain = [
            f"{_trace_path(nested.group('path'), key[0])}:"
            f"{nested.group('line')} {nested.group('call').strip()}"
            for nested in _TRACE_FRAME.finditer(stderr, frame.start(), error.start())
        ]
        if len(chain) > len(row["chain"]):
            row["chain"] = chain
    lines = []
    for (bundle, source, line, phase, kind, message), row in failures.items():
        label = (
            "VALIDATOR RUNTIME FAILURE"
            if validator_runtime_failure(kind, message)
            else "RAISED"
        )
        verb = "reached" if label == "VALIDATOR RUNTIME FAILURE" else "raised"
        lines.append(f"  {label:<26s} {source}:{line} in {phase} {verb} {kind}")
        lines.append(f"  {'bundle':<26s} {bundle}")
        lines.append(f"  {'error':<26s} {message}")
        if row["devices"]:
            devices = ", ".join(str(value) for value in sorted(row["devices"]))
            lines.append(f"  {'affected GPU/ranks':<26s} {devices}")
        elif row["occurrences"] > 1:
            lines.append(
                f"  {'affected GPU/ranks':<26s} {row['occurrences']} rank tracebacks; "
                "rank IDs absent from stream"
            )
        if row["chain"]:
            lines.append(f"  {'call chain':<26s} " + " -> ".join(row["chain"]))
    return lines


def _typed_ranks(value: object) -> tuple[RankExecution, ...]:
    """Rows as published, skipping any the renderer cannot read."""

    ranks = []
    for row in value if isinstance(value, list) else []:
        try:
            typed = EXECUTION_CODEC.decode(row)
        except ContinuationCodecError:
            continue
        if type(typed) is RankExecution:
            ranks.append(typed)
    return tuple(ranks)


def _path_lines(rows: list[dict]) -> list[str]:
    """Name the device kernels the bundle launched, per input shape.

    This is the answer to "which of my branches ran", and only a retained log
    carries it: the kernel trace is an audit-arm instrument, not part of the
    generation evidence. Ranks are merged by signature rather than listed
    separately: they execute the same code on the same shapes, so a per-rank
    listing would repeat one fact world size times, and a signature that
    appears on only some ranks is the interesting case precisely because it is
    rare.
    """

    by_signature: dict[str, dict[str, int]] = {}
    for row in rows:
        recorded = row.get("kernels")
        if not isinstance(recorded, dict):
            continue
        for signature, counts in recorded.items():
            if not isinstance(counts, dict):
                continue
            merged = by_signature.setdefault(str(signature), {})
            for name, count in counts.items():
                merged[str(name)] = max(merged.get(str(name), 0), int(count or 0))
    if not by_signature:
        return []
    lines = [f"  {'':<26s} which path your code took (GPU kernels, one sample per input size):"]
    for signature, counts in sorted(by_signature.items()):
        lines.append(f"  {'':<26s}   at {signature}")
        lines.extend(format_kernels(counts, indent=" " * 32))
    return lines


def _log_rows(stderr: object) -> list[dict]:
    """Every ``completed`` receipt row in a retained log, across its rank lines."""

    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")
    if not isinstance(stderr, str):
        return []
    return [
        row
        for parsed in _tagged(stderr, _SUMMARY_PREFIX)
        for row in (parsed.get("completed") or [])
        if isinstance(row, dict)
    ]


def ranks_from_log(stderr: object) -> tuple[RankExecution, ...]:
    """Read the per-rank execution record back out of a retained stderr stream.

    The worker's receipt directory is deleted when its container goes, so the
    per-rank facts reach a reader of the log only as machine-readable lines the
    host retained. Each resident rank prints its own line; the one-shot worker
    prints one line holding every rank's rows with their rank identity. Rows
    without an identity are taken to belong to the line they were printed on.

    Tolerates the stream being partial: stderr is retained as a bounded prefix,
    so the marker may be missing or a line cut in half, and neither is a reason
    to explain nothing.
    """

    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")
    if not isinstance(stderr, str):
        return ()
    by_rank: dict[int, dict[str, list]] = {}
    for index, parsed in enumerate(_tagged(stderr, _SUMMARY_PREFIX)):
        for kind, rows in parsed.items():
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                rank = row.get("rank")
                rank = rank if type(rank) is int and rank >= 0 else index
                by_rank.setdefault(rank, {}).setdefault(kind, []).append(row)
    ranks = []
    for rank, rows in sorted(by_rank.items()):
        try:
            ranks.append(RankExecution.from_receipts(rank, rows))
        except ValueError:
            continue
    return tuple(ranks)


def execution_lines(ranks: tuple[RankExecution, ...] | list[RankExecution]) -> list[str]:
    """Render what the ranks recorded for one closed generation."""

    if not ranks:
        return [f"  {'did not record':<26s} the run produced no execution record"]
    lines: list[str] = []
    loaded = [row for row in ranks if row.loaded]
    if loaded:
        slots = sorted({slot.slot for row in loaded for slot in row.slots})
        lines.append(
            f"  {'loaded':<26s} on {len(loaded)} GPU(s); it took over "
            f"{', '.join(slots) or 'nothing'}"
        )
    for row in ranks:
        if row.load_error:
            lines.append(
                f"  {'FAILED TO LOAD':<26s} GPU {row.rank}: {row.load_error} — that GPU "
                "ran SGLang's own kernel instead of yours"
            )
    eager = eager_slots()
    by_slot: dict[str, list[tuple[int, object]]] = {}
    for row in ranks:
        for slot in row.slots:
            by_slot.setdefault(slot.slot, []).append((row.rank, slot))
    called_anything = False
    for name, entries in sorted(by_slot.items()):
        facts = [slot for _rank, slot in entries]
        calls = [slot.calls for slot in facts if slot.calls >= 0]
        if not any(calls):
            continue
        called_anything = True
        captured = [slot.captured for slot in facts]
        # ``all``, not ``any``: one GPU serving SGLang's kernel out of a captured
        # graph makes the whole measurement SGLang's, so a partial yes is a no.
        graph = (
            "outside the CUDA graph, where SGLang runs this slot eagerly"
            if name in eager
            else "inside the CUDA graph on every GPU"
            if captured and all(c is True for c in captured)
            else "NOT inside the CUDA graph on every GPU — the timed runs would "
            "replay SGLang's kernel, not yours"
            if any(c is False for c in captured)
            else "we did not record whether it was inside the CUDA graph"
        )
        total = (
            f"called {sum(calls):,} times" if len(calls) == len(facts)
            else "we did not record the call count"
        )
        lines.append(
            f"  {'ran':<26s} {name}: {total} across {len(facts)} GPU(s), {graph}"
        )
        for rank, slot in entries:
            if slot.error:
                lines.append(
                    f"  {'RAISED':<26s} {name} on GPU {rank}: {slot.error} — the "
                    "engine went down with it; nothing served SGLang's kernel in "
                    "your name"
                )
    if loaded and not called_anything and not any(
        slot.error for row in ranks for slot in row.slots
    ):
        lines.append(
            f"  {'NEVER RAN':<26s} your kernel loaded and took over the slot, but "
            "the model never called it — nothing measured here is your kernel"
        )
    for row in ranks:
        for slot in row.slots:
            for reason in slot.skipped:
                lines.append(
                    f"  {'SKIPPED YOUR KERNEL':<26s} {slot.slot} on GPU {row.rank}: "
                    f"the model called this slot, but we ran SGLang's kernel "
                    f"instead ({reason})"
                )
    return lines or [f"  {'nothing recorded':<26s} the run produced no execution record"]


__all__ = [
    "config_lines",
    "execution_lines",
    "explain",
    "failure_lines",
    "ranks_from_log",
]
