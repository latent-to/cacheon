"""Canonical OCI CPU and NUMA-memory placement policy helpers."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import time


_CPUSET = re.compile(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*\Z")
_MAX_CPU_INDEX = 1_048_575
_MAX_MEMORY_NODE_INDEX = 65_535


def _canonical_index_set(
    value: object,
    *,
    field: str,
    maximum: int,
) -> tuple[str | None, int]:
    """Validate Docker/Linux cpuset syntax without expanding attacker-sized ranges.

    ``None`` means that placement is intentionally left to the host.  Present sets
    use one canonical spelling: decimal indices have no leading zeroes, ranges are
    increasing, and adjacent/overlapping ranges are merged.  Requiring that spelling
    makes the policy digest identify exactly one kernel cpuset.
    """

    if value is None:
        return None, 0
    if type(value) is not str or _CPUSET.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical cpuset or None")

    previous_end = -2
    cardinality = 0
    for token in value.split(","):
        raw_start, separator, raw_end = token.partition("-")
        if (raw_start != "0" and raw_start.startswith("0")) or (
            raw_end and raw_end != "0" and raw_end.startswith("0")
        ):
            raise ValueError(f"{field} must not contain leading zeroes")
        start = int(raw_start)
        end = int(raw_end) if separator else start
        if end < start:
            raise ValueError(f"{field} ranges must be increasing")
        if separator and end == start:
            raise ValueError(f"{field} singleton ranges must use one index")
        if end > maximum:
            raise ValueError(f"{field} index exceeds its hard bound")
        if start <= previous_end:
            raise ValueError(f"{field} ranges must be sorted and non-overlapping")
        if start == previous_end + 1:
            raise ValueError(f"{field} adjacent ranges must be merged")
        cardinality += end - start + 1
        previous_end = end
    return value, cardinality


def validate_cpuset_pair(
    cpuset_cpus: object,
    cpuset_mems: object,
    *,
    cpu_millis: int,
) -> tuple[str | None, str | None]:
    """Return a canonical paired CPU/memory-node placement policy.

    CPU and memory placement are paired deliberately: accepting just one would make
    an ostensibly isolated NUMA policy silently allocate or execute on another node.
    The CFS quota may be lower than the selected CPU count, but not higher than the
    maximum execution capacity of that set.
    """

    cpus, cpu_count = _canonical_index_set(
        cpuset_cpus,
        field="cpuset_cpus",
        maximum=_MAX_CPU_INDEX,
    )
    mems, _ = _canonical_index_set(
        cpuset_mems,
        field="cpuset_mems",
        maximum=_MAX_MEMORY_NODE_INDEX,
    )
    if (cpus is None) != (mems is None):
        raise ValueError("cpuset_cpus and cpuset_mems must be specified together")
    if cpus is not None and cpu_millis > cpu_count * 1_000:
        raise ValueError("cpu_millis exceeds the selected cpuset_cpus capacity")
    return cpus, mems


def canonical_cpu_pins(value: object) -> tuple[tuple[str, tuple[int, ...]], ...] | None:
    """Return sealed per-GPU pins sorted by GPU: each row is (scheduler vCPU, pool vCPUs...).

    The B300 VM's vCPUs differ 2x in single-thread speed and unpinned engines landed on different
    ones each boot: whole-boot copy-vs-copy offsets up to 6.7% (2026-09-30).
    """
    if value is None:
        return None
    rows = value.items() if isinstance(value, dict) else value
    pins = tuple(sorted((str(gpu), tuple(cpus)) for gpu, cpus in rows))
    if (not pins or len({gpu for gpu, _ in pins}) != len(pins)
            or len({cpus[0] for _, cpus in pins if cpus}) != len(pins)
            or any(not gpu.isdigit() or len(cpus) < 2 or len(set(cpus)) != len(cpus)
                   or any(type(c) is not int or not 0 <= c <= _MAX_CPU_INDEX for c in cpus)
                   for gpu, cpus in pins)):
        raise ValueError("cpu_pins is malformed")
    return pins


def canonical_cpu_roles(value: object) -> tuple[tuple[str, tuple[int, int]], ...] | None:
    """Return sealed per-engine pins keyed by the engine's first GPU: (DP controller vCPU, frontend vCPU).

    An engine's DP controller busy-polls a whole vCPU and its frontend parses every request. In the shared pool on
    Verda's half-speed vCPUs they ran at 64-100% of a slow core, on the path every first token takes (2026-11-02).
    """
    if value is None:
        return None
    rows = value.items() if isinstance(value, dict) else value
    roles = tuple(sorted((str(gpu), tuple(cpus)) for gpu, cpus in rows))
    flat = [c for _, cpus in roles for c in cpus]
    if (not roles or len({gpu for gpu, _ in roles}) != len(roles) or len(set(flat)) != len(flat)
            or any(not gpu.isdigit() or len(cpus) != 2
                   or any(type(c) is not int or not 0 <= c <= _MAX_CPU_INDEX for c in cpus)
                   for gpu, cpus in roles)):
        raise ValueError("cpu_roles is malformed")
    return roles


def lane_cpu_pin_plan(pins: tuple[tuple[str, tuple[int, ...]], ...], gpu_ids: tuple[str, ...],
                      roles: tuple[tuple[str, tuple[int, int]], ...] | None = None) -> str:
    """The engine worker's plan for one lane: pinned scheduler vCPUs in device order, the pool, and each engine's
    (DP controller, frontend) vCPUs in device order of its first GPU.

    A lane GPU without a pin shortens the scheduler list, which the worker rejects at launch.
    """
    table = dict(pins)
    plan = {"schedulers": [table[gpu][0] for gpu in gpu_ids if gpu in table],
            "pool": sorted({c for gpu in gpu_ids for c in table.get(gpu, ())[1:]})}
    if roles is not None:
        engines = dict(roles)
        plan["engines"] = [list(engines[gpu]) for gpu in gpu_ids if gpu in engines]
    return json.dumps(plan, separators=(",", ":"))


_SPIN_S = 0.05
# Verda 2026-11-02: fast vCPUs 0.99-1.00x, slow tier 0.49x; an SMT pair co-ran at 0.74x, separate cores >= 0.956x.
_FAST = 0.75
_SHARED = 0.85


def _spin(index: int, cpus: list[int], barrier, results) -> None:
    rates = []
    for cpu in cpus:
        os.sched_setaffinity(0, {cpu})
        barrier.wait()
        count, end = 0, time.perf_counter() + _SPIN_S
        while time.perf_counter() < end:
            count += 1
        rates.append(count / _SPIN_S)
    results.put((index, rates))


def _run(*lanes: list[int]) -> list[list[float]]:
    """Each lane's single-thread rate on each of its vCPUs; lanes step through their lists together."""
    context = multiprocessing.get_context("fork")
    barrier, results = context.Barrier(len(lanes)), context.Queue()
    workers = [context.Process(target=_spin, args=(i, lane, barrier, results), daemon=True)
               for i, lane in enumerate(lanes)]
    for worker in workers:
        worker.start()
    rates = dict(results.get(timeout=60 + 4 * _SPIN_S * len(lanes[0])) for _ in workers)
    for worker in workers:
        worker.join()
    return [rates[i] for i in range(len(lanes))]


def verify_cpu_plan(exclusive: dict[str, list[int]], shared: dict[str, list[int]], *, fast: set[str]) -> dict:
    """Time the pin table's vCPUs and refuse one that puts an engine thread on a slow or shared core.

    A guest can hide its SMT siblings and its clock tiers: Verda's 8x B300 VM listed 240 one-thread sockets while
    vCPU k and k+120 shared a core and 16 of its 120 cores ran 2x the rest, so the sibling-free pin rule silently put
    each prefill scheduler on a decode scheduler's core (window sd 3.6% vs 0.6-0.9% on the Lium host, 2026-11-01).
    lscpu and sysfs are not evidence here; timing is. Every vCPU of an ``exclusive`` group (schedulers, engine roles)
    must have its core to itself among the named vCPUs; a ``shared`` group (client, pool) may share cores only within
    itself. Groups named in ``fast`` must run at 0.9x the fastest vCPU this process may use.
    """
    host = sorted(os.sched_getaffinity(0))
    (solo,) = _run(host)
    rate = dict(zip(host, solo))
    top = max(solo)
    groups = {**exclusive, **shared}
    named = sorted({cpu for cpus in groups.values() for cpu in cpus})
    if set(named) - set(host):
        raise RuntimeError(f"pin table names vCPUs this host does not offer: {sorted(set(named) - set(host))}")
    owner = {}
    for name, cpus in groups.items():
        for cpu in cpus:
            if owner.setdefault(cpu, name) != name or len(set(cpus)) != len(cpus):
                raise RuntimeError(f"vCPU {cpu} is named twice in the pin table")
    problems = [f"{name} vCPU {cpu} runs at {rate[cpu] / top:.2f}x the fastest vCPU"
                for name in fast for cpu in groups[name] if rate[cpu] < _FAST * top]
    pairs = [(a, b) for i, a in enumerate(named) for b in named[i + 1:]
             if owner[a] in exclusive or owner[b] in exclusive or owner[a] != owner[b]]
    together = {}
    if pairs:
        first, second = _run([a for a, _ in pairs], [b for _, b in pairs])
        for (a, b), ra, rb in zip(pairs, first, second):
            together[(a, b)] = min(ra / rate[a], rb / rate[b])
            if together[(a, b)] < _SHARED:
                problems.append(f"{owner[a]} vCPU {a} and {owner[b]} vCPU {b} share a core "
                                f"({together[(a, b)]:.2f}x when both run)")
    if problems:
        raise RuntimeError("pin table fails its measured CPU check: " + "; ".join(problems))
    return {"fastest_per_s": round(top), "named": {str(cpu): round(rate[cpu] / top, 3) for cpu in named},
            "worst_pair": round(min(together.values(), default=1.0), 3)}


__all__ = ["canonical_cpu_pins", "canonical_cpu_roles", "lane_cpu_pin_plan", "validate_cpuset_pair",
           "verify_cpu_plan"]
