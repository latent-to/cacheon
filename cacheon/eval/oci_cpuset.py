"""Canonical OCI CPU and NUMA-memory placement policy helpers."""

from __future__ import annotations

import json
import re


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


def lane_cpu_pin_plan(pins: tuple[tuple[str, tuple[int, ...]], ...], gpu_ids: tuple[str, ...]) -> str:
    """The engine worker's plan for one lane: pinned scheduler vCPUs in device order and the pool.

    A lane GPU without a pin shortens the scheduler list, which the worker rejects at launch.
    """
    table = dict(pins)
    return json.dumps({"schedulers": [table[gpu][0] for gpu in gpu_ids if gpu in table],
                       "pool": sorted({c for gpu in gpu_ids for c in table.get(gpu, ())[1:]})},
                      separators=(",", ":"))


__all__ = ["canonical_cpu_pins", "lane_cpu_pin_plan", "validate_cpuset_pair"]
