"""Stock imports retain old lane identities without admitting unrelated reads."""

from dataclasses import replace

import pytest

from cacheon.eval.goodput_runtime import GoodputPolicy, GoodputReadSet
from cacheon.eval.service_capacity import LoadRead, ServiceContract, TurnRecord


def _reads(lane):
    def read(arm, index, seconds, identity):
        start = (index + 1) * 1000 * 10**9
        cold = TurnRecord("session", "main", 0, start, start, start + 10**8,
                          start + 10**9, 100, 2, "ok")
        start += 2 * 10**9
        warm = TurnRecord("session", "main", 1, start, start, start + 10**8,
                          start + round(seconds * 10**9), 100, 2, "ok")
        return LoadRead(arm, 1, identity, 24, (cold, warm))

    return GoodputReadSet(
        tuple(read("incumbent", i, 100, lane) for i in range(4)),
        tuple(read("candidate", i + 4, 95, "current") for i in range(2)),
        (("session", 2, 0),), 1, boots=2, reference_boots=4,
        reference_lane=lane if lane != "current" else "",
    )


def test_import_keeps_original_reads_and_regrades_identically():
    policy = GoodputPolicy(ServiceContract(1.0, 0.001, 0.8), 1.0, 0.0003,
                           0.0, 0.0, 0.01, 0.0001)
    imported, same = _reads("original"), _reads("current")
    assert imported.grade(policy) == same.grade(policy)
    assert {row.lane for row in imported.incumbent} == {"original"}
    assert GoodputReadSet.from_dict(imported.to_dict()) == imported
    legacy = same.to_dict()
    assert "reference_lane" not in legacy.get("fields", {})
    assert GoodputReadSet.from_dict(legacy) == same


def test_import_rejects_another_reference_or_candidate_lane():
    imported = _reads("original")
    with pytest.raises(ValueError, match="whole lifetimes"):
        replace(imported, reference_lane="unrelated")
    with pytest.raises(ValueError, match="whole lifetimes"):
        replace(imported, incumbent=(replace(imported.incumbent[0], lane="unrelated"),
                                     *imported.incumbent[1:]))
    with pytest.raises(ValueError, match="whole lifetimes"):
        replace(imported, candidate=(replace(imported.candidate[0], lane="unrelated"),
                                     imported.candidate[1]))
    bracket = replace(imported.incumbent[0], lane="original")
    with pytest.raises(ValueError, match="whole lifetimes"):
        replace(imported, incumbent=(*imported.incumbent, bracket))


def test_steady_span_policy_round_trips_and_grades_the_same_complete_work():
    policy = GoodputPolicy(ServiceContract(1.0, 0.001, 0.8), 1.0, 0.0003,
                           0.0, 0.0, 0.01, 0.0001, steady_span=0.5)
    assert GoodputPolicy.from_dict(policy.to_dict()) == policy
    assert "steady_span" not in replace(policy, steady_span=0.0).to_dict()
    reads = _reads("original")
    assert reads.grade(policy) == reads.grade(replace(policy, steady_span=0.0))


def test_single_stock_pair_regrades_with_its_own_uncertainty_and_control():
    from cacheon.eval.qualification_trajectories import _stock_pair

    policy = GoodputPolicy(ServiceContract(1.0, 0.001, 0.8), 1.0, 0.0003, 0.0, 0.0, 0.01, 0.0001)
    four = _reads("current")
    two = replace(four, incumbent=four.incumbent[:2], reference_boots=2)
    assert GoodputReadSet.from_dict(two.to_dict()) == two
    assert two.grade(policy).settled_speedup == four.grade(policy).settled_speedup
    assert two.grade(policy).verdict.noise > four.grade(policy).verdict.noise
    assert _stock_pair(four.incumbent, 4) == (four.incumbent[0], four.incumbent[2])
    assert _stock_pair(two.incumbent, 2) == two.incumbent
