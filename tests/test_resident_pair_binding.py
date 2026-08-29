from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from cacheon.eval.resident_pair_binding import (
    ResidentPairBindingError,
    ResidentPairLaneBinding,
    ResidentPairRuntimeBinding,
)
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverError,
    ResidentPairCrossoverHold,
    run_resident_pair_crossover,
)
from tests.test_resident_pair_crossover import _setup


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _lane(lane_id: str) -> ResidentPairLaneBinding:
    return ResidentPairLaneBinding(
        lane_id,
        lane_id.lower() * 32,
        _h(f"stock-launch-{lane_id}"),
        _h(f"lane-authority-{lane_id}"),
        _h(f"allocation-{lane_id}"),
        _h(f"executor-namespace-{lane_id}"),
    )


def _binding() -> ResidentPairRuntimeBinding:
    return ResidentPairRuntimeBinding(_h("service-epoch"), (_lane("A"), _lane("B")))


@pytest.fixture
def resident_pairs():
    pairs = []
    yield pairs
    for pair in pairs:
        pair.close()


def _replace_lane(
    binding: ResidentPairRuntimeBinding, lane_id: str = "A", **changes: str
) -> ResidentPairRuntimeBinding:
    changed = replace(binding.lookup(lane_id), **changes)
    lanes = tuple(
        changed if row.lane_id == lane_id else row for row in binding.lanes
    )
    return replace(binding, lanes=lanes)


def test_runtime_binding_is_closed_canonical_and_path_free() -> None:
    binding = _binding()

    assert tuple(row.lane_id for row in binding.lanes) == ("A", "B")
    assert tuple(
        (identity.lane_id, identity.session_id) for identity in binding.identities
    ) == (("A", "a" * 32), ("B", "b" * 32))
    assert binding.lookup("A") is binding.lanes[0]
    assert binding.lookup("B") is binding.lanes[1]
    assert len(binding.digest) == 64
    assert binding.digest == _binding().digest
    with pytest.raises(ResidentPairBindingError, match="lane A or B"):
        binding.lookup("C")

    text = repr(binding).lower()
    for forbidden in ("/users/", "/root/", "secret", "arnorm", "msa"):
        assert forbidden not in text


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("session_id", "a" * 32),
        ("stock_launch_digest", _h("stock-launch-A")),
        ("allocation_digest", _h("allocation-A")),
        ("executor_namespace_digest", _h("executor-namespace-A")),
    ),
)
def test_lanes_cannot_share_runtime_authority(field: str, value: str) -> None:
    lane_a, lane_b = _binding().lanes
    lane_b = replace(lane_b, **{field: value})
    with pytest.raises(ResidentPairBindingError, match="share"):
        ResidentPairRuntimeBinding(_h("epoch"), (lane_a, lane_b))


def test_binding_rejects_noncanonical_rows_and_digests() -> None:
    binding = _binding()
    with pytest.raises(ResidentPairBindingError, match="canonical A/B"):
        replace(binding, lanes=tuple(reversed(binding.lanes)))
    with pytest.raises(ResidentPairBindingError, match="lowercase 32-hex"):
        replace(binding.lanes[0], session_id="A" * 32)
    with pytest.raises(ResidentPairBindingError, match="lowercase 64-hex"):
        replace(binding, service_epoch_digest="F" * 64)


def test_actual_stock_launch_is_retained_separately_from_prepared_arm_launch(
    tmp_path, resident_pairs
) -> None:
    plan, pair, clock, *_ = _setup(tmp_path, resident_pairs)
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )

    prepared_launches = {
        plan.crossover_plan.baseline.launch.digest,
        plan.crossover_plan.candidate.launch.digest,
    }
    actual_stock_launches = {
        row.stock_launch_digest for row in plan.pair_binding.lanes
    }
    assert evidence.pair_binding == plan.pair_binding
    assert {row.launch_digest for row in evidence.rates} == prepared_launches
    assert actual_stock_launches.isdisjoint(prepared_launches)
    assert evidence.pair_binding.identities == pair.identities


@pytest.mark.parametrize(
    "authority",
    (
        "service_epoch_digest",
        "stock_launch_digest",
        "allocation_digest",
        "executor_namespace_digest",
        "lane_digest",
        "session_id",
    ),
)
def test_foreign_runtime_authority_fails_independent_regrade(
    tmp_path, resident_pairs, authority
) -> None:
    plan, pair, clock, *_ = _setup(tmp_path, resident_pairs)
    evidence = run_resident_pair_crossover(
        plan, pair=pair, deadline=clock() + 120.0, clock=clock
    )
    if authority == "service_epoch_digest":
        foreign = replace(plan.pair_binding, service_epoch_digest=_h("foreign-epoch"))
    elif authority == "session_id":
        foreign = _replace_lane(plan.pair_binding, session_id="c" * 32)
    else:
        foreign = _replace_lane(
            plan.pair_binding, **{authority: _h(f"foreign-{authority}")}
        )

    assert foreign.digest != plan.pair_binding.digest
    if authority == "lane_digest":
        with pytest.raises(ResidentPairCrossoverError, match="roles differ"):
            replace(plan, pair_binding=foreign)
    else:
        assert replace(plan, pair_binding=foreign).digest != plan.digest
    changed = replace(evidence, pair_binding=foreign)
    with pytest.raises(ResidentPairCrossoverHold, match="another plan"):
        changed.regrade(plan)


def test_foreign_session_fails_before_any_resident_work(tmp_path, resident_pairs) -> None:
    plan, pair, clock, *_ = _setup(tmp_path, resident_pairs)
    foreign = _replace_lane(plan.pair_binding, session_id="c" * 32)
    foreign_plan = replace(plan, pair_binding=foreign)
    before = pair.request_history

    with pytest.raises(ResidentPairCrossoverHold, match="sessions differ"):
        run_resident_pair_crossover(
            foreign_plan, pair=pair, deadline=clock() + 120.0, clock=clock
        )
    assert pair.request_history == before


def test_role_mapping_is_exact_and_both_orientations_execute(
    tmp_path, resident_pairs
) -> None:
    # Borderline candidates keep both stages on the three-leg terminal
    # schedule so each orientation shows its full lane sequence.
    plan_a, pair_a, clock_a, *_ = _setup(
        tmp_path / "orientation-a",
        resident_pairs,
        baseline_pair_lane="A",
        candidate=((1.0 / 1.006,) * 3,),
    )
    plan_b, pair_b, clock_b, *_ = _setup(
        tmp_path / "orientation-b",
        resident_pairs,
        baseline_pair_lane="B",
        candidate=((1.0 / 1.006,) * 3,),
    )
    plan_a = replace(plan_a, candidate_bundle_digest=_h("candidate-one"))
    plan_b = replace(plan_b, candidate_bundle_digest=_h("candidate-two"))

    evidence_a = run_resident_pair_crossover(
        plan_a, pair=pair_a, deadline=clock_a() + 120.0, clock=clock_a
    )
    evidence_b = run_resident_pair_crossover(
        plan_b, pair=pair_b, deadline=clock_b() + 120.0, clock=clock_b
    )
    assert tuple(row.lane_id for row in evidence_a.request_slices) == ("A", "B", "A")
    assert tuple(row.lane_id for row in evidence_b.request_slices) == ("B", "A", "B")
    assert plan_a.candidate_bundle_digest != plan_b.candidate_bundle_digest
    assert plan_a.pair_binding.digest != plan_b.pair_binding.digest

    with pytest.raises(ResidentPairCrossoverError, match="roles differ"):
        replace(
            plan_a,
            baseline_pair_lane="B",
            candidate_pair_lane="A",
        )
