"""Commissioning surface for qualification in the one existing pod service.

Full composition needs the commissioned B300 host (sealed deployment root,
eight-GPU lane pair, OCI runtime); these tests pin the CPU-checkable gates:
capability typing and identity binding, the tracked deadline policy, lane
disjointness, and sealed calibration package handling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval import b300_qualification_commission as commission
from cacheon.eval.b300_qualification_deployment import B300RegisteredProfileAuthority
from cacheon.eval.b300_registered_qualification import REGISTERED_B300_TARGET_IDS
from cacheon.eval.b300_sealed_qualification_commission import (
    QUALIFICATION_DEADLINE_MAXIMUM_SECONDS,
)
from cacheon.eval.device_state import GPUConfiguration
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.target_catalog import default_target_catalog


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


class _Judge:
    def __init__(self) -> None:
        self.binding = HiddenJudgeBinding(
            _h("hidden-corpus"), _h("hidden-judge"), _h("hidden-policy")
        )

    def __call__(self, **_kwargs):
        raise AssertionError("capability checks must not execute the hidden judge")


class _Resolver:
    def resolve_proposal(self, *_args, **_kwargs):
        raise AssertionError("capability checks must not resolve sources")

    def resolve_integrated(self, *_args, **_kwargs):
        raise AssertionError("capability checks must not resolve sources")


class _DeferredJudge:
    def __init__(self, binding: HiddenJudgeBinding, tokenizer_digest: str) -> None:
        self.binding = binding
        self.tokenizer_digest = tokenizer_digest
        self.calls: list[dict[str, object]] = []

    def bind_prompt_plan(self, **kwargs: object) -> _Judge:
        self.calls.append(dict(kwargs))
        result = _Judge()
        result.binding = self.binding
        return result


def _capabilities(**overrides: object) -> commission.B300QualificationCapabilities:
    values: dict[str, object] = {
        "secret_loader": lambda _reference: b"s" * 32,
        "entropy_provider": lambda *_args: None,
        "hidden_judge": _Judge(),
        "source_resolver": _Resolver(),
        "source_resolver_digest": _h("source-resolver"),
        "graph_facts_builder": lambda *_args: None,
        "graph_facts_builder_digest": _h("graph-facts"),
    }
    values.update(overrides)
    return commission.B300QualificationCapabilities(**values)


def test_capabilities_seal_exact_callables_and_identities() -> None:
    capabilities = _capabilities()
    assert capabilities.source_resolver_digest == _h("source-resolver")

    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(secret_loader=object())
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(source_resolver=object())
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(hidden_judge=lambda **_kwargs: None)
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(graph_facts_builder_digest="not-a-digest")
    with pytest.raises(commission.B300QualificationCommissionError):
        _capabilities(source_resolver_digest=_h("upper").upper())


def test_deferred_hidden_judge_binds_only_the_exact_composed_plan() -> None:
    binding = HiddenJudgeBinding(
        _h("hidden-corpus"), _h("hidden-judge"), _h("hidden-policy")
    )
    tokenizer = _h("tokenizer")
    deferred = _DeferredJudge(binding, tokenizer)
    capabilities = _capabilities(hidden_judge=deferred)
    assert capabilities.hidden_judge is deferred

    bound = commission._bind_hidden_judge(
        deferred,
        binding=binding,
        tokenizer_digest=tokenizer,
        prompt_batches=(("prompt",),),
        workload_digest=_h("workload"),
        hidden_tasks_per_prompt=1,
    )
    assert callable(bound)
    assert deferred.calls == [
        {
            "hidden_tasks_per_prompt": 1,
            "prompt_batches": (("prompt",),),
            "workload_digest": _h("workload"),
        }
    ]

    with pytest.raises(
        commission.B300QualificationCommissionError,
        match="tokenizer differs",
    ):
        commission._bind_hidden_judge(
            deferred,
            binding=binding,
            tokenizer_digest=_h("other-tokenizer"),
            prompt_batches=(("prompt",),),
            workload_digest=_h("workload"),
            hidden_tasks_per_prompt=1,
        )


def test_tracked_deadline_is_lease_bounded_monotonic() -> None:
    deadline = commission._tracked_deadline_provider(clock=lambda: 1000.0)
    assert deadline(object()) == 1000.0 + QUALIFICATION_DEADLINE_MAXIMUM_SECONDS
    # The policy never depends on the cohort it is asked about.
    assert deadline(None) == deadline(object())


def test_commission_rejects_an_eleven_row_factory_registry_before_runtime() -> None:
    catalog = default_target_catalog()
    profiles = tuple(
        B300RegisteredProfileAuthority(
            target_id,
            catalog.target_spec_digest(target_id),
            _h(f"resolver:{target_id}"),
            lambda _candidate, _prepared: object(),
        )
        for target_id in REGISTERED_B300_TARGET_IDS
    )

    assert tuple(commission._require_complete_factory_profiles(profiles)) == (
        REGISTERED_B300_TARGET_IDS
    )
    with pytest.raises(
        commission.B300QualificationCommissionError,
        match="full catalog",
    ):
        commission._require_complete_factory_profiles(profiles[:-1])


def _gpu(index: int) -> GPUConfiguration:
    return GPUConfiguration(
        physical_id=index,
        uuid=f"GPU-00000000-{index:04x}-0000-0000-{index:012x}",
        pci_bus_id=f"00000000:{index + 1:02x}:00.0",
        name="NVIDIA B300 SXM6 AC",
        memory_total_mib=288_000,
        driver_version="600.10.01",
        power_limit_mw=1_000_000,
        compute_mode="Default",
        persistence_mode="Enabled",
        application_graphics_clock_mhz=None,
        application_memory_clock_mhz=None,
        max_graphics_clock_mhz=2_500,
        max_memory_clock_mhz=5_000,
    )


def test_lane_gpus_require_one_disjoint_tp4_complement() -> None:
    eight = tuple(_gpu(index) for index in range(8))
    screen = eight[:4]
    inputs = SimpleNamespace(gpus=screen, qualification_gpus=eight)
    baseline, candidate = commission._lane_gpus(inputs)
    assert baseline == screen
    assert tuple(gpu.physical_id for gpu in candidate) == (4, 5, 6, 7)

    overlapping = SimpleNamespace(gpus=screen, qualification_gpus=eight[:7])
    with pytest.raises(commission.B300QualificationCommissionError):
        commission._lane_gpus(overlapping)


def _calibration_inputs(tmp_path: Path, payload: object) -> SimpleNamespace:
    path = tmp_path / "calibration-package.json"
    raw = json.dumps(payload).encode("utf-8")
    path.write_bytes(raw)
    return SimpleNamespace(
        authority_refs={
            "calibration_package": {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        }
    )


def test_sealed_calibration_rejects_reference_drift(tmp_path: Path) -> None:
    inputs = _calibration_inputs(
        tmp_path,
        {
            "schema": commission.CALIBRATION_PACKAGE_SCHEMA,
            "threshold_policy": {},
            "observations": [],
        },
    )
    inputs.authority_refs["calibration_package"]["sha256"] = _h("other-bytes")
    with pytest.raises(commission.B300QualificationCommissionError) as captured:
        commission._sealed_calibration(inputs)
    assert "deployment reference" in str(captured.value)


def test_sealed_calibration_rejects_open_or_foreign_packages(
    tmp_path: Path,
) -> None:
    for payload in (
        ["not", "a", "package"],
        {"schema": "cacheon-private-b300-calibration-package-v0"},
        {
            "schema": commission.CALIBRATION_PACKAGE_SCHEMA,
            "threshold_policy": {},
            "observations": [],
            "operator_note": "no",
        },
        {
            "schema": commission.CALIBRATION_PACKAGE_SCHEMA,
            "threshold_policy": {},
            "observations": {},
        },
    ):
        inputs = _calibration_inputs(tmp_path, payload)
        with pytest.raises(commission.B300QualificationCommissionError):
            commission._sealed_calibration(inputs)
        (tmp_path / "calibration-package.json").unlink()


def test_sealed_calibration_rejects_invalid_frozen_authorities(
    tmp_path: Path,
) -> None:
    inputs = _calibration_inputs(
        tmp_path,
        {
            "schema": commission.CALIBRATION_PACKAGE_SCHEMA,
            "threshold_policy": {"not": "a policy"},
            "observations": [],
        },
    )
    with pytest.raises(commission.B300QualificationCommissionError) as captured:
        commission._sealed_calibration(inputs)
    assert "invalid" in str(captured.value)


def test_compose_requires_a_sealed_commission_block() -> None:
    inputs = SimpleNamespace(qualification_commission=None)
    with pytest.raises(commission.B300QualificationCommissionError) as captured:
        commission.compose_commissioned_qualification(
            inputs, object(), object(), _capabilities()
        )
    assert "declares no qualification commission" in str(captured.value)

    with pytest.raises(commission.B300QualificationCommissionError):
        commission.compose_commissioned_qualification(
            inputs, object(), object(), object()
        )
