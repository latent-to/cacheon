from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import cacheon.eval.registered_resident_count_quality as registered_quality
from cacheon.eval.count_quality import CountQualityPolicy
from cacheon.eval.registered_resident_count_quality import (
    RegisteredResidentCountQualityAuthority,
    RegisteredResidentCountQualityHold,
    RegisteredResidentCountQualityResult,
    evaluate_registered_resident_count_quality,
)
from cacheon.eval.resident_count_quality import (
    ResidentCountPromptObservation,
    ResidentCountQualityObservation,
    ResidentCountQualityStockAuthority,
    publish_resident_count_observation,
    seal_resident_count_stock_authority,
)
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountQualityExecutionPlan,
    ResidentCountQualityExecutionResult,
    execute_candidate_count_quality,
)
from cacheon.target_catalog import (
    CorrectnessContractRef,
    TargetCatalog,
    TargetContractRef,
    TargetKind,
    TargetSpec,
)
from tests.test_resident_count_quality_execution import _fixture, _h


SINGLETON_TARGET = "slot.alpha"
ATOMIC_TARGET = "atomic.alpha-beta.v1"


def _contract(target_id: str, marker: str) -> TargetContractRef:
    return TargetContractRef(
        schema_version=1,
        slot_id=target_id,
        kind="op",
        entry="run",
        prepare=None,
        graph_dynamic_inputs=("x",),
        input_abi_id=f"{target_id}.input.v1",
        output_abi_id=f"{target_id}.output.v1",
        reference_id=f"{target_id}.reference.{marker}.v1",
        verification_profile_id=f"{target_id}.verify.{marker}.v1",
        binding_family_id=f"{target_id}.binding.v1",
        correctness=CorrectnessContractRef(),
        tolerances=(),
    )


def _catalog(*, marker: str = "base", extra_target: bool = False) -> TargetCatalog:
    left = TargetSpec(
        SINGLETON_TARGET,
        TargetKind.SLOT,
        (SINGLETON_TARGET,),
        contract_ref=_contract(SINGLETON_TARGET, marker),
    )
    right_id = "slot.beta"
    right = TargetSpec(
        right_id,
        TargetKind.SLOT,
        (right_id,),
        contract_ref=_contract(right_id, marker),
    )
    atomic = TargetSpec(
        ATOMIC_TARGET,
        TargetKind.ATOMIC,
        (SINGLETON_TARGET, right_id),
        displaces=frozenset({SINGLETON_TARGET, right_id}),
        atomic_semantics_id=f"atomic.alpha-beta.{marker}.v1",
    )
    rows = [left, right, atomic]
    if extra_target:
        extra_id = "slot.gamma"
        rows.append(
            TargetSpec(
                extra_id,
                TargetKind.SLOT,
                (extra_id,),
                contract_ref=_contract(extra_id, marker),
            )
        )
    return TargetCatalog(rows)


@dataclass(frozen=True)
class _Product:
    root: Path
    target_id: str
    plan: ResidentCountQualityExecutionPlan
    execution: ResidentCountQualityExecutionResult
    stock: ResidentCountQualityStockAuthority
    judge: object
    authority: RegisteredResidentCountQualityAuthority
    factory_calls: tuple[int, int]


def _product(
    tmp_path: Path,
    catalog: TargetCatalog,
    target_id: str,
    profile: str,
    *,
    candidate_drop: int = 0,
) -> _Product:
    plan, judge, pair, factory_a, factory_b = _fixture(
        4,
        barrier=False,
        profile=profile,
    )
    correct_outputs = tuple(
        factory_a.sessions[0].outputs[prompt] for prompt in plan.selected_prompts
    )
    decoder_rows = judge._decoder.__self__
    assert type(decoder_rows) is dict
    for index, prompt in enumerate(plan.selected_prompts[:candidate_drop]):
        wrong = tuple(900_000 + index * 16 + token for token in range(8))
        decoder_rows[wrong] = "no numeric answer"
        factory_a.sessions[0].outputs[prompt] = wrong
        factory_b.sessions[0].outputs[prompt] = wrong
    try:
        execution = execute_candidate_count_quality(
            plan,
            pair=pair,
            judge=judge,
            deadline=10**10,
        )
    finally:
        pair.close()

    stock_rows = []
    for ordinal, (occurrence, output_ids) in enumerate(
        zip(plan.selected_occurrences, correct_outputs, strict=True)
    ):
        stock_rows.append(
            ResidentCountPromptObservation(
                ordinal,
                occurrence.prompt_digest,
                occurrence.task_digests,
                output_ids,
                judge(
                    prompt_digest=occurrence.prompt_digest,
                    output_ids=output_ids,
                    task_digests=occurrence.task_digests,
                ),
            )
        )
    stock_observation = ResidentCountQualityObservation(
        "stock",
        plan.envelope,
        _h(f"{profile}-fixed-stock-execution"),
        tuple(stock_rows),
    )
    root = tmp_path / profile / "evidence"
    root.parent.mkdir(mode=0o700)
    policy = CountQualityPolicy(2)
    stock = seal_resident_count_stock_authority(
        root,
        publish_resident_count_observation(root, stock_observation),
        policy=policy,
    )
    authority = RegisteredResidentCountQualityAuthority.register(
        catalog,
        target_id,
        plan=plan,
        stock_authority=stock,
        judge=judge,
        policy=policy,
    )
    return _Product(
        root,
        target_id,
        plan,
        execution,
        stock,
        judge,
        authority,
        (factory_a.calls, factory_b.calls),
    )


def _evaluate(
    catalog: TargetCatalog,
    product: _Product,
    **changes: object,
) -> RegisteredResidentCountQualityResult:
    values = {
        "evidence_root": product.root,
        "catalog": catalog,
        "target_id": product.target_id,
        "authority": product.authority,
        "plan": product.plan,
        "execution": product.execution,
        "stock_authority": product.stock,
        "judge": product.judge,
    }
    values.update(changes)
    return evaluate_registered_resident_count_quality(**values)  # type: ignore[arg-type]


def _assert_hold(callable_) -> RegisteredResidentCountQualityHold:
    with pytest.raises(RegisteredResidentCountQualityHold) as raised:
        callable_()
    assert raised.value.decision == "HOLD"
    return raised.value


def test_two_structurally_distinct_profiles_share_one_path_and_reopen_is_pure(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    products = (
        _product(tmp_path, catalog, SINGLETON_TARGET, "registered-singleton"),
        _product(tmp_path, catalog, ATOMIC_TARGET, "registered-atomic"),
    )
    outcomes = []
    for product in products:
        before = product.factory_calls
        first = _evaluate(catalog, product)
        restarted = _evaluate(catalog, product)
        assert first == restarted
        assert first.digest == restarted.digest
        assert first.decision == "PASS"
        assert first.target_id == product.target_id
        assert first.catalog_digest == catalog.digest
        assert first.target_spec_digest == catalog.target_spec_digest(
            product.target_id
        )
        assert first.profile_digest == product.authority.digest
        assert first.execution_plan_digest == product.plan.digest
        assert first.execution_envelope_digest == product.plan.envelope.digest
        assert first.pair_binding_digest == product.plan.pair_binding.digest
        assert first.candidate_bundle_digest == product.plan.candidate_bundle_digest
        assert first.raw_execution_evidence_digest == product.execution.evidence.digest
        assert first.fixed_stock_authority_digest == product.stock.digest
        assert first.candidate_observation_digest == (
            product.execution.observation.digest
        )
        assert first.count_quality_result.evidence.stock_correct == 4
        assert first.count_quality_result.evidence.candidate_correct == 4
        assert product.factory_calls == before == (1, 1)
        outcomes.append(first)

        authority_json = json.dumps(product.authority.to_dict(), sort_keys=True)
        result_json = json.dumps(first.to_dict(), sort_keys=True)
        assert str(tmp_path) not in authority_json + result_json
        assert "candidate_bundle_digest" not in authority_json
        assert "pair_binding_digest" not in authority_json
        assert "stock_correct" not in authority_json

    assert catalog.require(SINGLETON_TARGET).kind is TargetKind.SLOT
    assert catalog.require(ATOMIC_TARGET).kind is TargetKind.ATOMIC
    assert outcomes[0].profile_digest != outcomes[1].profile_digest
    assert outcomes[0].pair_binding_digest != outcomes[1].pair_binding_digest


def test_exact_candidate_regression_returns_fail_not_hold(tmp_path: Path) -> None:
    catalog = _catalog()
    product = _product(
        tmp_path,
        catalog,
        SINGLETON_TARGET,
        "registered-candidate-fail",
        candidate_drop=2,
    )

    result = _evaluate(catalog, product)

    assert result.decision == "FAIL"
    assert result.count_quality_result.evidence.stock_correct == 4
    assert result.count_quality_result.evidence.candidate_correct == 2
    assert result.count_quality_result.verdict.observed_drop == 2


def test_catalog_spec_and_cross_target_profiles_fail_closed(tmp_path: Path) -> None:
    catalog = _catalog()
    product = _product(
        tmp_path,
        catalog,
        SINGLETON_TARGET,
        "registered-authority-drift",
    )

    catalog_drift = _catalog(extra_target=True)
    assert catalog_drift.digest != catalog.digest
    assert catalog_drift.target_spec_digest(SINGLETON_TARGET) == (
        catalog.target_spec_digest(SINGLETON_TARGET)
    )
    _assert_hold(lambda: _evaluate(catalog_drift, product))

    spec_drift = _catalog(marker="changed")
    stale_spec_authority = replace(
        product.authority,
        catalog_digest=spec_drift.digest,
    )
    assert spec_drift.target_spec_digest(SINGLETON_TARGET) != (
        product.authority.target_spec_digest
    )
    _assert_hold(
        lambda: _evaluate(
            spec_drift,
            product,
            authority=stale_spec_authority,
        )
    )

    _assert_hold(
        lambda: _evaluate(
            catalog,
            product,
            target_id=ATOMIC_TARGET,
        )
    )


def test_stock_plan_pair_judge_and_policy_substitutions_hold(tmp_path: Path) -> None:
    catalog = _catalog()
    first = _product(
        tmp_path,
        catalog,
        SINGLETON_TARGET,
        "registered-substitution-one",
    )
    second = _product(
        tmp_path,
        catalog,
        ATOMIC_TARGET,
        "registered-substitution-two",
    )

    _assert_hold(lambda: _evaluate(catalog, first, stock_authority=second.stock))
    _assert_hold(
        lambda: _evaluate(
            catalog,
            first,
            plan=second.plan,
            execution=second.execution,
        )
    )
    _assert_hold(lambda: _evaluate(catalog, first, judge=second.judge))

    foreign_pair = replace(
        first.plan.pair_binding,
        service_epoch_digest=_h("substituted-service-epoch"),
    )
    foreign_plan = replace(first.plan, pair_binding=foreign_pair)
    foreign_profile = replace(
        first.authority,
        execution_plan_digest=foreign_plan.digest,
    )
    _assert_hold(
        lambda: _evaluate(
            catalog,
            first,
            authority=foreign_profile,
            plan=foreign_plan,
        )
    )

    drifted_stock = replace(first.stock, policy=CountQualityPolicy(3))
    drifted_profile = replace(
        first.authority,
        fixed_stock_authority_digest=drifted_stock.digest,
    )
    _assert_hold(
        lambda: _evaluate(
            catalog,
            first,
            authority=drifted_profile,
            stock_authority=drifted_stock,
        )
    )


def test_candidate_observation_substitution_is_rejected_after_raw_regrade(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    product = _product(
        tmp_path,
        catalog,
        SINGLETON_TARGET,
        "registered-observation-substitution",
    )
    original = product.execution.observation.prompts[0]
    wrong_ids = tuple(990_000 + token for token in range(8))
    decoder_rows = product.judge._decoder.__self__
    decoder_rows[wrong_ids] = "no numeric answer"
    wrong_receipt = product.judge(
        prompt_digest=original.prompt_digest,
        output_ids=wrong_ids,
        task_digests=original.task_digests,
    )
    changed_row = replace(
        original,
        output_ids=wrong_ids,
        judge_receipt=wrong_receipt,
    )
    changed_observation = replace(
        product.execution.observation,
        prompts=(changed_row, *product.execution.observation.prompts[1:]),
    )
    substituted = ResidentCountQualityExecutionResult(
        product.execution.evidence,
        changed_observation,
    )

    hold = _assert_hold(
        lambda: _evaluate(catalog, product, execution=substituted)
    )
    assert "independent raw regrade" in str(hold)


def test_missing_stock_is_hold_and_gate_has_no_stock_or_model_executor(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    product = _product(
        tmp_path,
        catalog,
        SINGLETON_TARGET,
        "registered-missing-stock",
    )
    missing_root = tmp_path / "missing-evidence-root"

    _assert_hold(
        lambda: _evaluate(catalog, product, evidence_root=missing_root)
    )

    public_names = tuple(registered_quality.__all__)
    assert not any(
        "stock" in name.lower() and "execute" in name.lower()
        for name in public_names
    )
    parameters = inspect.signature(
        evaluate_registered_resident_count_quality
    ).parameters
    assert not {"pair", "model", "executor", "stock_executor"} & set(parameters)
    source = inspect.getsource(registered_quality)
    assert "ResidentEvaluationPair" not in source
    assert "execute_candidate_count_quality" not in source
    assert "execute_stock" not in source
    assert SINGLETON_TARGET not in source
    assert ATOMIC_TARGET not in source
