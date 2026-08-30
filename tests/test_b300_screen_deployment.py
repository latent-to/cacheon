"""CPU-only commission/replay tests for the fixed B300 screen deployment."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from dataclasses import replace
from pathlib import Path

import pytest

import cacheon.eval.b300_screen_deployment as deployment
from cacheon.arena_service import ArenaCandidateBinding, WorkloadCell
from cacheon.bundle_hash import content_hash
from cacheon.chain.publication import publish_worker_bundle
from cacheon.engine_tree import inspect_contribution
from cacheon.eval.b300_arena_provider import B300ResidentScreenLifetime
from cacheon.eval.b300_screen_stages import B300ScreenExecutionPlan
from cacheon.eval.device_state import GPUConfiguration
from cacheon.eval.oci_backend import runtime_identity_from_preflight
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.eval.runtime_preflight import RuntimePreflightReceipt
from cacheon.target_catalog import default_target_catalog
from tests.support.b300 import (
    GLM53_REGISTERED_TARGET_IDS,
    M3_REGISTERED_TARGET_IDS,
    gpu as _gpu,
)
from tests.support.preflight import preflight_receipt


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write(path: Path, value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(raw)
    path.chmod(0o400)
    return hashlib.sha256(raw).hexdigest()


def _preflight() -> RuntimePreflightReceipt:
    return preflight_receipt(
        image=_h("image"),
        platform=_h("platform"),
        worker=_h("worker-distribution"),
        worker_file_count=100,
        worker_total_bytes=100_000,
    )


def _m3_engine_config() -> dict[str, object]:
    return {
        "attention_backend": None,
        "deterministic": False,
        "disable_custom_all_reduce": True,
        "dtype": "bfloat16",
        "engine_kwargs": {
            "chunked_prefill_size": 4096,
            "cuda_graph_backend_prefill": "disabled",
            "disable_radix_cache": True,
            "kv_cache_dtype": "auto",
            "page_size": 128,
            "quantization": "modelopt_fp4",
            "trust_remote_code": True,
        },
        "log_level": "error",
        "mem_fraction_static": 0.70,
        "moe_runner_backend": "flashinfer_cutlass",
        "tp_size": 4,
    }


def _case(tmp_path: Path) -> tuple[dict[str, Path], tuple[GPUConfiguration, ...], dict[str, object]]:
    tmp_path.chmod(0o700)
    model = tmp_path / "model"
    model.mkdir(mode=0o700)
    (model / "config.json").write_text("{}\n")
    (model / "config.json").chmod(0o400)
    preflight = _preflight()
    runtime = runtime_identity_from_preflight(preflight)
    device = tmp_path / "device-execution.json"
    device_sha = _write(
        device,
        {
            "schema": "cacheon-private-device-control-v1",
            "worker": {"preflight": preflight.canonical_payload()},
        },
    )
    prompt = tmp_path / "prompt-authority.json"
    prompt_value = {
        "accepted_token_subsequences": [],
        "registered_targets": list(M3_REGISTERED_TARGET_IDS),
        "hidden_corpus_commitment": _h("hidden-corpus"),
        "hidden_judge_digest": _h("hidden-judge"),
        "hidden_task_policy_digest": _h("hidden-task-policy"),
        "engine_config": _m3_engine_config(),
        "model_profile_key": "MiniMax-M3",
        "prompt_batches": [["one"], ["two"], ["three"]],
        "prompt_seed_scheme": "sealed-prompts-v1",
        "schema": "cacheon-private-prompt-authority-v1",
        "selection_policy_digest": _h("selection-policy"),
        "tokenizer_digest": _h("tokenizer"),
        "workload_cell": {
            "cell_id": "s8",
            "concurrency": 1,
            "input_tokens": 8192,
            "output_tokens": 1024,
            "timed_reads": 2,
        },
    }
    prompt_sha = _write(prompt, prompt_value)
    calibration = tmp_path / "calibration-package.json"
    calibration_sha = _write(calibration, {"schema": "cacheon-calibration-v1"})
    projection = tmp_path / "calibration-projection-receipt.json"
    _write(projection, {"schema": "cacheon-calibration-projection-v1"})
    lane_digest = _h("sealed-lane")
    qualification_builder = _h("qualification-builder")
    authority_value = {
        "arena_id": "minimax-m3-b300-tp4-mainnet",
        "authority_role": "primary",
        "calibration": {
            "evidence_root": str(tmp_path / "calibration-evidence"),
            "package": str(calibration),
            "package_sha256": calibration_sha,
        },
        "device_execution": {"path": str(device), "sha256": device_sha},
        "fmha_cache_seed": {
            "directory_count": 2,
            "file_count": 3,
            "plan_sha256": _h("fmha-plan"),
            "root": str(tmp_path / "fmha-cache"),
            "schema": "cacheon-fmha-cache-v1",
            "sealed_root_owned_read_only": True,
            "total_bytes": 4096,
            "tree_sha256": _h("fmha-tree"),
        },
        "model": {
            "content_digest": _h("model-content"),
            "manifest_digest": _h("model-manifest"),
            "revision_digest": _h("model-revision"),
            "root": str(model),
        },
        "prompt": {"path": str(prompt), "sha256": prompt_sha},
        "qualification_builder_digest": qualification_builder,
        "schema": "cacheon-private-b300-authority-v1",
        "source": {"controller_digest": _h("old-controller")},
        "topology": {
            "architecture": "sm103",
            "gpu_count": 4,
            "lane": ["0", "1", "2", "3"],
            "lane_digest": lane_digest,
            "tensor_parallel_size": 4,
            "topology_class": "nvlink-sxm6",
        },
        "worker": {
            "base_engine_digest": runtime.base_engine_digest,
            "image": preflight.requested_image,
            "local_image_id": preflight.local_image_id,
            "runtime_digest": runtime.runtime_digest,
            "validator_overlay_digest": runtime.validator_overlay_digest,
            "worker_distribution_digest": preflight.worker_distribution_digest,
        },
    }
    authority = tmp_path / "authority-config.json"
    _write(authority, authority_value)
    measurement = tmp_path / "measurement-config.json"
    _write(measurement, authority_value)
    gpus = tuple(_gpu(index) for index in range(8))
    inventory = [
        {
            "index": index,
            "memory_mib": 288_000,
            "name": "NVIDIA B300 SXM6 AC",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            "uuid": gpus[index].uuid,
        }
        for index in range(8)
    ]
    ready_value = {
        "created_at_unix": 1,
        "gpu": {"count": 8, "inventory": inventory},
        "lane": {
            "devices": [0, 1, 2, 3],
            "lane_digest": _h("ready-lane"),
            "tensor_parallel_size": 4,
        },
        "model": {
            "content_digest": _h("model-content"),
            "path": str(model),
            "readonly_inventory_verified": True,
        },
        "receipt_digest": _h("ready-receipt"),
        "runtime": {"path": str(tmp_path / "runtime"), "tree_digest": _h("runtime-tree")},
        "schema": "cacheon-current-pod-commission-v1",
        "source": {"path": str(tmp_path / "source"), "tree_digest": _h("source-tree")},
        "state": "READY_FOR_REGISTRATION",
        "worker_epoch": "1" * 32,
        "worker_image": preflight.requested_image,
    }
    ready = tmp_path / "ready-receipt.json"
    _write(ready, ready_value)
    output = tmp_path / "commissioned"
    output.mkdir(mode=0o700)
    return (
        {
            "authority_config": authority,
            "calibration_package": calibration,
            "calibration_projection_receipt": projection,
            "measurement_config": measurement,
            "output_root": output,
            "prompt_authority": prompt,
            "ready_receipt": ready,
        },
        gpus,
        ready_value,
    )


def test_materialize_and_replay_exact_service_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, gpus, ready = _case(tmp_path)
    result = deployment.materialize_b300_screen_identities(
        **paths,
        gpu_provisioner=lambda selected, *, deadline: gpus,
    )
    output = paths["output_root"]
    manifest = deployment._manifest_from_dict(
        json.loads((output / deployment.MANIFEST_FILE).read_text())
    )
    readiness_row = json.loads((output / deployment.READINESS_FILE).read_text())
    from cacheon.chain.evaluation_coordinator import WorkerReadiness

    readiness = WorkerReadiness(**readiness_row)
    assert result["service_digest"] == manifest.digest
    assert result["worker_readiness_digest"] == readiness.digest
    assert manifest.runtime.gpu_count == 4
    assert manifest.runtime.tensor_parallel_size == 4
    assert manifest.runtime.target_architecture == "sm103"
    assert manifest.runtime.topology_digest == _h("sealed-lane")
    deployment_row = json.loads((output / deployment.DEPLOYMENT_FILE).read_text())
    pair = deployment_row["declared_qualification"]["lane_pair"]
    assert pair["lane_a"]["physical_gpu_ids"] == [0, 1, 2, 3]
    assert pair["lane_b"]["physical_gpu_ids"] == [4, 5, 6, 7]
    assert set(pair["lane_a"]["gpu_uuids"]).isdisjoint(
        pair["lane_b"]["gpu_uuids"]
    )

    monkeypatch.setattr(deployment, "DEFAULT_OUTPUT_ROOT", output)
    registration = {
        "lane_devices": [0, 1, 2, 3],
        "ready_receipt_digest": ready["receipt_digest"],
        "service_identity": manifest.service_id,
        "worker_epoch": ready["worker_epoch"],
        "worker_readiness": readiness.to_dict(),
        "worker_readiness_digest": readiness.digest,
    }
    worker = deployment.build_commissioned_b300_screen_worker(
        registration, ready
    )
    try:
        assert worker.service.manifest == manifest
        assert worker.readiness == readiness
    finally:
        worker.close()

    registration["service_identity"] = manifest.digest
    with pytest.raises(
        deployment.B300ScreenDeploymentError,
        match="registration differs",
    ):
        deployment.build_commissioned_b300_screen_worker(registration, ready)


def test_calibration_authority_does_not_create_service_identity_cycle(
    tmp_path: Path,
) -> None:
    case_a = tmp_path / "a"
    case_b = tmp_path / "b"
    case_a.mkdir()
    case_b.mkdir()
    paths_a, gpus_a, _ready_a = _case(case_a)
    paths_b, gpus_b, _ready_b = _case(case_b)
    calibration_b = paths_b["calibration_package"]
    calibration_b.chmod(0o600)
    calibration_sha = _write(
        calibration_b,
        {"schema": "cacheon-calibration-v1", "generation": 2},
    )
    authority_b = paths_b["authority_config"]
    authority_value = json.loads(authority_b.read_text())
    authority_value["calibration"]["package_sha256"] = calibration_sha
    authority_b.chmod(0o600)
    _write(authority_b, authority_value)

    result_a = deployment.materialize_b300_screen_identities(
        **paths_a,
        gpu_provisioner=lambda selected, *, deadline: gpus_a,
    )
    result_b = deployment.materialize_b300_screen_identities(
        **paths_b,
        gpu_provisioner=lambda selected, *, deadline: gpus_b,
    )

    assert result_a["service_digest"] == result_b["service_digest"]
    deployment_a = json.loads(
        (paths_a["output_root"] / deployment.DEPLOYMENT_FILE).read_text()
    )
    deployment_b = json.loads(
        (paths_b["output_root"] / deployment.DEPLOYMENT_FILE).read_text()
    )
    assert deployment_a["plan_resolver_digest"] == deployment_b["plan_resolver_digest"]
    assert (
        deployment_a["authorities"]["calibration_package"]["sha256"]
        != deployment_b["authorities"]["calibration_package"]["sha256"]
    )


def test_materializer_rejects_mutated_sealed_prompt(tmp_path: Path) -> None:
    paths, gpus, _ready = _case(tmp_path)
    prompt = paths["prompt_authority"]
    prompt.chmod(0o600)
    prompt.write_text(prompt.read_text() + " ")
    prompt.chmod(0o400)
    with pytest.raises(
        deployment.B300ScreenDeploymentError,
        match="explicit sealed authority paths or SHA-256",
    ):
        deployment.materialize_b300_screen_identities(
            **paths,
            gpu_provisioner=lambda selected, *, deadline: gpus,
        )


def test_materializer_refuses_single_tp4_as_declared_qualification_pair(
    tmp_path: Path,
) -> None:
    paths, gpus, _ready = _case(tmp_path)
    with pytest.raises(
        deployment.B300ScreenDeploymentError,
        match="exact commissioned eight-B300 pair",
    ):
        deployment.materialize_b300_screen_identities(
            **paths,
            gpu_provisioner=lambda selected, *, deadline: gpus[:4],
        )


def test_concrete_resolver_materializes_published_bundle_and_binds_tp4_launches(
    tmp_path: Path,
) -> None:
    paths, gpus, _ready = _case(tmp_path)
    inputs = deployment._authority_inputs(
        **paths,
        provisioner=None,
        provisioned_gpus=gpus,
    )
    composition = deployment._compose(inputs)
    try:
        source = tmp_path / "candidate-source"
        shutil.copytree(
            Path(__file__).parents[1]
            / "examples"
            / "miner_moe_fused_experts_reduce_torch",
            source,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for child in sorted(source.rglob("*")):
            child.chmod(0o700 if child.is_dir() else 0o600)
        source.chmod(0o700)
        catalog = default_target_catalog()
        inspected = inspect_contribution(source, catalog=catalog)
        publication = publish_worker_bundle(
            source,
            tmp_path / "candidate-publications",
            content_hash(source),
        )
        target = catalog.require(inspected.target_id)
        candidate = ArenaCandidateBinding(
            QualificationReservation(
                _h("reservation"),
                publication.digest,
                inspected.target_id,
                inspected.selected_delta_digest,
                0,
                "miner",
                100,
                0,
                0,
                target.members,
            ),
            publication,
            1,
        )
        static_result = composition.authorities.screen_handlers[0].run_screen(
            composition.manifest,
            composition.manifest.screens.stages[0],
            candidate,
        )
        assert static_result.grade.value == "fail"
        plan = composition.pipeline._plan_resolver(  # noqa: SLF001 - exact deployment seam
            composition.manifest, candidate
        )
        assert type(plan) is B300ScreenExecutionPlan
        assert plan.service_digest == composition.manifest.digest
        assert plan.eager_launch.arena_digest == composition.manifest.digest
        assert plan.graph_launch.arena_digest == composition.manifest.digest
        assert plan.model_mount.arena_digest == composition.manifest.digest
        assert plan.binding.physical_hardware.physical_gpu_ids == (
            "0",
            "1",
            "2",
            "3",
        )
        assert plan.eager_launch.hardware.tp_size == 4
        assert plan.eager_session.engine_config.disable_cuda_graph is True
        assert plan.graph_session.engine_config.disable_cuda_graph is False
        assert plan.eager_launch.tree_digest == plan.graph_launch.tree_digest
        assert plan.eager_launch.digest != plan.graph_launch.digest
        composition.pipeline._validate_plan(  # noqa: SLF001 - regression gate
            composition.manifest, candidate, plan
        )
    finally:
        composition.close()


def test_graph_engine_config_derives_from_the_declared_cell() -> None:
    cell = WorkloadCell("s8", 8192, 1024, 64, 8)
    template = deployment._engine_template(
        {"engine_config": _m3_engine_config()}
    )
    eager = deployment._engine_config(
        template, ("msa",), cell, disable_cuda_graph=True
    )
    graph = deployment._engine_config(
        template, ("msa",), cell, disable_cuda_graph=False
    )
    assert "watchdog_timeout" not in eager.engine_kwargs
    assert graph.engine_kwargs["watchdog_timeout"] == 1800
    for config in (eager, graph):
        assert config.engine_kwargs["context_length"] == 8192 + 1024 + 128
        assert config.engine_kwargs["disable_radix_cache"] is True
        assert config.max_running_requests == 64


def test_glm_engine_profile_is_data_not_an_evaluator_branch() -> None:
    glm = _m3_engine_config()
    glm.update(
        engine_kwargs={
            "chunked_prefill_size": 16384,
            "disable_radix_cache": True,
            "dp_size": 4,
            "enable_dp_attention": True,
            "kv_cache_dtype": "auto",
            "quantization": "modelopt_fp4",
            "trust_remote_code": True,
        },
        mem_fraction_static=0.80,
        moe_runner_backend=None,
    )
    template = deployment._engine_template({"engine_config": glm})
    cell = WorkloadCell("l65", 65536, 4096, 24, 3)
    config = deployment._engine_config(
        template,
        ("moe.fused_routed_experts",),
        cell,
        disable_cuda_graph=False,
    )

    assert config.engine_kwargs["enable_dp_attention"] is True
    assert config.engine_kwargs["dp_size"] == 4
    assert config.engine_kwargs["chunked_prefill_size"] == 16384
    assert config.engine_kwargs["context_length"] == 65536 + 4096 + 128
    assert config.moe_runner_backend is None
    assert deployment._data_parallel_size(config) == 4

    mixed = deployment._engine_config(
        template,
        ("moe.fused_routed_experts",),
        (
            WorkloadCell("s8", 8192, 1024, 128, 2),
            WorkloadCell("l65", 65536, 4096, 24, 3),
        ),
        disable_cuda_graph=False,
    )
    assert mixed.max_running_requests == 128
    assert mixed.engine_kwargs["context_length"] == 65536 + 4096 + 128


def test_workload_parser_seals_cell_against_batches() -> None:
    prompt = {
        "prompt_seed_scheme": "sealed-prompts-v1",
        "workload_cell": {
            "cell_id": "s8",
            "concurrency": 2,
            "input_tokens": 8192,
            "output_tokens": 1024,
            "timed_reads": 2,
        },
    }
    batches = (("a", "b"), ("c", "d"), ("e", "f"))
    workload = deployment._workload(prompt, batches, _h("corpus"))
    assert deployment._scored_cell(workload).concurrency == 2
    assert workload.prompt_corpus_digest == _h("corpus")

    with pytest.raises(deployment.B300ScreenDeploymentError, match="concurrency"):
        deployment._workload(prompt, (("a",), ("b", "c"), ("d", "e")), _h("corpus"))
    widened = dict(prompt, workload_cell=dict(prompt["workload_cell"], extra=1))
    with pytest.raises(deployment.B300ScreenDeploymentError, match="closed"):
        deployment._workload(widened, batches, _h("corpus"))

    mixed_prompt = {
        "prompt_seed_scheme": "sealed-prompts-v1",
        "workload_cells": [
            {
                "cell_id": "s8",
                "concurrency": 2,
                "input_tokens": 8192,
                "output_tokens": 1024,
                "timed_reads": 2,
            },
            {
                "cell_id": "l65",
                "concurrency": 1,
                "input_tokens": 65536,
                "output_tokens": 4096,
                "timed_reads": 3,
            },
        ],
        "prompt_batch_cells": ["s8", "s8", "l65", "l65", "l65"],
    }
    mixed_batches = (("a", "b"), ("c", "d"), ("e",), ("f",), ("g",))
    mixed = deployment._workload(mixed_prompt, mixed_batches, _h("corpus"))
    assert tuple(cell.cell_id for cell in mixed.cells) == ("s8", "l65")
    assert deployment._prompt_batch_cells(
        mixed_prompt, mixed_batches, mixed
    ) == ("s8", "s8", "l65", "l65", "l65")


def test_registered_targets_are_sealed_arena_data() -> None:
    from cacheon.target_catalog import default_target_catalog

    catalog = default_target_catalog()
    registered, closed = deployment._target_partition(
        {"registered_targets": list(M3_REGISTERED_TARGET_IDS)}, catalog
    )
    assert registered == M3_REGISTERED_TARGET_IDS
    assert set(closed).isdisjoint(M3_REGISTERED_TARGET_IDS)
    assert set(closed) | set(M3_REGISTERED_TARGET_IDS) == set(
        row["target_id"] for row in catalog.snapshot()["targets"]
    )

    glm_registered, glm_closed = deployment._target_partition(
        {"registered_targets": list(GLM53_REGISTERED_TARGET_IDS)}, catalog
    )
    assert glm_registered == GLM53_REGISTERED_TARGET_IDS
    assert set(glm_closed).isdisjoint(GLM53_REGISTERED_TARGET_IDS)
    assert set(glm_closed) | set(GLM53_REGISTERED_TARGET_IDS) == set(
        row["target_id"] for row in catalog.snapshot()["targets"]
    )
    with pytest.raises(
        deployment.B300ScreenDeploymentError, match="not registered"
    ):
        deployment._target_partition(
            {"registered_targets": ["made.up_target"]}, catalog
        )
    # An authority that omits the field fails commissioning closed.
    with pytest.raises(
        deployment.B300ScreenDeploymentError, match="list of strings"
    ):
        deployment._target_partition({}, catalog)


def test_resident_intake_is_traversable_by_non_owner(tmp_path: Path) -> None:
    paths, gpus, _ready = _case(tmp_path)
    inputs = deployment._authority_inputs(
        **paths,
        provisioner=None,
        provisioned_gpus=gpus,
    )
    composition = deployment._compose(inputs)
    lifetime = None
    try:
        lifetime = composition.authorities.resident_screen_factory.create()
        intake = inputs.root / "resident-intake"
        assert intake.is_dir()
        assert stat.S_IMODE(intake.stat().st_mode) == 0o711
        # exist_ok must not leave a prior private root in place
        intake.chmod(0o700)
        lifetime.close()
        lifetime = composition.authorities.resident_screen_factory.create()
        assert stat.S_IMODE(intake.stat().st_mode) == 0o711
    finally:
        if lifetime is not None:
            lifetime.close()
        composition.close()


def test_commissioned_resident_factory_builds_real_stock_lifetime(
    tmp_path: Path,
) -> None:
    paths, gpus, _ready = _case(tmp_path)
    inputs = deployment._authority_inputs(
        **paths,
        provisioner=None,
        provisioned_gpus=gpus,
    )
    composition = deployment._compose(inputs)
    lifetime = None
    try:
        lifetime = composition.authorities.resident_screen_factory.create()
        assert type(lifetime) is B300ResidentScreenLifetime
        assert composition.build_executor is not composition.resident_executor
        assert (
            composition.build_executor.manager
            is not composition.resident_executor.manager
        )
        stock_roots = tuple((inputs.root / "engine-trees").glob("resident-stock-*"))
        assert len(stock_roots) == 1
    finally:
        if lifetime is not None:
            lifetime.close()
        composition.close()


def test_ready_gpu_ids_require_one_canonical_eight_device_set(
    tmp_path: Path,
) -> None:
    _paths, _gpus, ready = _case(tmp_path)

    short = json.loads(json.dumps(ready))
    short["gpu"]["count"] = 7
    with pytest.raises(
        deployment.B300ScreenDeploymentError,
        match="not an eight-B300 pod",
    ):
        deployment._ready_gpu_ids(short)

    duplicated = json.loads(json.dumps(ready))
    duplicated["gpu"]["inventory"][7]["index"] = 6
    with pytest.raises(
        deployment.B300ScreenDeploymentError,
        match="one canonical eight-device set",
    ):
        deployment._ready_gpu_ids(duplicated)

    unordered = json.loads(json.dumps(ready))
    unordered["gpu"]["inventory"][0]["index"] = 1
    unordered["gpu"]["inventory"][1]["index"] = 0
    with pytest.raises(
        deployment.B300ScreenDeploymentError,
        match="one canonical eight-device set",
    ):
        deployment._ready_gpu_ids(unordered)


def test_materializer_refuses_id_drift_and_non_b300_names(tmp_path: Path) -> None:
    paths, gpus, _ready = _case(tmp_path)
    drifted = gpus[:7] + (_gpu(9),)
    with pytest.raises(
        deployment.B300ScreenDeploymentError,
        match="exact commissioned eight-B300 pair",
    ):
        deployment.materialize_b300_screen_identities(
            **paths,
            gpu_provisioner=lambda selected, *, deadline: drifted,
        )

    renamed = gpus[:7] + (replace(gpus[7], name="NVIDIA H100 SXM5"),)
    with pytest.raises(
        deployment.B300ScreenDeploymentError,
        match="exact commissioned eight-B300 pair",
    ):
        deployment.materialize_b300_screen_identities(
            **paths,
            gpu_provisioner=lambda selected, *, deadline: renamed,
        )


def test_materializer_refuses_lane_absent_from_eight_device_pair(
    tmp_path: Path,
) -> None:
    paths, gpus, ready = _case(tmp_path)
    mutated = json.loads(json.dumps(ready))
    mutated["lane"]["devices"] = [8, 9, 10, 11]
    ready_path = paths["ready_receipt"]
    ready_path.chmod(0o600)
    _write(ready_path, mutated)
    with pytest.raises(
        deployment.B300ScreenDeploymentError,
        match="screen lane is absent from the eight-device pair",
    ):
        deployment.materialize_b300_screen_identities(
            **paths,
            gpu_provisioner=lambda selected, *, deadline: gpus,
        )
