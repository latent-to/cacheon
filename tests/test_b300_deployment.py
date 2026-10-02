"""Commission/replay contracts for TP4/B300 and TP1/H100 allocations."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import cacheon.eval.b300_arena_definition as definition
import cacheon.eval.b300_deployment as deployment
from cacheon.arena_service import ArenaCapacityPolicy
from cacheon.eval.b300_arena_provider import B300DeclaredAuthorities
from cacheon.eval.device_state import GPUConfiguration
from cacheon.eval.oci_backend import runtime_identity_from_preflight
from cacheon.eval.runtime_preflight import RuntimePreflightReceipt
from tests.support.b300 import (
    NODE_AND_CACHE_TARGET_IDS,
    NODE_TARGET_IDS,
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


def _preflight(version="0.5.18") -> RuntimePreflightReceipt:
    return preflight_receipt(
        image=_h("image"),
        platform=_h("platform"),
        worker=_h("worker-distribution"),
        sglang_version=version,
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


def _qwen_engine_config():
    config = _m3_engine_config()
    config.update(tp_size=1, mem_fraction_static=0.93, moe_runner_backend="triton")
    # The measured H100 arena settings. SGLang 0.5.19 refuses language_model_only for
    # this model, and chunked prefill at 4096 served 1,599 tok/s against 1,903 at the
    # engine default (2026-09-20).
    config["engine_kwargs"] = {
        "disable_radix_cache": True,
        "cuda_graph_bs_decode": [1, 2, 4, 8, 16, 24, 32, 40, 48],
        "kv_cache_dtype": "fp8_e4m3",
        "max_mamba_cache_size": 48, "mamba_ssm_dtype": "float32",
    }
    return config


def _case(
    tmp_path: Path, *, gpu_model="b300", host_size=8,
    lane=(0, 1, 2, 3), baseline=None,
) -> tuple[dict[str, Path], tuple[GPUConfiguration, ...], dict[str, object]]:
    tmp_path.chmod(0o700)
    model = tmp_path / "model"
    model.mkdir(mode=0o700)
    (model / "config.json").write_text("{}\n")
    (model / "config.json").chmod(0o400)
    preflight = _preflight("0.5.18" if gpu_model == "b300" else "0.5.19")
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
        "registered_targets": list(NODE_TARGET_IDS),
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
    if gpu_model == "h100":
        prompt_value.update(
            model_profile_key="Qwen3.6-35B-A3B-BF16", engine_config=_qwen_engine_config(),
        )
    prompt_value["engine_config"]["tp_size"] = len(lane)
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
            "architecture": "sm103" if gpu_model == "b300" else "sm90",
            "gpu_count": len(lane),
            "lane": list(map(str, lane)),
            "lane_digest": lane_digest,
            "tensor_parallel_size": len(lane),
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
    gpus = tuple(_gpu(index, gpu_model) for index in range(host_size))
    inventory = [
        {
            "index": index,
            "memory_mib": gpus[index].memory_total_mib,
            "name": gpus[index].name,
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            "uuid": gpus[index].uuid,
        }
        for index in range(host_size)
    ]
    ready_value = {
        "created_at_unix": 1,
        "gpu": {"count": host_size, "inventory": inventory,
                "inventory_sha256": _h("inventory"), "topology_sha256": _h("topology")},
        "lane": {
            "devices": list(lane),
            "lane_digest": _h("ready-lane"),
            "tensor_parallel_size": len(lane),
        },
        "model": {
            "content_digest": _h("model-content"),
            "path": str(model),
            "readonly_inventory_verified": True,
            "receipt_digest": _h("model-receipt"), "receipt_file_sha256": _h("model-file"),
            "receipt_path": str(model / "receipt.json"),
        },
        "provider": {"hostname": "worker", "machine_id_sha256": _h("machine"), "pod_endpoint": "worker:22"},
        "python": {"path": sys.executable, "resolved_path": str(Path(sys.executable).resolve()),
                   "executable_sha256": _h("python"), "version": "Python 3.12.0"},
        "runtime": {"path": str(tmp_path / "runtime"), "tree_digest": _h("runtime-tree")},
        "schema": "cacheon-current-pod-commission-v1",
        "source": {"path": str(tmp_path / "source"), "tree_digest": _h("source-tree"), "revision": "1" * 40},
        "state": "READY_FOR_REGISTRATION",
        "worker_epoch": "1" * 32,
        "worker_image": preflight.requested_image,
    }
    if baseline is not None:
        ready_value["lane"]["baseline_devices"] = list(baseline)
    from cacheon.chain.remote_worker_spool import spool_digest
    ready_value["receipt_digest"] = spool_digest("cacheon.current-pod-commission.v1", ready_value)
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


@pytest.mark.parametrize("model,host_size,lane,baseline", (
    ("b300", 8, (0, 1, 2, 3), None),
    ("h100", 2, (0,), None),
    ("h100", 8, (2,), (5,)),
))
def test_materialize_and_replay_exact_service_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model, host_size, lane, baseline,
) -> None:
    paths, gpus, ready = _case(tmp_path, gpu_model=model, host_size=host_size, lane=lane, baseline=baseline)
    from cacheon.chain.remote_worker_registration import verify_ready_receipt
    assert verify_ready_receipt(ready) == ready
    other = baseline or tuple(index for index in range(host_size) if index not in lane)
    allocated = tuple(sorted((*lane, *other)))
    assert deployment._device_policy(tuple(gpus[index] for index in lane)).drain_timeout_s == 300.0

    def provision(selected, *, deadline):
        assert selected == allocated
        return tuple(gpus[index] for index in selected)

    result = deployment.materialize_b300_identities(
        **paths,
        gpu_provisioner=provision,
    )
    output = paths["output_root"]
    manifest = deployment._manifest_from_dict(
        json.loads((output / deployment.MANIFEST_FILE).read_text())
    )
    readiness_row = json.loads((output / deployment.READINESS_FILE).read_text())
    from cacheon.chain.evaluation_coordinator import WorkerReadiness

    readiness = WorkerReadiness(**readiness_row)
    assert result["schema"] == deployment.MATERIALIZATION_SCHEMA
    assert result["service_digest"] == manifest.digest
    assert result["worker_readiness_digest"] == readiness.digest
    assert manifest.runtime.gpu_count == len(lane)
    assert manifest.runtime.tensor_parallel_size == len(lane)
    assert manifest.runtime.target_architecture == ("sm103" if model == "b300" else "sm90")
    assert manifest.runtime.topology_digest == _h("sealed-lane")
    deployment_row = json.loads((output / deployment.DEPLOYMENT_FILE).read_text())
    pair = deployment_row["declared_qualification"]["lane_pair"]
    assert pair["lane_a"]["physical_gpu_ids"] == list(lane)
    assert pair["lane_b"]["physical_gpu_ids"] == list(other)
    assert set(pair["lane_a"]["gpu_uuids"]).isdisjoint(
        pair["lane_b"]["gpu_uuids"]
    )

    registration = {
        "lane_devices": list(lane),
        "ready_receipt_digest": ready["receipt_digest"],
        "service_identity": manifest.service_id,
        "worker_epoch": ready["worker_epoch"],
        "worker_readiness": readiness.to_dict(),
        "worker_readiness_digest": readiness.digest,
    }
    _inputs, composition, replayed = deployment.replay_commissioned_composition(
        registration, ready, commissioned_root=output
    )
    assert composition.manifest == manifest
    assert replayed == readiness

    registration["service_identity"] = manifest.digest
    with pytest.raises(
        deployment.B300DeploymentError,
        match="registration differs",
    ):
        deployment.replay_commissioned_composition(registration, ready, commissioned_root=output)


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

    result_a = deployment.materialize_b300_identities(
        **paths_a,
        gpu_provisioner=lambda selected, *, deadline: gpus_a,
    )
    result_b = deployment.materialize_b300_identities(
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
    # The provider identity names the declared qualification; calibration names
    # the arena manifest, so it may only ride in the deployment's refs.
    assert (
        deployment_a["declared_qualification"]
        == deployment_b["declared_qualification"]
    )
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
        deployment.B300DeploymentError,
        match="explicit sealed authority paths or SHA-256",
    ):
        deployment.materialize_b300_identities(
            **paths,
            gpu_provisioner=lambda selected, *, deadline: gpus,
        )


def test_materializer_refuses_single_tp4_as_declared_qualification_pair(
    tmp_path: Path,
) -> None:
    paths, gpus, _ready = _case(tmp_path)
    with pytest.raises(
        deployment.B300DeploymentError,
        match="provisioned GPU set differs from the commissioned lane pair",
    ):
        deployment.materialize_b300_identities(
            **paths,
            gpu_provisioner=lambda selected, *, deadline: gpus[:4],
        )


@pytest.mark.parametrize("gpu_model,count", (("b300", 4), ("h100", 1)))
def test_commissioning_seals_only_declared_authority(tmp_path, gpu_model, count):
    paths, gpus, _ready = _case(tmp_path, gpu_model=gpu_model, host_size=count * 2, lane=tuple(range(count)))
    inputs = deployment._authority_inputs(
        **paths,
        provisioner=None,
        provisioned_gpus=gpus,
    )
    composition = deployment._compose(inputs)

    # The qualification worker builds its executors and trees after replay;
    # commissioning seals identities and starts nothing.
    assert type(composition.authorities) is B300DeclaredAuthorities
    assert composition.authorities.qualification == inputs.declared_qualification
    assert composition.manifest.runtime.tensor_parallel_size == count
    assert composition.manifest.capacity == ArenaCapacityPolicy(32, 64, 4, 4)
    assert "screens" not in composition.manifest.to_dict()
    assert not (inputs.root / "oci").exists()
    assert not (inputs.root / "engine-trees").exists()


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
    # The measured arms launch this sealed template with graphs on.
    config = deployment._engine_template({"engine_config": glm})

    assert config.engine_kwargs["enable_dp_attention"] is True
    assert config.engine_kwargs["dp_size"] == 4
    assert config.engine_kwargs["chunked_prefill_size"] == 16384
    assert config.moe_runner_backend is None
    assert definition.data_parallel_size(config) == 4


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
    assert definition.scored_cell(workload).concurrency == 2
    assert workload.prompt_corpus_digest == _h("corpus")

    with pytest.raises(deployment.B300DeploymentError, match="concurrency"):
        deployment._workload(prompt, (("a",), ("b", "c"), ("d", "e")), _h("corpus"))
    widened = dict(prompt, workload_cell=dict(prompt["workload_cell"], extra=1))
    with pytest.raises(deployment.B300DeploymentError, match="closed"):
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
        {"registered_targets": list(NODE_TARGET_IDS)}, catalog
    )
    assert registered == NODE_TARGET_IDS
    assert set(closed).isdisjoint(NODE_TARGET_IDS)
    assert set(closed) | set(NODE_TARGET_IDS) == set(
        row["target_id"] for row in catalog.snapshot()["targets"]
    )

    glm_registered, glm_closed = deployment._target_partition(
        {"registered_targets": list(NODE_AND_CACHE_TARGET_IDS)}, catalog
    )
    assert glm_registered == NODE_AND_CACHE_TARGET_IDS
    assert set(glm_closed).isdisjoint(NODE_AND_CACHE_TARGET_IDS)
    assert set(glm_closed) | set(NODE_AND_CACHE_TARGET_IDS) == set(
        row["target_id"] for row in catalog.snapshot()["targets"]
    )
    with pytest.raises(
        deployment.B300DeploymentError, match="not registered"
    ):
        deployment._target_partition(
            {"registered_targets": ["made.up_target"]}, catalog
        )
    # An authority that omits the field fails commissioning closed.
    with pytest.raises(
        deployment.B300DeploymentError, match="list of strings"
    ):
        deployment._target_partition({}, catalog)


def test_ready_gpu_ids_require_a_counted_canonical_device_set(
    tmp_path: Path,
) -> None:
    _paths, _gpus, ready = _case(tmp_path)

    short = json.loads(json.dumps(ready))
    short["gpu"]["count"] = 7
    with pytest.raises(
        deployment.B300DeploymentError,
        match="count differs from its inventory",
    ):
        definition.ready_gpu_ids(short)

    duplicated = json.loads(json.dumps(ready))
    duplicated["gpu"]["inventory"][7]["index"] = 6
    with pytest.raises(
        deployment.B300DeploymentError,
        match="ordered unique GPU indices",
    ):
        definition.ready_gpu_ids(duplicated)

    unordered = json.loads(json.dumps(ready))
    unordered["gpu"]["inventory"][0]["index"] = 1
    unordered["gpu"]["inventory"][1]["index"] = 0
    with pytest.raises(
        deployment.B300DeploymentError,
        match="ordered unique GPU indices",
    ):
        definition.ready_gpu_ids(unordered)


def test_materializer_refuses_id_and_gpu_model_drift(tmp_path: Path) -> None:
    paths, gpus, _ready = _case(tmp_path)
    drifted = gpus[:7] + (_gpu(9),)
    with pytest.raises(
        deployment.B300DeploymentError,
        match="provisioned GPU set differs from the commissioned lane pair",
    ):
        deployment.materialize_b300_identities(
            **paths,
            gpu_provisioner=lambda selected, *, deadline: drifted,
        )

    renamed = gpus[:7] + (replace(gpus[7], name="NVIDIA H100 SXM5"),)
    with pytest.raises(
        deployment.B300DeploymentError,
        match="GPU configuration differs from READY inventory",
    ):
        deployment.materialize_b300_identities(
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
        deployment.B300DeploymentError,
        match="equal disjoint lanes within its inventory",
    ):
        deployment.materialize_b300_identities(
            **paths,
            gpu_provisioner=lambda selected, *, deadline: gpus,
        )
