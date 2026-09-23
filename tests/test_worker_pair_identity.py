"""A physical pair changes readiness and execution binding, not its competition."""

import json
from dataclasses import replace

import pytest

from cacheon.eval.device_policy_identity import logical_device_policy_digest
from tests import test_b300_screen_deployment as fixture


@pytest.mark.parametrize("gpu_model,width", [("h100", 1), ("b300", 4)])
def test_disjoint_equivalent_pairs_share_service_and_keep_physical_authority(tmp_path, gpu_model, width):
    results, deployments = [], []
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir(mode=0o700)
        start = index * 2 * width
        paths, gpus, _ = fixture._case(
            root, gpu_model=gpu_model, host_size=width * 4,
            lane=tuple(range(start, start + width)),
            baseline=tuple(range(start + width, start + 2 * width)),
        )
        results.append(fixture.deployment.materialize_b300_screen_identities(
            **paths, gpu_provisioner=lambda selected, *, deadline: tuple(gpus[i] for i in selected),
        ))
        deployments.append(json.loads((paths["output_root"] / fixture.deployment.DEPLOYMENT_FILE).read_text()))
    assert results[0]["service_digest"] == results[1]["service_digest"]
    assert results[0]["worker_readiness_digest"] != results[1]["worker_readiness_digest"]
    pairs = [row["declared_qualification"]["lane_pair"] for row in deployments]
    assert pairs[0] != pairs[1]
    assert set(pairs[0]["lane_a"]["gpu_uuids"]).isdisjoint(pairs[1]["lane_a"]["gpu_uuids"])


def test_logical_identity_preserves_clock_memory_driver_and_execution_bounds():
    policy = fixture.deployment._device_policy((fixture._gpu(0),))
    moved = replace(policy, expected_gpus=(replace(
        policy.expected_gpus[0], physical_id=7, uuid=fixture._gpu(7).uuid,
        pci_bus_id=fixture._gpu(7).pci_bus_id,
    ),))
    assert policy.policy_sha256 != moved.policy_sha256
    assert logical_device_policy_digest(policy) == logical_device_policy_digest(moved)
    for field, value in (("power_limit_mw", 200000), ("memory_total_mib", 10000),
                         ("driver_version", "999.1"), ("max_graphics_clock_mhz", 1000)):
        changed = replace(policy, expected_gpus=(replace(policy.expected_gpus[0], **{field: value}),))
        assert logical_device_policy_digest(changed) != logical_device_policy_digest(policy)
    assert logical_device_policy_digest(replace(policy, maximum_samples=2000)) != logical_device_policy_digest(policy)
