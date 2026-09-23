"""Competition identity of a GPU policy, independent of its host allocation."""

from dataclasses import asdict
import hashlib
import json

from cacheon.eval.device_state import DeviceStatePolicy


def logical_device_policy_digest(policy: DeviceStatePolicy) -> str:
    """Bind GPU capabilities and every execution bound, excluding device addresses.

    Physical policies and launch bindings still retain indices, UUIDs and PCI
    addresses. This projection is only for the shared arena service identity;
    it never substitutes for a worker's launch or telemetry policy.
    """
    if type(policy) is not DeviceStatePolicy:
        raise TypeError("logical device policy requires an exact physical policy")
    payload = asdict(policy)
    payload["expected_gpus"] = [
        {key: value for key, value in gpu.items()
         if key not in {"physical_id", "uuid", "pci_bus_id"}}
        for gpu in payload["expected_gpus"]
    ]
    payload["schema"] = "cacheon.eval.logical-device-policy.v1"
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()
