"""Node routing must preserve participation across differing DP batch widths."""

import pytest
import torch

from cacheon import audit
from cacheon.integrations import sglang_nodes as nodes
from cacheon.registry import (
    Eligibility, KernelImpl, KernelRegistry, VariantRegistrationError,
    eligibility_from_metadata,
)


@pytest.mark.parametrize("metadata", [
    {"min_num_tokens": 1}, {"max_num_tokens": 128},
    {"capabilities": {"num_tokens": {"min": 1}}},
    {"capabilities": {"num_tokens": [0, 128]}},
])
def test_node_token_domains_fail_before_execution_and_legacy_domains_remain(metadata):
    registry = KernelRegistry()
    eligibility = eligibility_from_metadata(metadata, ())
    with pytest.raises(VariantRegistrationError, match="inside the entry"):
        registry.register(KernelImpl(
            slot="model.layers.*", bundle_id="candidate", entry=lambda: None,
            eligibility=eligibility,
        ))
    assert registry.slots() == []
    registry.register(KernelImpl(
        slot="norm.fused_add_rmsnorm", bundle_id="retained", entry=lambda: None,
        eligibility=eligibility,
    ))
    assert registry.variants("norm.fused_add_rmsnorm")[0].eligibility == eligibility


def test_packed_parameter_does_not_bypass_the_candidate_at_any_local_batch_width(monkeypatch):
    module = torch.nn.Module()
    module.register_parameter("packed_weight", torch.nn.Parameter(
        torch.zeros(4, 4, dtype=torch.uint8), requires_grad=False,
    ))
    calls = []

    def candidate(prepared, x):
        calls.append(x.shape[0])
        return x + 2

    registry = KernelRegistry()
    registry.register(KernelImpl(
        slot="model.layers.0", bundle_id="packed", entry=candidate,
        eligibility=Eligibility(dtypes=frozenset({"bfloat16"})),
    ))
    registry.enable()
    monkeypatch.setattr(audit, "sampled", lambda: False)
    dispatch = nodes.make_node_dispatcher(
        "model.layers.0", module, lambda x: x + 1, None, [], registry=registry,
    )
    for rows in (0, 1, 17):
        x = torch.zeros(rows, 4, dtype=torch.bfloat16)
        assert torch.equal(dispatch(x), x + 2)
        assert "num_tokens" not in nodes._descriptor(module, (x,), {}, False)
    assert calls == [0, 1, 17]
