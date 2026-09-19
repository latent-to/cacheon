"""The generic node adapter: any named module of the served model is a slot."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cacheon import audit
from cacheon.integrations import sglang_nodes as nodes
from cacheon.registry import Eligibility, KernelImpl, KernelRegistry


class _Block(nn.Module):
    """Stock node with every side effect the check must handle.

    It rewrites its input in place, writes this batch's cache rows and advances the
    recurrent state of the batch's requests.
    """

    def __init__(self, runner):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1,), 2.0))
        self.runner = [runner]

    def forward(self, x, batch):
        runner = self.runner[0]
        x.add_(1.0)
        runner.token_to_kv_pool.k_buffer[0][batch.out_cache_loc] = x[:, :2]
        index = runner.req_to_token_pool.get_mamba_indices(batch.req_pool_indices)
        runner.req_to_token_pool.mamba_pool.mamba_cache.temporal[0, index] += 1.0
        return x * self.scale


def _served_model():
    cache = SimpleNamespace(conv=[torch.zeros(1, 4, 3)], temporal=torch.zeros(1, 4, 3))
    runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(k_buffer=[torch.zeros(8, 2)], v_buffer=[torch.zeros(8, 2)]),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba_cache=cache),
            get_mamba_indices=lambda index: index,
        ),
    )
    model = nn.Module()
    model.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.mlp = _Block(runner)
        model.layers.append(layer)
    runner.model = model
    batch = SimpleNamespace(
        out_cache_loc=torch.tensor([1, 2]), req_pool_indices=torch.tensor([0])
    )
    return runner, batch


def _registry(slot, entry, *, prepare=None):
    registry = KernelRegistry()
    registry.register(
        KernelImpl(
            slot=slot, bundle_id="test", entry=entry, prepare=prepare,
            eligibility=Eligibility(dtypes=frozenset({"float32"})),
        )
    )
    registry.enable()
    return registry


@pytest.fixture()
def audited(monkeypatch):
    monkeypatch.setattr(audit, "sampled", lambda: True)
    monkeypatch.setattr(audit, "_stats", {})
    monkeypatch.setattr(nodes._receipts, "completed", lambda slot: None)
    return audit._stats


def _do_nothing(module, *args, **kwargs):
    return module.forward(*args, **kwargs)


def test_star_stands_for_exactly_one_segment():
    pattern = nodes.node_pattern("model.layers.*.mlp")
    assert pattern.match("model.layers.3.mlp")
    assert not pattern.match("model.layers.3.mlp.experts")
    assert not pattern.match("model.layers.mlp")
    assert nodes.node_pattern("model").match("model")


def test_one_row_binds_every_matching_node_and_a_do_nothing_candidate_passes(audited):
    runner, batch = _served_model()
    slot = "layers.*.mlp"
    assert nodes.bind(runner, _registry(slot, _do_nothing)) == ["layers.0.mlp", "layers.1.mlp"]
    x = torch.randn(2, 4)
    expected = (x + 1.0) * 2.0
    for layer in runner.model.layers:
        assert torch.allclose(layer.mlp(x.clone(), batch), expected)
    assert (audited[slot]["n"], audited[slot]["violations"]) == (2, 0)
    # Stock ran, then the candidate: without the restore the state would read 4.
    assert runner.req_to_token_pool.mamba_pool.mamba_cache.temporal[0, 0, 0] == 2.0


def test_candidate_sees_the_call_stock_saw_and_a_wrong_answer_is_a_violation(audited):
    runner, batch = _served_model()
    seen = []

    def wrong(module, x, batch):
        seen.append(x.clone())
        return module.forward(x, batch) * 1.5

    nodes.bind(runner, _registry("layers.0.mlp", wrong))
    x = torch.randn(2, 4)
    original = x.clone()
    runner.model.layers[0].mlp(x, batch)
    assert torch.equal(seen[0], original)
    assert audited["layers.0.mlp"]["violations"] == 1


def test_a_candidate_that_leaves_its_arguments_alone_passes_where_stock_overwrites_them(audited):
    runner, batch = _served_model()

    def out_of_place(module, x, batch):
        runner.token_to_kv_pool.k_buffer[0][batch.out_cache_loc] = (x + 1.0)[:, :2]
        index = runner.req_to_token_pool.get_mamba_indices(batch.req_pool_indices)
        runner.req_to_token_pool.mamba_pool.mamba_cache.temporal[0, index] += 1.0
        return (x + 1.0) * module.scale

    nodes.bind(runner, _registry("layers.0.mlp", out_of_place))
    x = torch.randn(2, 4)
    original = x.clone()
    runner.model.layers[0].mlp(x, batch)
    assert torch.equal(x, original)
    assert (audited["layers.0.mlp"]["n"], audited["layers.0.mlp"]["violations"]) == (1, 0)


def test_skipping_a_state_write_is_a_violation_even_with_the_right_output(audited):
    runner, batch = _served_model()

    def stateless(module, x, batch):
        return (x + 1.0) * module.scale

    nodes.bind(runner, _registry("layers.0.mlp", stateless))
    runner.model.layers[0].mlp(torch.randn(2, 4), batch)
    assert audited["layers.0.mlp"]["violations"] == 1


def test_a_result_that_is_a_record_is_graded_through_its_fields(audited):
    class _Head(nn.Module):
        def forward(self, x):
            return SimpleNamespace(next_token_logits=x * 2.0, note="not a tensor")

    runner, _ = _served_model()
    runner.model.head = _Head()

    def wrong(module, x):
        result = module.forward(x)
        result.next_token_logits = result.next_token_logits * 1.5
        return result

    nodes.bind(runner, _registry("head", wrong))
    runner.model.head(torch.randn(2, 4))
    assert audited["head"]["violations"] == 1


def test_prepare_runs_once_per_node_and_nested_nodes_serve_stock_to_a_running_candidate(audited):
    runner, batch = _served_model()
    prepared = []

    def prepare(module):
        prepared.append(module)
        return module

    registry = _registry("layers.*.mlp", _do_nothing, prepare=prepare)
    nodes.bind(runner, registry)
    for _ in range(2):
        runner.model.layers[0].mlp(torch.randn(2, 4), batch)
    # The do-nothing candidate re-enters the bound forward; it must get stock, not itself.
    assert prepared == [runner.model.layers[0].mlp]


def test_out_of_domain_call_serves_stock_and_a_raising_candidate_never_does(monkeypatch):
    completed, failed = [], []
    monkeypatch.setattr(audit, "sampled", lambda: False)
    monkeypatch.setattr(nodes._receipts, "completed", completed.append)
    monkeypatch.setattr(
        nodes._receipts, "failed",
        lambda slot, exc, **_details: failed.append((slot, type(exc).__name__)),
    )
    runner, batch = _served_model()

    def boom(module, *args, **kwargs):
        raise RuntimeError("candidate path failed")

    registry = _registry("layers.1.mlp", boom)
    registry.register(
        KernelImpl(
            slot="layers.0.mlp", bundle_id="test", entry=boom, prepare=None,
            eligibility=Eligibility(dtypes=frozenset({"float16"})),
        )
    )
    nodes.bind(runner, registry)
    # A declared domain that excludes the call is not a fallback: stock is the
    # answer and no receipt claims the candidate ran.
    x = torch.randn(2, 4)
    assert torch.allclose(runner.model.layers[0].mlp(x.clone(), batch), (x + 1.0) * 2.0)
    assert completed == []
    # A selected candidate that raises takes the run down, receipted by name.
    with pytest.raises(RuntimeError, match="candidate path failed"):
        runner.model.layers[1].mlp(torch.randn(2, 4), batch)
    assert (completed, failed) == ([], [("layers.1.mlp", "RuntimeError")])


@pytest.mark.parametrize("second", ["layers.0.mlp", "layers.0"])
def test_a_slot_that_names_no_module_or_overlaps_another_claim_fails_as_the_candidate(
    monkeypatch, second
):
    failed = []
    monkeypatch.setattr(
        nodes._receipts, "failed",
        lambda slot, exc, **details: failed.append((slot, details.get("phase"))),
    )
    runner, _ = _served_model()
    with pytest.raises(RuntimeError, match="names no module"):
        nodes.bind(runner, _registry("layers.*.attention", _do_nothing))
    # The same node twice, or a node inside another claimed node: the inner one would
    # run as the candidate while the outer node's stock reference is taken.
    registry = _registry("layers.*.mlp", _do_nothing)
    registry.register(
        KernelImpl(
            slot=second, bundle_id="other", entry=_do_nothing, prepare=None,
            eligibility=Eligibility(dtypes=frozenset({"float32"})),
        )
    )
    stock_forward = runner.model.layers[1].mlp.forward
    with pytest.raises(RuntimeError, match="overlaps"):
        nodes.bind(runner, registry)
    assert runner.model.layers[1].mlp.forward == stock_forward
    assert [phase for _, phase in failed] == ["prepare", "prepare"]
