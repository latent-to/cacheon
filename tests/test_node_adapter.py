"""The generic node adapter: any named module of the served model is a slot."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from cacheon import audit, dispatch
from cacheon.integrations import sglang_nodes as nodes
from cacheon.integrations.sglang_dsa_state import dsa_state_rows, state_values
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


@pytest.fixture(params=["one piece", "row by row"])
def audited(monkeypatch, request):
    # Row by row, engine state is copied aside, compared and put back in pieces: the
    # path a 48-request batch's recurrent state takes on the GPU.
    if request.param == "row by row":
        monkeypatch.setattr(nodes, "_PIECE", 1)
    monkeypatch.setattr(audit, "sampled", lambda: True)
    monkeypatch.setattr(audit, "_stats", {})
    monkeypatch.setattr(nodes._receipts, "completed", lambda slot: None)
    # Every call is its own graded unit here; the engine pools rows into windows.
    monkeypatch.setattr(nodes, "_WINDOW", 1)
    monkeypatch.setattr(nodes, "_noise", {})
    monkeypatch.setattr(nodes, "_pooled", {})
    return audit._stats


def _do_nothing(module, *args, **kwargs):
    return module.forward(*args, **kwargs)


def _dsa_pool():
    keys = torch.ones(130, 1, 512).to(torch.float8_e4m3fn)
    scales = torch.full((130, 1, 4), 2.0)
    rope = torch.arange(64).to(torch.bfloat16).expand(130, 1, 64).contiguous()
    mla = torch.cat((keys.view(torch.uint8), scales.view(torch.uint8),
                     rope.view(torch.uint8)), -1)
    index = torch.cat((torch.ones(2, 64 * 128).to(torch.float8_e4m3fn).view(torch.uint8),
                       torch.full((2, 64), 3.0).view(torch.uint8)), -1)
    return SimpleNamespace(
        kv_buffer=[mla], index_k_with_scale_buffer=[index, index[:0]],
        dtype=torch.float8_e4m3fn, dsa_kv_cache_store_fp8=True,
        page_size=64, index_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64,
    )


def test_dsa_records_grade_dequantized_values_and_keep_rotary_bf16():
    pool = _dsa_pool()
    rows = dsa_state_rows(pool, torch.tensor([1, 64, 65]))
    assert len(rows) == 2  # skip-topk's zero-page placeholder owns no state
    buffer, dim, index, held = rows[0]
    raw = buffer.index_select(dim, index)
    expected = torch.cat((torch.full((3, 1, 512), 2.0),
                          torch.arange(64).float().expand(3, 1, 64)), -1)
    assert torch.equal(state_values(raw, held), expected)
    equivalent = raw.clone()
    equivalent[..., :512] = torch.full((3, 1, 512), 2.0).to(torch.float8_e4m3fn).view(torch.uint8)
    equivalent[..., 512:528] = torch.ones(3, 1, 4).view(torch.uint8)
    assert torch.equal(state_values(equivalent, held), expected)
    assert (nodes._row_errors(equivalent.view(pool.dtype), raw.view(pool.dtype), 0) > 0.02).all()
    buffer, dim, pages, held = rows[1]
    assert pages.tolist() == [0, 1]
    assert torch.equal(state_values(buffer.index_select(dim, pages), held), torch.full((2, 64, 128), 3.0))


@pytest.mark.parametrize("corrupt", [False, True])
def test_dsa_sidecar_is_restored_before_candidate_and_wrong_state_fails(audited, corrupt):
    runner, batch = _served_model()
    runner.token_to_kv_pool = pool = _dsa_pool()
    batch.out_cache_loc = torch.tensor([1, 64])

    class DsaNode(nn.Module):
        def forward(self, x, batch):
            pool.index_k_with_scale_buffer[0][:, :64 * 128] += 1
            return x * 2

    seen = []
    before = pool.index_k_with_scale_buffer[0].clone()

    def candidate(module, x, batch):
        seen.append(pool.index_k_with_scale_buffer[0].clone())
        result = module.forward(x, batch)
        if corrupt:
            pool.index_k_with_scale_buffer[0].zero_()
        return result

    runner.model.dsa = DsaNode()
    nodes.bind(runner, _registry("dsa", candidate))
    assert torch.equal(runner.model.dsa(torch.ones(2, 4), batch), torch.full((2, 4), 2.0))
    assert torch.equal(seen[0], before)
    assert audited["dsa"]["violations"] == int(corrupt)


def test_dsa_indexer_twin_uses_supported_children_and_still_rejects_wrong_output(audited, monkeypatch):
    class Indexer(_Wide):
        def __init__(self):
            super().__init__()
            self._forward_method = self.forward_fast

        def forward_fast(self, x):
            return self.op(x)

        def forward(self, x):
            return self._forward_method(x)

        def forward_native(self, x):
            raise NotImplementedError("Indexer has no native (pure-torch) path")

    module = ModuleType("sglang.srt.layers.attention.dsa.dsa_indexer")
    module.Indexer = Indexer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runner, _ = _served_model()
    runner.model.indexer = indexer = Indexer()
    with nodes._native(indexer):
        assert torch.allclose(indexer(torch.ones(2, 4)), torch.full((2, 4), 2.1))
    assert indexer._forward_method == indexer.forward_fast
    nodes.bind(runner, _registry("indexer", lambda m, x: m.forward(x) * 1.5))
    indexer(torch.ones(2, 4))
    assert (audited["indexer"]["violations"], audited["indexer"]["baseline_refused"]) == (1, 0)
    def broken_native(x):
        raise RuntimeError("native failed")
    monkeypatch.setattr(indexer.op, "forward_native", broken_native)
    with pytest.raises(RuntimeError, match="native failed"), nodes._native(indexer.op):
        indexer.op(torch.ones(2, 4))


def test_wildcard_failure_log_names_each_concrete_node(audited, caplog):
    caplog.set_level("ERROR")  # The commissioned GLM engine suppresses warnings.
    runner, batch = _served_model()
    nodes.bind(runner, _registry("layers.*.mlp", lambda m, *a: m.forward(*a) * 1.5))
    for layer in runner.model.layers:
        layer.mlp(torch.ones(2, 4), batch)
    assert "node=layers.0.mlp tensor_position=0" in caplog.text
    assert "node=layers.1.mlp tensor_position=0" in caplog.text


def test_unordered_choices_preserve_members_and_multiplicity():
    expected = torch.tensor([[-1, -1, 10, 20], [10, 20, 30, 40], [10, 20, 30, 40]])
    actual = torch.tensor([[-1, 20, -1, 10], [10, 10, 10, 10], [20, 30, 40, 50]])
    assert torch.equal(nodes._row_errors(actual, expected, 0, unordered=True),
                       torch.tensor([0.0, 0.75, 0.25]))
    assert nodes._row_errors(actual[:1], expected[:1], 0).item() == 0.75


@pytest.mark.parametrize("kind", ["layer", "attention", "indexer"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_dsa_choice_permutation_is_not_a_false_failure(audited, monkeypatch, corrupt, kind):
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = SimpleNamespace(use_dsa=True)
            self.use_dsa = True
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            ids = torch.arange(8).repeat(x.shape[0], 1).roll(self.calls, -1)
            if kind == "indexer":
                return ids
            return (x * 2, x + 1, ids) if kind == "layer" else (x * 2, ids)

    module_name = ("sglang.srt.layers.attention.dsa.dsa_indexer" if kind == "indexer"
                   else "sglang.srt.models.deepseek_v2")
    model = ModuleType(module_name)
    setattr(model, {"layer": "DeepseekV2DecoderLayer", "attention": "DeepseekV2AttentionMLA",
                    "indexer": "Indexer"}[kind], Decoder)
    monkeypatch.setitem(sys.modules, model.__name__, model)
    runner, _ = _served_model()
    runner.model.decoder = Decoder()

    def candidate(module, x):
        result = module.forward(x)
        ids = result if kind == "indexer" else result[-1]
        ids = ids // 4 * 4 if corrupt else ids
        return ids if kind == "indexer" else (*result[:-1], ids)

    nodes.bind(runner, _registry("decoder", candidate))
    result = runner.model.decoder(torch.ones(4, 4))
    if kind != "indexer":
        assert torch.equal(result[0], torch.full((4, 4), 2.0))
    assert audited["decoder"]["violations"] == int(corrupt)


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


class _Fused(nn.Module):
    """Stands in for an SGLang fused op: a fast path, a native reference path, and a
    dispatch target that stays empty until the first call caches it."""

    def __init__(self):
        super().__init__()
        self._forward_method = None

    def forward_fast(self, x):
        return x * 2.0

    def forward_native(self, x):
        return x * 2.0 * 1.05

    def forward(self, x):
        if self._forward_method is None:
            self._forward_method = self.forward_fast
        return self._forward_method(x)


class _Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = _Fused()

    def forward(self, x):
        return self.op(x)


def test_the_tolerance_is_the_noise_the_native_twin_shows_on_the_same_call(audited):
    runner, _ = _served_model()
    runner.model.wide, runner.model.leaf = _Wide(), nn.Identity()

    def off_by_eight_percent(module, x):
        return module.forward(x) * 1.08

    registry = _registry("wide", off_by_eight_percent)
    registry.register(
        KernelImpl(
            slot="leaf", bundle_id="test", entry=off_by_eight_percent, prepare=None,
            eligibility=Eligibility(dtypes=frozenset({"float32"})),
        )
    )
    nodes.bind(runner, registry)
    # Twice: the op caches its dispatch target on the first call, after the node's
    # methods were sealed, and the second audited call must not read that as a rebind.
    runner.model.wide(torch.randn(4, 8))
    runner.model.wide(torch.randn(4, 8))
    runner.model.leaf(torch.randn(4, 8))
    # The twin sits 5% from stock inside the wide node, so 8% is rounding there; the
    # leaf has no native path, its twin is stock, and the same 8% is a wrong answer.
    assert (audited["wide"]["violations"], audited["leaf"]["violations"]) == (0, 1)
    assert runner.model.wide.op._forward_method == runner.model.wide.op.forward_fast


@pytest.mark.parametrize("attr", ["forward_native", "_forward_method"])
def test_a_candidate_cannot_widen_its_tolerance_by_making_the_twin_noisy(
    audited, monkeypatch, attr
):
    failed = []
    monkeypatch.setattr(
        nodes._receipts, "failed", lambda slot, exc, **_d: failed.append((slot, str(exc)))
    )
    runner, _ = _served_model()
    runner.model.wide = _Wide()

    def prepare(module):
        # Hostile: the twin's native path now answers garbage, so its measured noise,
        # and with it the tolerance, would be as wide as the candidate likes; or the
        # op's dispatch target is the candidate's, so stock agrees with it.
        setattr(module.op, attr, lambda x: x * 40.0)
        return module

    nodes.bind(runner, _registry("wide", lambda module, x: module.forward(x) * 1.5, prepare=prepare))
    # The first audited call takes both references before ``prepare`` has ever run.
    runner.model.wide(torch.randn(4, 8))
    assert audited["wide"]["violations"] == 1
    with pytest.raises(RuntimeError, match=rf"rebound after binding \(_Fused: {attr}\)"):
        runner.model.wide(torch.randn(4, 8))
    assert failed and failed[0][0] == "wide"


def test_the_measured_tolerance_has_a_ceiling_below_a_wrong_answer(audited, monkeypatch):
    runner, _ = _served_model()
    runner.model.wide = _Wide()
    # A twin that is honestly 30% noisy would earn 0.9; the ceiling holds it at 0.4.
    monkeypatch.setattr(_Fused, "forward_native", lambda self, x: x * 1.3)
    nodes.bind(runner, _registry("wide", lambda module, x: module.forward(x) * 1.5))
    runner.model.wide(torch.randn(4, 8))
    assert audited["wide"]["violations"] == 1


def test_a_wrong_answer_on_tiny_activations_is_still_a_violation(audited):
    # Early-layer MoE outputs sit below 0.04, where a flat absolute tolerance hid a
    # 1.5x-wrong block on 109 of 240 calls; a row's relative error has no scale.
    runner, _ = _served_model()
    runner.model.leaf = nn.Identity()
    nodes.bind(runner, _registry("leaf", lambda module, x: x * 1.5))
    runner.model.leaf(torch.randn(64, 64) * 1e-6)
    assert audited["leaf"]["violations"] == 1


@pytest.mark.parametrize("flipped, violations", [(1, 0), (5, 1)])
def test_rows_pool_across_calls_so_one_flipped_token_is_not_a_verdict(
    audited, monkeypatch, flipped, violations
):
    monkeypatch.setattr(nodes, "_WINDOW", 16)
    runner, _ = _served_model()
    runner.model.leaf = nn.Identity()
    calls = iter(range(16))

    def flips_some_rows(module, x):
        return x * (1.5 if next(calls) < flipped else 1.0)

    nodes.bind(runner, _registry("leaf", flips_some_rows))
    for _ in range(16):
        runner.model.leaf(torch.randn(1, 8))
    assert (audited["leaf"]["n"], audited["leaf"]["violations"]) == (1, violations)


def test_skipping_a_state_write_is_a_violation_even_with_the_right_output(audited):
    runner, batch = _served_model()

    def stateless(module, x, batch):
        return (x + 1.0) * module.scale

    nodes.bind(runner, _registry("layers.0.mlp", stateless))
    runner.model.layers[0].mlp(torch.randn(2, 4), batch)
    assert audited["layers.0.mlp"]["violations"] == 1


@pytest.mark.parametrize(("written", "violations"), [("negative zero", 0), ("doubled", 1)])
def test_an_fp8_cache_stored_as_bytes_is_graded_as_the_numbers_it_holds(
    audited, written, violations
):
    fp8 = torch.float8_e4m3fn

    class _Fp8Block(nn.Module):
        def __init__(self, runner):
            super().__init__()
            self.runner = [runner]

        def forward(self, x, batch, *, scale=1.0, zero=0.0):
            rows = torch.cat([x[:, :3] * scale, torch.full((x.shape[0], 1), zero)], dim=1)
            self.runner[0].token_to_kv_pool.k_buffer[0][batch.out_cache_loc] = (
                rows.to(fp8).view(torch.uint8)
            )
            return x * 2.0

    runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            dtype=fp8, k_buffer=[torch.zeros(8, 4, dtype=torch.uint8)]
        ),
        req_to_token_pool=SimpleNamespace(),
    )
    runner.model = nn.Module()
    runner.model.leaf = _Fp8Block(runner)
    batch = SimpleNamespace(
        out_cache_loc=torch.tensor([1, 2]), req_pool_indices=torch.tensor([0])
    )

    def candidate(module, x, batch):
        # As bytes -0.0 is 128 away from +0.0; as a number it is the same value.
        how = dict(zero=-0.0) if written == "negative zero" else dict(scale=2.0)
        return module.forward(x, batch, **how)

    nodes.bind(runner, _registry("leaf", candidate))
    runner.model.leaf(torch.rand(2, 4) + 1.0, batch)
    assert (audited["leaf"]["n"], audited["leaf"]["violations"]) == (1, violations)


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


@pytest.mark.parametrize(("picks", "violations"), [("stock's", 0), ("a quarter of the experts", 1)])
def test_whole_number_results_are_graded_as_choices_not_magnitudes(audited, picks, violations):
    class _Router(nn.Module):
        def forward(self, x):
            weights, ids = x.topk(8, dim=-1)
            return SimpleNamespace(topk_weights=weights, topk_ids=ids)

    runner, _ = _served_model()
    runner.model.router = _Router()

    def candidate(module, x):
        result = module.forward(x)
        if picks != "stock's":
            # Three ids in four move by at most 3 of 255: as magnitudes, under the floor.
            result.topk_ids = result.topk_ids // 4 * 4
        return result

    nodes.bind(runner, _registry("router", candidate))
    runner.model.router(torch.randn(64, 256))
    assert (audited["router"]["n"], audited["router"]["violations"]) == (1, violations)


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


def test_cuda_graph_detector_supports_current_legacy_and_direct_capture(monkeypatch):
    current_name = (
        "sglang.srt.model_executor.runner_backend_utils."
        "tc_piecewise_cuda_graph"
    )
    legacy_name = "sglang.srt.compilation.piecewise_context_manager"
    current = ModuleType(current_name)
    legacy = ModuleType(legacy_name)
    current.is_in_tc_piecewise_cuda_graph = lambda: True
    legacy.is_in_piecewise_cuda_graph = lambda: False
    monkeypatch.setitem(sys.modules, current_name, current)
    monkeypatch.setitem(sys.modules, legacy_name, legacy)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dispatch._in_cuda_graph()

    current.is_in_tc_piecewise_cuda_graph = lambda: False
    legacy.is_in_piecewise_cuda_graph = lambda: True
    assert dispatch._in_cuda_graph()

    legacy.is_in_piecewise_cuda_graph = lambda: False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert dispatch._in_cuda_graph()
