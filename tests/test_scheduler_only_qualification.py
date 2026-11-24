"""The scheduler target: what intake admits, what binds, what the audit grades, what parks."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cacheon.engine_tree import EngineTreeError, inspect_contribution
from cacheon.eval.b300_registered_qualification import B300RegisteredQualificationError, _audit_members
from cacheon.eval.engine_worker import _eager_audit_engine
from cacheon.integrations import sglang_engine, sglang_order
from cacheon.manifest import load_manifest
from cacheon.registry import KernelImpl, KernelRegistry, eligibility_from_metadata
from cacheon.seams import ENGINE_OBJECTS
from cacheon.target_catalog import TargetResolutionError, default_target_catalog
from tests.test_chain_intake import _fingerprint, _h, _publish, _reserve_one, _store

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ADDER = "sglang.srt.managers.schedule_policy.PrefillAdder"


def _bundle(root: Path, *slots: str, extra: str = "") -> Path:
    """A bundle whose every op is entry ``order`` of one module, with ``extra`` appended to each op."""
    (root / "engine").mkdir(parents=True)
    (root / "engine" / "order.py").write_text("def order(view):\n    return list(range(len(view)))\n")
    ops = "".join(f'\n[[ops]]\nslot = "{slot}"\nsource = "engine/order.py"\nentry = "order"\n{extra}'
                  for slot in slots)
    (root / "manifest.toml").write_text('bundle_id = "scheduler-surface"\nabi_version = "cacheon-op-abi-v0"\n\n'
                                        '[competition]\ntarget = "scheduler"\nmode = "slot"\n' + ops)
    return root


def test_scheduler_admits_the_queue_order_and_listed_decisions_and_nothing_else(tmp_path):
    catalog = default_target_catalog()
    admitted = ("queue_order", f"{ADDER}.add_one_req", "sglang.srt.mem_cache.prefill_budget.SWAPrefillBudget",
                *ENGINE_OBJECTS)
    for index, slot in enumerate(admitted):
        assert catalog.resolve_manifest(load_manifest(_bundle(tmp_path / f"ok{index}", slot))).target_id == "scheduler"
    # Validator-owned surfaces (audit #3), the two output-length oracles (audit #4), and the model's and the cache's
    # roots, which stay their own targets so a scheduler entry never claims a slot a kernel or cache entry holds.
    for index, slot in enumerate((
        "model.layers.*.mlp", "logits_processor", "tree_cache",
        "sglang.srt.runtime_context.publish", "sglang.srt.managers.scheduler.Scheduler.__init__",
        "sglang.srt.managers.scheduler.Scheduler.flush_cache", "sglang.srt.managers.scheduler.Scheduler.run_batch",
        "sglang.srt.managers.scheduler.Scheduler.init_req_max_new_tokens",
        "sglang.srt.managers.schedule_policy.SchedulePolicy.calc_priority",
        "sglang.srt.managers.schedule_policy.SchedulePolicy", "sglang.srt.speculative.dspark_worker.DSparkWorker",
        "sglang.srt.layers.sampler.Sampler.forward", "sglang.srt.layers.logits_processor.LogitsProcessor",
        "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.match_prefix",
        "sglang.srt.mem_cache.registry.create_tree_cache", "sglang.srt.models.inkling2.Inkling2ForCausalLM",
        "sglang.srt.model_executor.cuda_graph_runner.CudaGraphRunner", "sglang.srt.model_executor.model_runner.ModelRunner",
        "sglang.srt.managers.scheduler.run_scheduler_process", "sglang.srt.disaggregation.decode.DecodePreallocQueue",
        "sglang",
    )):
        with pytest.raises(TargetResolutionError, match="node addresses must sit under"):
            catalog.resolve_manifest(load_manifest(_bundle(tmp_path / f"no{index}", slot)))


def test_intake_refuses_a_declared_domain_on_a_scheduler_entry(tmp_path):
    catalog = default_target_catalog()
    assert inspect_contribution(_bundle(tmp_path / "ok", "queue_order", f"{ADDER}.add_one_req"),
                                catalog=catalog).target_id == "scheduler"
    for index, extra in enumerate(('dtypes = ["bfloat16"]\n', 'architectures = ["sm_90"]\n', 'variant = "fast"\n',
                                   'prepare = "order"\n')):
        for slot in ("queue_order", f"{ADDER}.add_one_req"):
            with pytest.raises(EngineTreeError, match="declares no dtypes, architectures, metadata, prepare"):
                inspect_contribution(_bundle(tmp_path / f"no{index}{slot[:3]}", slot, extra=extra), catalog=catalog)
    with pytest.raises(EngineTreeError, match="'queue_order' takes no sub-addresses"):
        inspect_contribution(_bundle(tmp_path / "view", "queue_order.view"), catalog=catalog)


def test_scheduler_audit_covers_the_assembled_stack_and_a_decoy_never_narrows_it(tmp_path):
    decoy, nodes = "model.layers.0.self_attn", ("model.layers.*.mlp", "logits_processor")
    tree = SimpleNamespace(root=_bundle(tmp_path / "tree", *nodes, decoy, "tree_cache", "queue_order"))
    assert _audit_members(("queue_order",), tree) == tuple(sorted((*nodes, decoy, "tree_cache")))
    assert _audit_members((decoy, "queue_order"), tree) == tuple(sorted((*nodes, decoy, "tree_cache")))  # U17
    assert _audit_members((decoy,), tree) == (decoy,)  # a node submission keeps its own audit
    assert _audit_members(("queue_order",), SimpleNamespace(root=EXAMPLES / "miner_node_identity")) == (
        "model.layers.*.mlp",)
    with pytest.raises(B300RegisteredQualificationError, match="no model or cache slot"):
        _audit_members(("queue_order",), SimpleNamespace(root=EXAMPLES / "miner_prefill_scheduler"))


class _Req:
    """A waiting request; the replay's output length (``sampling_params``) must stay unread."""

    def __init__(self, prompt, cached, arrival):
        self.origin_input_ids, self.output_ids = [0] * prompt, []
        self.num_matched_prefix_tokens, self.arrival_processed_tokens = cached, arrival

    @property
    def sampling_params(self):
        raise AssertionError("the queue view read the output length")


def test_queue_order_view_hides_output_length_and_holds_the_answer_to_a_permutation(monkeypatch):
    class SchedulePolicy:  # SGLang PR #51926's signature: returns the ordered list, takes chunked_req
        def calc_priority(self, waiting, running_batch=None, processed_tokens=0, bootstrap_queue=(),
                          chunked_req=None):
            return list(waiting)

    module = ModuleType("sglang.srt.managers.schedule_policy")
    module.SchedulePolicy = SchedulePolicy
    monkeypatch.setitem(sys.modules, module.__name__, module)
    asked, answers = [], iter((lambda n: [n - 1, *range(n - 1)], lambda n: [0] * n))
    answer = next(answers)
    registry = KernelRegistry()
    registry.register(KernelImpl("queue_order", "test", lambda view: asked.append(view) or answer(len(view))))
    registry.enable()
    sglang_order.install(registry)
    sglang_order.install(registry)  # idempotent
    waiting = [_Req(4096, 1024, 0), _Req(512, 0, 300), _Req(64, 64, 900)]
    ordered = SchedulePolicy().calc_priority(waiting, None, processed_tokens=1000, bootstrap_queue=(),
                                             chunked_req=waiting[0])
    assert asked == [((3072, 1024, 1000), (512, 0, 700), (0, 64, 100))]
    assert ordered == [waiting[2], waiting[0], waiting[1]]
    answer = next(answers)
    with pytest.raises(ValueError, match="each of the 3 queue positions once"):
        SchedulePolicy().calc_priority(waiting, None, processed_tokens=1000)


def test_engine_binder_refuses_a_name_sglang_lacks_and_an_entry_nothing_selects(monkeypatch):
    class PrefillAdder:
        def add_one_req(self, req):
            return "stock"

    module = ModuleType("sglang.srt.managers.schedule_policy")
    module.__name__, module.PrefillAdder = "sglang.srt.managers.schedule_policy", PrefillAdder
    PrefillAdder.__module__ = module.__name__
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sglang_engine, "_bound", set())

    def registry(address, **declared):
        held = KernelRegistry()
        held.register(KernelImpl(address, "test", lambda stock: lambda self, req: ("candidate", stock(self, req)),
                                 **declared))
        held.enable()
        return held

    sglang_engine.install(registry(f"{ADDER}.add_one_req"))
    assert PrefillAdder().add_one_req(None) == ("candidate", "stock")
    with pytest.raises(LookupError, match="names no class or function the engine defines"):  # audit #14
        sglang_engine.install(registry(f"{ADDER}.add_one_reqq"))
    dtyped = registry(f"{ADDER}.should_defer", eligibility=eligibility_from_metadata(None, ("bfloat16",)))
    with pytest.raises(TypeError, match="no entry is selected"):
        sglang_engine.install(dtyped)


def test_eager_audit_seals_prefill_graphs_off_wherever_the_engine_defines_the_key(monkeypatch):
    for defined in (True, False):
        server_args = ModuleType("sglang.srt.server_args")
        server_args.ServerArgs = type("ServerArgs", (), {"cuda_graph_backend_prefill": None} if defined else {})
        monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
        kwargs = {"tp_size": 2}  # an Inkling profile that never named the key (U3)
        _eager_audit_engine(kwargs)
        assert kwargs == {"tp_size": 2, "disable_cuda_graph": True,
                          **({"cuda_graph_backend_prefill": "disabled"} if defined else {})}


def test_a_scheduler_change_parks_until_the_arena_crowns_a_model_or_cache(tmp_path):
    with _store(tmp_path) as store:
        rows = [_reserve_one(store, index=index) for index in range(3)]
        for row, target, member, marker in zip(rows, ("scheduler", "scheduler", "forward_pass"),
                                               ("queue_order", ADDER, "model.layers.*.mlp"), "abc"):
            _publish(store, row.reservation_id, _fingerprint(target, member, marker),
                     digest=marker * 64, root=f"/published/{marker}")
        parked = store.prepare_qualification_queue(service_digest=_h("service"))
        assert parked == tuple((row.reservation_id, "target_unavailable:scheduler") for row in rows[:2])
        assert [(store.get(row.reservation_id).status, store.get(row.reservation_id).decision) for row in rows] == [
            ("expired", "NO_DECISION"), ("expired", "NO_DECISION"), ("published", "")]
        store._db.execute("INSERT INTO target_lineage_tips VALUES('', 'forward_pass', ?, '', '1.1', ?, ?, ?, 1)",
                          ("d" * 64, _h("arena"), _h("stack"), _h("event")))
        later = _reserve_one(store, index=3)
        _publish(store, later.reservation_id, _fingerprint("scheduler", "queue_order", "d"),
                 digest="d" * 64, root="/published/d")
        assert store.prepare_qualification_queue(service_digest=_h("service")) == ()
        assert store.get(later.reservation_id).status == "published"
