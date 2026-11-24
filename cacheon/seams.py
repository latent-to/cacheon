"""Single source of truth for the seam ADAPTERS (the sglang chokepoints we patch).

Three places must agree on the set of seam adapters: the ``.pth`` bootstrap (which
modules to watch for import), ``seam.activate()`` (which adapters to install), and the
``compat`` canary (which chokepoints to assert survived an sglang bump). Keeping three
hand-maintained lists in lockstep is exactly the drift sglang itself fought — a clean
registry PLUS a parallel hardcoded choices list turned "add one backend" into an
N-file edit (see the project review). So all three derive from the ONE table here:
adding a seam is a single entry, and the bootstrap watch-list, the install loop, and
the canary all pick it up.

No row names an operation of the model. Model code is served by the one ``nodes``
row, which binds whatever modules a bundle named; the per-operation rows it replaced
each pinned one sglang method and churned on every ``PINNED_SGLANG`` bump. The
``cache`` row serves the scheduler's prefix cache object and checks its content. The
``order`` row asks a bundle for the order of the waiting queue, and the ``engine`` row
serves the listed scheduler decisions (``ENGINE_OBJECTS``) a bundle names by dotted path.

Import-light on purpose (stdlib only): the ``.pth`` bootstrap imports this at
interpreter startup, before — and without — importing torch or sglang.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeamAdapter:
    name: str  # short label (also the integration module stem: cacheon.integrations.sglang_<name-ish>)
    target_module: str  # the sglang module whose import triggers installation
    integration: str  # the cacheon.integrations submodule exposing install(registry)
    # The patched chokepoint, for the compat canary + docs: "Class.method" for a method
    # patch, a bare "function_name" (no dot) for a module-LEVEL function rebind, or
    # "attr:Name" for a (possibly non-callable) module attribute the adapter rebinds.
    chokepoint: str


# THE table. Add a seam here and the bootstrap watch-list, the activate() install loop,
# and the compat canary all pick it up — no parallel list to keep in sync.
SEAM_ADAPTERS: tuple[SeamAdapter, ...] = (
    # NOT a slot seam: the candidate-bundle load gate. sglang spawns scheduler ranks
    # AND a detokenizer (output-path!) through the same bootstrap, and the detokenizer
    # imports watched modules too — so seam.activate() never loads miner code; this
    # adapter wraps the scheduler spawn entry so the load happens only in positively-
    # identified scheduler execution processes (active receipts == tp_size exactly).
    SeamAdapter("scheduler_gate", "sglang.srt.managers.scheduler",
                "sglang_scheduler_gate", "run_scheduler_process"),
    # The generic node binder. After the model loads it binds every registered slot
    # other than the cache to that module of the served model. Which addresses an
    # arena opens is decided at admission by the target catalog, not here.
    SeamAdapter("nodes", "sglang.srt.model_executor.model_runner",
                "sglang_nodes", "ModelRunner.load_model"),
    # Bind the scheduler's initialized cache after backend selection and wrappers,
    # preserving the pool and transfer-worker references created by that backend.
    SeamAdapter("cache", "sglang.srt.mem_cache.registry",
                "sglang_cache", "create_tree_cache"),
    # Ask the candidate for the order of the waiting queue after stock has ordered it; the answer is held to a
    # permutation of a view without output lengths and applied by stock's admission.
    SeamAdapter("order", "sglang.srt.managers.schedule_policy", "sglang_order", "SchedulePolicy.calc_priority"),
    # Bind each listed scheduler decision a bundle names by dotted path, at the scheduler's process entry
    # right after the bundle loads. No call is checked; the engine boundary judges the result.
    SeamAdapter("engine", "sglang.srt.managers.scheduler", "sglang_engine", "attr:Scheduler"),
)

# The scheduler decisions a ``scheduler`` bundle may replace by dotted path, each a class or a method where SGLang
# defines it: prefill admission and chunking, the token, row and KV budgets, the decode reserve, prefill/decode
# interleaving, and under KV pressure the decode memory check, retraction and its order. The queue's order is asked
# through ``queue_order``, whose view hides output lengths: the replay fixes them, so ``SchedulePolicy.calc_priority``
# or ``Scheduler.init_req_max_new_tokens`` would read the future. Nothing else is listed (audit #3, 2026-11-23):
# configuration publication, ``Scheduler.__init__`` and ``flush_cache``, speculative verification, the sampler and
# logits processor, cache classes and their handoffs (``tree_cache``), model and layer code (node addresses), the CUDA
# graph runner, ``ModelRunner`` and these seams' own chokepoints stay stock.
ENGINE_OBJECTS = (
    "sglang.srt.managers.schedule_batch.ScheduleBatch._get_decode_retraction_order",
    "sglang.srt.managers.schedule_batch.ScheduleBatch.check_decode_mem",
    "sglang.srt.managers.schedule_batch.ScheduleBatch.retract_decode",
    "sglang.srt.managers.schedule_policy.PrefillAdder",
    "sglang.srt.managers.schedule_policy.SchedulePolicy.prefill_token_budget",
    "sglang.srt.managers.schedule_policy.SchedulePolicy.shortest_prefill_chunk_limit",
    "sglang.srt.managers.scheduler.Scheduler._should_defer_prefill",
    "sglang.srt.managers.scheduler.Scheduler.get_num_allocatable_reqs",
    "sglang.srt.managers.scheduler_components.new_token_ratio_tracker.NewTokenRatioTracker",
    "sglang.srt.mem_cache.prefill_budget.ChainPrefillBudget",
    "sglang.srt.mem_cache.prefill_budget.PrefillBudget",
    "sglang.srt.mem_cache.prefill_budget.SWAPrefillBudget",
)
# Roots of addresses that name an object or question of the scheduler instead of a module of the served model.
# The node binder skips them and SGLang serves them outside its CUDA graph. ``sglang`` roots the listed objects;
# ``queue_order`` and ``tree_cache`` take no sub-address.
ENGINE_OBJECT = "sglang"
QUEUE_ORDER = "queue_order"
SCHEDULER_ADDRESSES = (QUEUE_ORDER, ENGINE_OBJECT, "tree_cache")


def scheduler_address(slot: str) -> bool:
    return slot.split(".")[0] in SCHEDULER_ADDRESSES


def audited(slot: str) -> bool:
    """Whether the audit role counts calls for this address: a model node against stock, the cache by content.
    An engine object and the queue order have no check per call."""

    return slot.split(".")[0] not in (ENGINE_OBJECT, QUEUE_ORDER)


def admit(tree, op, *, error: type[Exception]) -> None:
    """Check a scheduler address's entry in a parsed source, without importing the miner into intake."""

    import ast

    from cacheon.manifest import DEFAULT_VARIANT

    slot, entry = op.slot, op.entry
    address = slot.split(".")[0]
    if (slot != address) != (address == ENGINE_OBJECT) or "*" in slot:
        raise error(f"{slot!r}: {address!r} takes " + ("a dotted path to one engine object"
                                                         if address == ENGINE_OBJECT else "no sub-addresses"))
    # An engine entry is selected with no call descriptor, so a declared domain left it unbound and stock for a
    # whole lifetime while coverage counted it bound (audit #14).
    if address != "tree_cache" and (op.dtypes or op.architectures or op.metadata or op.prepare
                                    or op.variant != DEFAULT_VARIANT):
        raise error(f"{slot!r}: an engine entry declares no dtypes, architectures, metadata, prepare or variant")
    factory = next((node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == entry), None)
    if factory is None:
        raise error(f"{address} entry {entry!r} must be a function of one argument")
    args = factory.args
    positional = len(args.posonlyargs) + len(args.args)
    if (positional - len(args.defaults) > 1 or (positional < 1 and args.vararg is None)
            or any(default is None for default in args.kw_defaults)):
        raise error(f"{address} entry {entry!r} must be callable with one argument")

# The modules whose import should trigger seam installation (consumed by bootstrap).
TARGET_MODULES = frozenset(a.target_module for a in SEAM_ADAPTERS)
