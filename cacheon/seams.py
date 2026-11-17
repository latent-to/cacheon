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
``engine`` row serves any other object of the engine a bundle names by its dotted path.

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
    # Bind any class, method or function of the engine a bundle names by dotted path, at the scheduler's
    # process entry right after the bundle loads. No call is checked; the engine boundary judges the result.
    SeamAdapter("engine", "sglang.srt.managers.scheduler", "sglang_engine", "attr:Scheduler"),
)

# Roots of addresses that name an object of the engine instead of a module of the served model. The node
# binder skips them and SGLang serves them outside its CUDA graph. ``sglang`` roots a dotted path to any
# class, method or function of the engine; ``tree_cache`` takes no sub-address.
ENGINE_OBJECT = "sglang"
SCHEDULER_ADDRESSES = (ENGINE_OBJECT, "tree_cache")


def scheduler_address(slot: str) -> bool:
    return slot.split(".")[0] in SCHEDULER_ADDRESSES


def audited(slot: str) -> bool:
    """Whether the audit role counts calls for this address: a model node against stock, the cache by content.
    An engine object has no check per call."""

    return slot.split(".")[0] != ENGINE_OBJECT


def admit(tree, slot: str, entry: str, *, error: type[Exception]) -> None:
    """Check a scheduler address's entry in a parsed source, without importing the miner into intake."""

    import ast

    address = slot.split(".")[0]
    if (slot != address) != (address == ENGINE_OBJECT) or "*" in slot:
        raise error(f"{slot!r}: {address!r} takes " + ("a dotted path to one engine object"
                                                         if address == ENGINE_OBJECT else "no sub-addresses"))
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
