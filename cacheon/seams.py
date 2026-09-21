"""Single source of truth for the seam ADAPTERS (the sglang chokepoints we patch).

Three places must agree on the set of seam adapters: the ``.pth`` bootstrap (which
modules to watch for import), ``seam.activate()`` (which adapters to install), and the
``compat`` canary (which chokepoints to assert survived an sglang bump). Keeping three
hand-maintained lists in lockstep is exactly the drift sglang itself fought — a clean
registry PLUS a parallel hardcoded choices list turned "add one backend" into an
N-file edit (see the project review). So all three derive from the ONE table here:
adding a seam is a single entry, and the bootstrap watch-list, the install loop, and
the canary all pick it up.

No row names an operation of the model. Candidate code is served by the one ``nodes``
row, which binds whatever modules a bundle named; the per-operation rows it replaced
each pinned one sglang method and churned on every ``PINNED_SGLANG`` bump.

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
    # NOT a slot seam: the resident-SCREEN-tier hot-swap hook. Inert unless the
    # validator sets CACHEON_RESIDENT_SWAP (a control directory) — which only the
    # persistent screening engine does, never qualification/crown launches. BEFORE
    # hook on decode-graph (re)capture: applies a pending bundle swap in-process so
    # the recapture warmup JIT-compiles the new kernel and the recorded graphs bake
    # it in. See cacheon/integrations/sglang_resident_swap.py.
    SeamAdapter("resident_swap", "sglang.srt.model_executor.model_runner",
                "sglang_resident_swap", "ModelRunner.init_decode_cuda_graph"),
    # The generic node binder. After the model loads it binds every registered slot
    # that is a node address (a name outside cacheon.slots) to that module of the
    # served model. Which addresses an arena opens is decided at admission by the
    # target catalog, not here.
    SeamAdapter("nodes", "sglang.srt.model_executor.model_runner",
                "sglang_nodes", "ModelRunner.load_model"),
)

# The modules whose import should trigger seam installation (consumed by bootstrap).
TARGET_MODULES = frozenset(a.target_module for a in SEAM_ADAPTERS)
