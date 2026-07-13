"""Validator-owned allowlist for GPU-kernel DSL tracing-JIT compile entrypoints.

Single source of truth for the ONE carve-out in the engine-tree dynamic-import
policy (optima/engine_tree.py): a call like ``cute.compile(<@cute.jit fn>, ...)``.

WHY A CARVE-OUT AT ALL.  The engine-tree policy bans ``compile``/``eval``/
``exec``/``__import__`` (bare and as attribute calls) because each is a
string-to-code path that defeats the static inspectability the whole intake
pipeline rests on: every executable byte a candidate can reach must already be
in the scanned source tree.  A DSL *tracing JIT* entrypoint is different in
kind, not degree.  ``cutlass.cute.compile`` consumes a Python *callable object*
(the ``@cute.jit``-decorated kernel, whose body is in the already-scanned
bundle source) and lowers it to PTX; there is no ``source_string -> code``
step, so the inspectability property is preserved.  Triton needs no entry here
at all — ``@triton.jit`` compiles lazily on first launch, with no explicit
``compile`` call to admit.

The admission is deliberately narrow and fail-closed (see ``admitted_receivers``
/ ``is_admitted_call``): the receiver name must be bound EXACTLY ONCE in the
whole module, by a plain absolute import alias of a table module, that does not
resolve inside the contribution tree (a vendored ``cutlass/`` withdraws it — the
compile must reach the trusted pinned DSL, never bundle code); and the call's
first positional argument must not be a string/bytes literal (a tracing JIT is
handed an object, never source text).  Any other binding of the name anywhere
in the file — assignment, parameter, def/class, loop/with/except target, del,
match capture, a second import — withdraws the admission.

Adding a DSL is one row in ``DSL_JIT_ENTRYPOINTS``.  Import-light on purpose
(stdlib only): both the engine-tree materializer and the AST sandbox import this
without pulling torch/sglang.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DslJitEntrypoint:
    """One allowlisted ``<module>.<attr>(<callable>, ...)`` tracing-JIT entry.

    ``module`` is the fully-qualified, ABSOLUTE import path of the trusted DSL
    module (must be provided by the pinned base engine, never by the bundle).
    ``attr`` is the compile method name on that module.
    """

    module: str
    attr: str
    note: str = ""


# THE table.  One row per trusted DSL tracing-JIT compile entrypoint.
DSL_JIT_ENTRYPOINTS: tuple[DslJitEntrypoint, ...] = (
    DslJitEntrypoint(
        module="cutlass.cute",
        attr="compile",
        note="CuTe DSL: cute.compile(<@cute.jit callable>, *args) -> compiled kernel.",
    ),
)

# attr names that a table entry may admit — used to keep the engine-tree ban and
# this carve-out talking about the same set (only 'compile' today).
ADMITTED_ATTRS: frozenset[str] = frozenset(e.attr for e in DSL_JIT_ENTRYPOINTS)

# Module dotted-paths, for quick membership tests during alias analysis.
_ENTRY_BY_MODULE: dict[str, DslJitEntrypoint] = {e.module: e for e in DSL_JIT_ENTRYPOINTS}


def _all_bound_names(tree: ast.AST) -> dict[str, int]:
    """Count every binding of every name in the module, across all scopes.

    Fail-closed by construction: any construct that could rebind or shadow a
    name is counted, so a receiver admitted below is provably bound exactly
    once and only by the import alias we inspected.
    """

    counts: dict[str, int] = {}

    def bind(name: str | None) -> None:
        if name:
            counts[name] = counts.get(name, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bind(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(node.name)
        elif isinstance(node, ast.arg):
            bind(node.arg)
        elif isinstance(node, ast.ExceptHandler):
            bind(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bind(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bind(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bind(alias.asname or alias.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            bind(node.name)
        elif isinstance(node, ast.MatchMapping):
            bind(node.rest)
    return counts


def _import_alias_modules(tree: ast.AST) -> dict[str, str]:
    """Map name -> absolute dotted module for plain import aliases of table modules.

    Only two spellings resolve a table module to a single Name receiver:
      * ``import <module> as <name>``           (asname REQUIRED — bare
        ``import a.b`` binds ``a``, not ``a.b``, so it is not a receiver)
      * ``from <pkg> import <leaf> [as <name>]`` where ``pkg.leaf`` is a table
        module and the import is absolute (``level == 0``)
    Relative imports (``from . import x``) never name an external trusted DSL.
    """

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and alias.name in _ENTRY_BY_MODULE:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue
            for alias in node.names:
                dotted = f"{node.module}.{alias.name}"
                if dotted in _ENTRY_BY_MODULE:
                    aliases[alias.asname or alias.name] = dotted
    return aliases


def admitted_receivers(
    tree: ast.AST,
    *,
    module_resolves_locally: Callable[[tuple[str, ...]], bool] | None = None,
) -> dict[str, DslJitEntrypoint]:
    """Names usable as ``<name>.<attr>(...)`` DSL-JIT receivers (fail-closed).

    A name qualifies iff its SINGLE binding in the whole module is a plain
    absolute import alias of a table module, and — when a resolver is supplied —
    that module does not resolve inside the contribution tree.  Without a
    resolver (e.g. the single-file sandbox has no tree context) the local-shadow
    check is simply skipped; callers that own a tree MUST pass the resolver.
    """

    counts = _all_bound_names(tree)
    aliases = _import_alias_modules(tree)
    admitted: dict[str, DslJitEntrypoint] = {}
    for name, module in aliases.items():
        if counts.get(name, 0) != 1:
            continue  # rebound / shadowed somewhere -> withdraw
        if module_resolves_locally is not None:
            parts = tuple(module.split("."))
            if any(
                module_resolves_locally(parts[:end]) for end in range(1, len(parts) + 1)
            ):
                continue  # a bundle-local module could shadow the trusted DSL
        admitted[name] = _ENTRY_BY_MODULE[module]
    return admitted


def _first_positional(node: ast.Call) -> ast.expr | None:
    for arg in node.args:
        if isinstance(arg, ast.Starred):
            return None  # *args -> first real positional is opaque; treat as absent
        return arg
    return None


def is_admitted_call(
    node: ast.Call, receivers: dict[str, DslJitEntrypoint]
) -> bool:
    """True iff ``node`` is an admitted ``<receiver>.<attr>(<non-string>, ...)`` call.

    Requires: an attribute call whose base is a bare Name in ``receivers``,
    whose attribute equals that entry's ``attr``, and whose first positional
    argument (if any) is not a string/bytes literal — a tracing JIT is handed a
    callable object, never source text.
    """

    func = node.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return False
    entry = receivers.get(func.value.id)
    if entry is None or func.attr != entry.attr:
        return False
    first = _first_positional(node)
    if isinstance(first, ast.Constant) and isinstance(first.value, (str, bytes)):
        return False
    return True
