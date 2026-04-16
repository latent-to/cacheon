"""Phase 3 — Static sandbox for miner policy submissions.

Parses Python source with the `ast` module and rejects anything that
doesn't meet the safety allowlist before any code is executed.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "torch", "numpy", "math", "einops",
    "inference_engine",
})

# Full dotted paths miners may import from inference_engine.
# Everything else (runner, sandbox, harness, scoring, …) is off-limits
# because those modules re-export stdlib objects like os / subprocess.
_ALLOWED_IE_SUBMODULES: frozenset[str] = frozenset({
    "inference_engine",
    "inference_engine.policy",
})

BLOCKED_CALLS: frozenset[str] = frozenset({
    "eval", "exec", "compile", "open", "input", "breakpoint", "__import__",
    "getattr", "setattr", "delattr",
    "globals", "locals", "vars",
    "dir", "type", "super",
})

BLOCKED_ATTR_TARGETS: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "socket", "importlib",
    "ctypes", "cffi", "builtins", "pickle", "shelve",
})

REQUIRED_METHODS: frozenset[str] = frozenset({
    "setup", "write", "attend", "memory_bytes",
})


@dataclass
class CheckResult:
    ok: bool
    reason: str | None = None


def _top_level_module(name: str) -> str:
    return name.split(".")[0]


class _SafetyVisitor(ast.NodeVisitor):
    """Walk the AST and collect violation reasons."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = _top_level_module(alias.name)
            if top not in ALLOWED_IMPORTS:
                self.violations.append(
                    f"blocked import: '{alias.name}' "
                    f"(only {sorted(ALLOWED_IMPORTS)} allowed)"
                )
            elif top == "inference_engine" and alias.name not in _ALLOWED_IE_SUBMODULES:
                self.violations.append(
                    f"blocked import: '{alias.name}' "
                    f"(only {sorted(_ALLOWED_IE_SUBMODULES)} allowed)"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            self.violations.append(
                "relative imports are not allowed in policy submissions"
            )
        elif node.module is not None:
            top = _top_level_module(node.module)
            if top not in ALLOWED_IMPORTS:
                self.violations.append(
                    f"blocked from-import: 'from {node.module} import ...' "
                    f"(only {sorted(ALLOWED_IMPORTS)} allowed)"
                )
            elif top == "inference_engine" and node.module not in _ALLOWED_IE_SUBMODULES:
                self.violations.append(
                    f"blocked from-import: 'from {node.module} import ...' "
                    f"(only {sorted(_ALLOWED_IE_SUBMODULES)} allowed)"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node)
        if name in BLOCKED_CALLS:
            self.violations.append(f"blocked call: '{name}()'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        target = self._attr_target(node)
        if target in BLOCKED_ATTR_TARGETS:
            self.violations.append(
                f"blocked attribute access: '{target}.{node.attr}'"
            )
        self.generic_visit(node)

    @staticmethod
    def _call_name(node: ast.Call) -> str | None:
        """Return the callee name only for bare calls (e.g. eval(...)),
        not method calls (e.g. torch.compile(...)) — those are guarded
        by visit_Attribute + BLOCKED_ATTR_TARGETS instead."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        return None

    @staticmethod
    def _attr_target(node: ast.Attribute) -> str | None:
        if isinstance(node.value, ast.Name):
            return node.value.id
        return None


def _check_structure(tree: ast.Module) -> str | None:
    """Verify that exactly one KVCachePolicy subclass exists with all
    required methods defined."""

    policy_classes: list[ast.ClassDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name == "KVCachePolicy":
                policy_classes.append(node)
                break

    if len(policy_classes) == 0:
        return "no class subclassing KVCachePolicy found"
    if len(policy_classes) > 1:
        names = [c.name for c in policy_classes]
        return f"multiple KVCachePolicy subclasses found: {names}"

    cls = policy_classes[0]
    defined_methods = {
        node.name for node in ast.walk(cls)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    missing = REQUIRED_METHODS - defined_methods
    if missing:
        return f"KVCachePolicy subclass '{cls.name}' missing methods: {sorted(missing)}"

    return None


def check(source: str) -> CheckResult:
    """Parse and validate miner policy source. No execution."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return CheckResult(ok=False, reason=f"syntax error: {exc}")

    visitor = _SafetyVisitor()
    visitor.visit(tree)
    if visitor.violations:
        return CheckResult(ok=False, reason=visitor.violations[0])

    struct_err = _check_structure(tree)
    if struct_err is not None:
        return CheckResult(ok=False, reason=struct_err)

    return CheckResult(ok=True)
