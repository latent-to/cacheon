#!/usr/bin/env python3
"""Validate production-module reachability and code-volume hygiene.

Every module under ``cacheon/`` must be reachable from a real consumer: an
executable entry point, the CLI, bootstrap, a registered seam, a packaging
entry point, repository tooling under ``scripts/`` or ``examples/``, or the
declared capability manifest for digest-bound private callers
(``cacheon/capability_manifest.py``). Test imports are deliberately not
consumers: a module whose only reader is its own test suite is scaffolding.

Reachability follows three edge kinds from each module:

- static ``import``/``from`` statements;
- fully-qualified ``cacheon.*`` dotted-path string literals, which is how the
  seam registry, dependency-patch tiers, and ``python -m`` invocations refer
  to modules dynamically;
- unique ``<name>.py`` filename literals, which is how path-based subprocess
  invocations refer to modules.

Modules with no consumer are compared against ``scripts/island_baseline.txt``.
A module missing from the baseline fails the check; a baseline entry that has
become reachable or has been deleted is reported for removal. The goal is a
ratchet: shrinking the baseline is cleanup, growing it is a reviewed decision
visible in the diff.

The checker also reports (without failing) files above the physical-line
watermark and test files using a ``_part<N>`` suffix, both of which are
design smells under the repository's code-volume discipline in ``AGENTS.md``.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

DOTTED_REFERENCE = re.compile(r"[\"']((?:cacheon)(?:\.\w+)+)[\"']")
FILENAME_REFERENCE = re.compile(r"[\"'](\w+\.py)[\"']")
PART_SUFFIX = re.compile(r"_part\d+\.py\Z")
MAIN_GUARD = re.compile(r"if\s+__name__\s*==")

LINE_WATERMARK = 900

# Module families consumed through mechanisms a static scan cannot follow.
# Each entry names its wiring; do not extend this list without one.
ALLOWLIST_PREFIXES = (
    # sitecustomize injected into validator-owned OCI lifetimes.
    "cacheon.eval.oci_site",
    # Seam adapters registered by short stem in cacheon/seams.py rows.
    "cacheon.integrations",
)


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


class RepoGraph:
    """Import graph over the production package of one repository tree."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.modules: dict[str, Path] = {}
        for path in sorted((root / "cacheon").rglob("*.py")):
            rel = path.relative_to(root).with_suffix("")
            name = ".".join(rel.parts)
            if name.endswith(".__init__"):
                name = name.removesuffix(".__init__")
            self.modules[name] = path
        self.by_filename: dict[str, list[str]] = {}
        for name, path in self.modules.items():
            self.by_filename.setdefault(path.name, []).append(name)

    def resolve(self, dotted: str) -> str | None:
        while dotted:
            if dotted in self.modules:
                return dotted
            dotted = dotted.rpartition(".")[0]
        return None

    def _import_targets(self, tree: ast.Module, package_parts: list[str]) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (resolved := self.resolve(alias.name)) is not None:
                        found.add(resolved)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package_parts[: len(package_parts) - node.level + 1]
                    stem = ".".join(base + ([node.module] if node.module else []))
                else:
                    stem = node.module or ""
                if stem and (resolved := self.resolve(stem)) is not None:
                    found.add(resolved)
                for alias in node.names:
                    if stem and (resolved := self.resolve(f"{stem}.{alias.name}")) is not None:
                        found.add(resolved)
        return found

    def edges(self, name: str) -> set[str]:
        path = self.modules[name]
        tree = _parse(path)
        if tree is None:
            return set()
        package_parts = name.split(".")
        if path.name != "__init__.py":
            package_parts = package_parts[:-1]
        found = self._import_targets(tree, package_parts)
        text = path.read_text(encoding="utf-8", errors="replace")
        for dotted in DOTTED_REFERENCE.findall(text):
            if (resolved := self.resolve(dotted)) is not None:
                found.add(resolved)
        for filename in FILENAME_REFERENCE.findall(text):
            owners = self.by_filename.get(filename, [])
            if len(owners) == 1:
                found.add(owners[0])
        return found

    def external_import_roots(self) -> set[str]:
        roots: set[str] = set()
        # dashboard/ is the deployed operator app (run.sh: python -m
        # dashboard.app); its cacheon imports are production consumption.
        for directory in ("scripts", "examples", "dashboard"):
            base = self.root / directory
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*.py")):
                tree = _parse(path)
                if tree is not None:
                    roots |= self._import_targets(tree, [])
        return roots

    def pyproject_roots(self) -> set[str]:
        pyproject = self.root / "pyproject.toml"
        if not pyproject.exists():
            return set()
        roots: set[str] = set()
        for dotted in DOTTED_REFERENCE.findall(pyproject.read_text(encoding="utf-8")):
            if (resolved := self.resolve(dotted)) is not None:
                roots.add(resolved)
        return roots


def manifest_roots(graph: RepoGraph, problems: list[str]) -> set[str]:
    manifest = graph.modules.get("cacheon.capability_manifest")
    if manifest is None:
        problems.append("cacheon/capability_manifest.py is missing")
        return set()
    tree = _parse(manifest)
    declared: list[str] = []
    for node in ast.walk(tree) if tree is not None else ():
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        if "CAPABILITY_MODULES" not in targets or node.value is None:
            continue
        if not isinstance(node.value, (ast.Tuple, ast.List)):
            problems.append("CAPABILITY_MODULES must be a literal tuple or list")
            return set()
        for element in node.value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                declared.append(element.value)
            else:
                problems.append("CAPABILITY_MODULES entries must be string literals")
    roots: set[str] = set()
    for dotted in declared:
        if dotted in graph.modules:
            roots.add(dotted)
        else:
            problems.append(f"capability manifest names unknown module: {dotted}")
    return roots


def collect_roots(graph: RepoGraph, problems: list[str]) -> set[str]:
    roots: set[str] = set()
    # capability_manifest is consumed by this checker itself.
    for fixed in ("cacheon", "cacheon.cli", "cacheon.bootstrap", "cacheon.capability_manifest"):
        if fixed in graph.modules:
            roots.add(fixed)
    for name, path in graph.modules.items():
        if any(name == p or name.startswith(p + ".") for p in ALLOWLIST_PREFIXES):
            roots.add(name)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "__main__" in text and MAIN_GUARD.search(text):
            roots.add(name)
    roots |= graph.pyproject_roots()
    roots |= graph.external_import_roots()
    roots |= manifest_roots(graph, problems)
    return roots


def reachable_from(graph: RepoGraph, roots: set[str]) -> set[str]:
    seen: set[str] = set()
    stack = sorted(roots)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        # Importing a submodule executes every ancestor package __init__.
        parent = current.rpartition(".")[0]
        while parent and parent in graph.modules and parent not in seen:
            seen.add(parent)
            stack.append(parent)
            parent = parent.rpartition(".")[0]
        stack.extend(graph.edges(current))
    return seen


def read_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    entries: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.add(line)
    return entries


def volume_warnings(root: Path, verbose: bool) -> list[str]:
    oversized: list[tuple[int, Path]] = []
    for directory in ("cacheon", "tests"):
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root)
            lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
            if lines > LINE_WATERMARK:
                oversized.append((lines, rel))
    warnings: list[str] = []
    if oversized:
        oversized.sort(reverse=True)
        if verbose:
            warnings.extend(
                f"{rel} has {lines} lines (watermark {LINE_WATERMARK})"
                for lines, rel in oversized
            )
        else:
            largest, rel = oversized[0]
            warnings.append(
                f"{len(oversized)} files exceed {LINE_WATERMARK} lines "
                f"(largest: {rel} at {largest}); rerun with --verbose for the list"
            )
    tests = root / "tests"
    if tests.is_dir():
        for path in sorted(tests.rglob("*.py")):
            if PART_SUFFIX.search(path.name):
                warnings.append(
                    f"{path.relative_to(root)} uses a _part suffix; split by behavior instead"
                )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to scan (defaults to this repository)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="list every file above the line watermark instead of a summary",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    problems: list[str] = []
    graph = RepoGraph(root)
    if not graph.modules:
        print(f"error: no cacheon package under {root}", file=sys.stderr)
        return 2
    roots = collect_roots(graph, problems)
    reached = reachable_from(graph, roots)
    islands = sorted(set(graph.modules) - reached)
    baseline_path = root / "scripts" / "island_baseline.txt"
    baseline = read_baseline(baseline_path)

    new_islands = [name for name in islands if name not in baseline]
    for name in new_islands:
        lines = len(graph.modules[name].read_text(encoding="utf-8").splitlines())
        problems.append(
            f"{name} ({lines} lines) has no production consumer and is not in "
            f"{baseline_path.relative_to(root)}"
        )

    warnings = volume_warnings(root, args.verbose)
    for entry in sorted(baseline):
        if entry not in islands:
            warnings.append(
                f"baseline entry {entry} is reachable or deleted; remove it from "
                f"{baseline_path.relative_to(root)}"
            )

    print(
        f"check_islands: {len(graph.modules)} modules, {len(roots)} roots, "
        f"{len(islands)} islands ({len(new_islands)} outside baseline)"
    )
    for warning in warnings:
        print(f"warning: {warning}")
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        print(
            "\nWire each module into its production owner, declare it in "
            "cacheon/capability_manifest.py, or delete it. Growing "
            "scripts/island_baseline.txt is a reviewed last resort.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
