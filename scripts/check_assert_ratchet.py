"""Hold every test file to its recorded assertion floor.

The deletion program's test tranche rewrites arrangement and parametrizes
long-form tests; the invariant is that no conversion loses an assertion.
``scripts/assert_baseline.json`` records, per test file, the number of
``assert`` statements plus ``pytest.raises`` blocks. A file may only fall
below its floor when the same diff edits the baseline — which is how a
production casualty legitimately takes its whole test file with it, as a
reviewed line. New files and higher counts pass without ceremony.

Run: ``python scripts/check_assert_ratchet.py`` (CI runs it beside the LOC
ratchet). ``--write`` rewrites the baseline to current counts.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "assert_baseline.json"


def _is_raises(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    callee = node.func
    return (
        isinstance(callee, ast.Attribute) and callee.attr == "raises"
    ) or (isinstance(callee, ast.Name) and callee.id == "raises")


def file_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        total = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                total += 1
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                total += sum(1 for item in node.items if _is_raises(item.context_expr))
        counts[str(path.relative_to(ROOT))] = total
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the baseline")
    args = parser.parse_args()
    counts = file_counts()
    if args.write:
        BASELINE.write_text(
            json.dumps(counts, indent=0, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"assert_ratchet: baseline rewritten ({sum(counts.values())} assertions)")
        return 0
    floors: dict[str, int] = json.loads(BASELINE.read_text(encoding="utf-8"))
    failures = []
    for name, floor in sorted(floors.items()):
        count = counts.get(name)
        if count is None:
            failures.append(f"{name} was deleted but still has a floor of {floor}")
        elif count < floor:
            failures.append(f"{name} fell to {count} assertions, floor {floor}")
    for failure in failures:
        print(f"error: {failure}")
    if failures:
        print(
            "An assertion may only leave with its production casualty: edit "
            "scripts/assert_baseline.json in the same diff and justify it in review."
        )
    print(
        f"assert_ratchet: {sum(counts.values())} assertions in {len(counts)} files, "
        f"floor {sum(floors.values())}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
