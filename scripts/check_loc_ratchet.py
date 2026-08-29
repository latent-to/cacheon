"""Fail when tracked line counts grow past the reviewed ratchet.

The deletion program (chainops/DELETION_PROGRAM_2026-08-26.md, private) froze
accretion: tracked lines only move down. ``scripts/loc_baseline.txt`` holds
the reviewed ceiling per top-level directory. A change that must grow a
directory edits the baseline in the same diff — the baseline edit is the
justification, reviewed like any other line. A change that shrinks a
directory below its ceiling should lower the ceiling in the same diff so the
gain is locked in; this checker prints the exact replacement line to use.

Run: ``python scripts/check_loc_ratchet.py`` (CI runs it beside
``check_islands.py``). ``--write`` rewrites the baseline to current counts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "loc_baseline.txt"


def tracked_counts() -> dict[str, int]:
    names = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    counts: dict[str, int] = {}
    for name in names.split("\0"):
        if not name:
            continue
        top = name.split("/", 1)[0] if "/" in name else "(root)"
        try:
            data = (ROOT / name).read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        counts[top] = counts.get(top, 0) + data.count(b"\n")
    return counts


def read_baseline() -> dict[str, int]:
    ceilings: dict[str, int] = {}
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, value = line.rsplit(None, 1)
            ceilings[name] = int(value)
    return ceilings


def write_baseline(counts: dict[str, int]) -> None:
    lines = [
        "# Reviewed tracked-line ceilings per top-level directory.",
        "# Raising a ceiling is a reviewed decision made in the same diff",
        "# that needs it; lowering one locks in a deletion. check_loc_ratchet.py",
    ]
    lines += [f"{name} {counts[name]}" for name in sorted(counts)]
    BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the baseline")
    args = parser.parse_args()
    counts = tracked_counts()
    if args.write:
        write_baseline(counts)
        print(f"loc_ratchet: baseline rewritten ({sum(counts.values())} lines)")
        return 0
    ceilings = read_baseline()
    failures = []
    slack = []
    for name, count in sorted(counts.items()):
        ceiling = ceilings.get(name)
        if ceiling is None:
            failures.append(f"{name} ({count} lines) is not in the baseline")
        elif count > ceiling:
            failures.append(f"{name} grew to {count}, ceiling {ceiling} (+{count - ceiling})")
        elif count < ceiling:
            slack.append(f"{name} {count}")
    for failure in failures:
        print(f"error: {failure}")
    if failures:
        print(
            "Delete elsewhere in the same change, or raise the ceiling in "
            "scripts/loc_baseline.txt in this diff and justify it in review."
        )
    if slack:
        print("lock in deletions — lower these baseline lines to: " + "; ".join(slack))
    total = sum(counts.values())
    print(f"loc_ratchet: {total} tracked lines, ceiling {sum(ceilings.values())}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
