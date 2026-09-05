"""Fail when tracked line counts grow past the reviewed ratchets.

Two ratchets, one script, one CI step:

``scripts/loc_baseline.txt`` holds the reviewed ceiling per top-level
directory. A change that must grow a directory edits the baseline in the
same diff — the baseline edit is the justification, reviewed like any other
line. A change that shrinks a directory should lower the ceiling in the same
diff so the gain is locked in.

``scripts/file_ceilings.txt`` holds the reviewed ceiling for every tracked
Python file at or above ``OVERSIZED`` lines. A listed file may only shrink.
An unlisted file that reaches ``OVERSIZED`` lines must be added to the
ceilings in the same diff and justified in review, or split first. This is
what keeps the files already over the AGENTS.md cap from growing by even one
line while their decomposition waits.

Run: ``python scripts/check_loc_ratchet.py`` (CI runs it beside
``check_islands.py``). ``--write`` rewrites both baselines to current counts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "loc_baseline.txt"
FILE_BASELINE = ROOT / "scripts" / "file_ceilings.txt"
OVERSIZED = 900


def _tracked_line_counts() -> dict[str, int]:
    names = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    counts: dict[str, int] = {}
    for name in names.split("\0"):
        if not name:
            continue
        try:
            data = (ROOT / name).read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        counts[name] = data.count(b"\n")
    return counts


def directory_counts(files: dict[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, lines in files.items():
        top = name.split("/", 1)[0] if "/" in name else "(root)"
        counts[top] = counts.get(top, 0) + lines
    return counts


def oversized_files(files: dict[str, int]) -> dict[str, int]:
    return {
        name: lines
        for name, lines in files.items()
        if name.endswith(".py") and lines >= OVERSIZED
    }


def read_baseline(path: Path) -> dict[str, int]:
    ceilings: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, value = line.rsplit(None, 1)
            ceilings[name] = int(value)
    return ceilings


def write_baseline(path: Path, header: list[str], counts: dict[str, int]) -> None:
    lines = header + [f"{name} {counts[name]}" for name in sorted(counts)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


DIRECTORY_HEADER = [
    "# Reviewed tracked-line ceilings per top-level directory.",
    "# Raising a ceiling is a reviewed decision made in the same diff",
    "# that needs it; lowering one locks in a deletion. check_loc_ratchet.py",
]
FILE_HEADER = [
    f"# Reviewed line ceilings for every tracked Python file at or above {OVERSIZED} lines.",
    "# A listed file may only shrink. An unlisted file that reaches the band must be",
    "# added here in the same diff and justified in review, or split. check_loc_ratchet.py",
]


def _check(
    label: str, counts: dict[str, int], ceilings: dict[str, int], *, unlisted_ok: bool
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    slack: list[str] = []
    for name, count in sorted(counts.items()):
        ceiling = ceilings.get(name)
        if ceiling is None:
            if not unlisted_ok:
                failures.append(f"{label} {name} ({count} lines) is not in the baseline")
        elif count > ceiling:
            failures.append(
                f"{label} {name} grew to {count}, ceiling {ceiling} (+{count - ceiling})"
            )
        elif count < ceiling:
            slack.append(f"{name} {count}")
    return failures, slack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite both baselines")
    args = parser.parse_args()
    files = _tracked_line_counts()
    directories = directory_counts(files)
    oversized = oversized_files(files)
    if args.write:
        write_baseline(BASELINE, DIRECTORY_HEADER, directories)
        write_baseline(FILE_BASELINE, FILE_HEADER, oversized)
        print(
            f"loc_ratchet: baselines rewritten ({sum(files.values())} lines, "
            f"{len(oversized)} oversized files)"
        )
        return 0
    dir_failures, dir_slack = _check(
        "directory", directories, read_baseline(BASELINE), unlisted_ok=False
    )
    file_failures, file_slack = _check(
        "file", oversized, read_baseline(FILE_BASELINE), unlisted_ok=False
    )
    for failure in dir_failures + file_failures:
        print(f"error: {failure}")
    if dir_failures:
        print(
            "Delete elsewhere in the same change, or raise the ceiling in "
            "scripts/loc_baseline.txt in this diff and justify it in review."
        )
    if file_failures:
        print(
            f"An oversized file may only shrink. Move new behavior into a focused "
            f"module under {OVERSIZED} lines, or edit scripts/file_ceilings.txt in "
            "this diff and justify it in review."
        )
    if dir_slack:
        print("lock in deletions — lower these loc_baseline.txt lines to: " + "; ".join(dir_slack))
    if file_slack:
        print("lock in shrinkage — lower these file_ceilings.txt lines to: " + "; ".join(file_slack))
    dropped = sorted(name for name in read_baseline(FILE_BASELINE) if name not in oversized)
    if dropped:
        print("no longer oversized — remove from file_ceilings.txt: " + ", ".join(dropped))
    total = sum(files.values())
    print(
        f"loc_ratchet: {total} tracked lines, ceiling {sum(read_baseline(BASELINE).values())}; "
        f"{len(oversized)} files at or above {OVERSIZED} lines"
    )
    return 1 if dir_failures or file_failures else 0


if __name__ == "__main__":
    sys.exit(main())
