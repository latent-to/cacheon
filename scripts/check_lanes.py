#!/usr/bin/env python3
"""Fail a pull request that modifies live-lane paths from outside that lane.

``scripts/live_lane_paths.txt`` names the repository paths under active
modification on the live operations lane, together with the branch patterns
that lane works on (``branch <pattern>`` lines). A change built on any other
branch that touches those paths will collide with the live lane's merge, so
this check fails it early with the two legitimate remedies:

- move the work to a live-lane branch, or
- transfer the path out of the live lane by deleting its entry in the same
  pull request, making the ownership change a reviewable line in the diff.

The check compares the merge-base diff against the base ref, so entries
deleted by the pull request itself no longer protect their paths. Branches
matching a declared live-lane pattern always pass. Without a resolvable base
ref the check reports and exits zero; it is a pull-request gate, not a push
gate.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path


def read_lane_file(path: Path) -> tuple[list[str], list[str]]:
    """Return (live-lane branch patterns, protected path patterns)."""

    if not path.is_file():
        raise SystemExit(f"error: lane file {path} is missing")
    branch_patterns: list[str] = []
    path_patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("branch "):
            branch_patterns.append(line[len("branch "):].strip())
        else:
            path_patterns.append(line)
    return branch_patterns, path_patterns


def git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"error: git {' '.join(args)} failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line]


def current_branch() -> str:
    return git_lines("rev-parse", "--abbrev-ref", "HEAD")[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="origin/main",
        help="base ref to diff against (merge-base semantics)",
    )
    parser.add_argument(
        "--branch",
        default="",
        help="branch name under review (default: the checked-out branch)",
    )
    parser.add_argument(
        "--paths-file",
        default=str(Path(__file__).resolve().parent / "live_lane_paths.txt"),
    )
    args = parser.parse_args()

    branch_patterns, path_patterns = read_lane_file(Path(args.paths_file))

    branch = args.branch or current_branch()
    if any(fnmatch.fnmatch(branch, pattern) for pattern in branch_patterns):
        print(f"check_lanes: branch {branch!r} is the live lane; not checked")
        return 0

    base = args.base.strip()
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if not base or base == "origin/" or probe.returncode != 0:
        print(f"check_lanes: base ref {base!r} is not resolvable; not checked")
        return 0

    changed = git_lines("diff", "--name-only", f"{base}...HEAD")

    violations = sorted(
        path
        for path in changed
        if any(fnmatch.fnmatch(path, pattern) for pattern in path_patterns)
    )
    print(
        f"check_lanes: branch {branch!r} vs {base}: "
        f"{len(changed)} changed paths, {len(violations)} in the live lane"
    )
    if not violations:
        return 0
    for path in violations:
        print(f"error: {path} is a live-lane path")
    print(
        "Move this work to a live-lane branch (declared in "
        "scripts/live_lane_paths.txt), or transfer ownership by deleting the "
        "matching path entry there in this same change."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
