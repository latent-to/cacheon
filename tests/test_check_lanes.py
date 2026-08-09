"""Executable contract for scripts/check_lanes.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_lanes.py"


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )


def run_checker(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "live_lane_paths.txt").write_text(
        "# comment\nbranch live/*\ncacheon/live_module.py\ncacheon/family_*\n",
        encoding="utf-8",
    )
    (root / "cacheon").mkdir()
    (root / "cacheon" / "live_module.py").write_text("A = 1\n", encoding="utf-8")
    (root / "cacheon" / "family_one.py").write_text("B = 1\n", encoding="utf-8")
    (root / "cacheon" / "free_module.py").write_text("C = 1\n", encoding="utf-8")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


def paths_file(root: Path) -> str:
    return str(root / "scripts" / "live_lane_paths.txt")


def test_free_path_change_passes(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    git(root, "switch", "-q", "-c", "ops/tidy")
    (root / "cacheon" / "free_module.py").write_text("C = 2\n", encoding="utf-8")
    git(root, "commit", "-q", "-am", "free change")
    result = run_checker(root, "--base", "main", "--paths-file", paths_file(root))
    assert result.returncode == 0, result.stdout + result.stderr


def test_live_path_change_fails_with_remedies(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    git(root, "switch", "-q", "-c", "ops/tidy")
    (root / "cacheon" / "live_module.py").write_text("A = 2\n", encoding="utf-8")
    (root / "cacheon" / "family_two.py").write_text("D = 1\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "live change")
    result = run_checker(root, "--base", "main", "--paths-file", paths_file(root))
    assert result.returncode == 1
    assert "cacheon/live_module.py is a live-lane path" in result.stdout
    assert "cacheon/family_two.py is a live-lane path" in result.stdout
    assert "live-lane branch" in result.stdout


def test_live_lane_branch_is_exempt(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    git(root, "switch", "-q", "-c", "live/launch-work")
    (root / "cacheon" / "live_module.py").write_text("A = 2\n", encoding="utf-8")
    git(root, "commit", "-q", "-am", "live change")
    result = run_checker(root, "--base", "main", "--paths-file", paths_file(root))
    assert result.returncode == 0
    assert "live lane" in result.stdout


def test_deleting_the_entry_transfers_ownership(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    git(root, "switch", "-q", "-c", "ops/transfer")
    (root / "cacheon" / "live_module.py").write_text("A = 2\n", encoding="utf-8")
    (root / "scripts" / "live_lane_paths.txt").write_text(
        "# comment\ncacheon/family_*\n", encoding="utf-8"
    )
    git(root, "commit", "-q", "-am", "transfer ownership")
    result = run_checker(root, "--base", "main", "--paths-file", paths_file(root))
    assert result.returncode == 0, result.stdout + result.stderr


def test_unresolvable_base_is_not_checked(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    git(root, "switch", "-q", "-c", "ops/tidy")
    result = run_checker(
        root, "--base", "origin/absent", "--paths-file", paths_file(root)
    )
    assert result.returncode == 0
    assert "not resolvable" in result.stdout
