"""Executable contract for scripts/check_islands.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_islands.py"


def run_checker(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def write_tree(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


BASE_TREE = {
    "cacheon/__init__.py": "",
    "cacheon/capability_manifest.py": "CAPABILITY_MODULES: tuple[str, ...] = ()\n",
    "cacheon/cli.py": "import cacheon.wired\n",
    "cacheon/wired.py": "VALUE = 1\n",
}


def test_wired_modules_pass(tmp_path: Path) -> None:
    write_tree(tmp_path, BASE_TREE)
    result = run_checker(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 islands" in result.stdout


def test_island_outside_baseline_fails(tmp_path: Path) -> None:
    write_tree(tmp_path, BASE_TREE | {"cacheon/orphan.py": "VALUE = 2\n"})
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert "cacheon.orphan" in result.stderr
    assert "no production consumer" in result.stderr


def test_baseline_entry_passes_with_warning_when_reachable(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        BASE_TREE
        | {
            "cacheon/orphan.py": "VALUE = 2\n",
            "scripts/island_baseline.txt": "cacheon.orphan\ncacheon.wired\n",
        },
    )
    result = run_checker(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "baseline entry cacheon.wired is reachable or deleted" in result.stdout


def test_dotted_string_reference_counts_as_consumer(tmp_path: Path) -> None:
    tree = dict(BASE_TREE)
    tree["cacheon/cli.py"] = 'import cacheon.wired\nTARGET = "cacheon.dynamic"\n'
    tree["cacheon/dynamic.py"] = "VALUE = 3\n"
    write_tree(tmp_path, tree)
    result = run_checker(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_filename_reference_counts_as_consumer(tmp_path: Path) -> None:
    tree = dict(BASE_TREE)
    tree["cacheon/cli.py"] = 'import cacheon.wired\nPATCHER = "patch_tool.py"\n'
    tree["cacheon/patchers/__init__.py"] = ""
    tree["cacheon/patchers/patch_tool.py"] = "VALUE = 4\n"
    write_tree(tmp_path, tree)
    result = run_checker(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_capability_manifest_declares_private_consumer(tmp_path: Path) -> None:
    tree = dict(BASE_TREE)
    tree["cacheon/private_only.py"] = "VALUE = 5\n"
    tree["cacheon/capability_manifest.py"] = (
        'CAPABILITY_MODULES: tuple[str, ...] = ("cacheon.private_only",)\n'
    )
    write_tree(tmp_path, tree)
    result = run_checker(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_manifest_entry_fails(tmp_path: Path) -> None:
    tree = dict(BASE_TREE)
    tree["cacheon/capability_manifest.py"] = (
        'CAPABILITY_MODULES: tuple[str, ...] = ("cacheon.ghost",)\n'
    )
    write_tree(tmp_path, tree)
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert "unknown module: cacheon.ghost" in result.stderr


def test_part_suffix_test_file_warns(tmp_path: Path) -> None:
    write_tree(tmp_path, BASE_TREE | {"tests/test_thing_part2.py": "VALUE = 6\n"})
    result = run_checker(tmp_path)
    assert result.returncode == 0
    assert "_part suffix" in result.stdout


def test_repository_tree_is_clean() -> None:
    result = run_checker(Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stdout + result.stderr
