"""CPU-only tests for the session worker's private runtime-cache gate."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import cacheon.eval.oci_session_worker as worker


@pytest.fixture()
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runtime-cache"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(worker, "CONTAINER_CACHE_PATH", str(root))
    monkeypatch.setattr(
        os.path, "ismount", lambda value: str(value) == str(root)
    )
    for name in worker._WRITABLE_RUNTIME_DIRECTORIES:
        monkeypatch.setenv(name, str(root / name.lower()))
    return root


def test_valid_private_cache_passes_and_leaves_no_probe_residue(
    cache_root: Path,
) -> None:
    worker._validate_private_cache()
    residue = tuple(cache_root.rglob(".preflight-*"))
    assert residue == ()
    for name in worker._WRITABLE_RUNTIME_DIRECTORIES:
        assert (cache_root / name.lower()).is_dir()


def test_cache_root_must_be_an_owned_0700_mount(
    cache_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root.chmod(0o755)
    with pytest.raises(
        worker.SessionWorkerError,
        match="not an owned writable 0700 mount",
    ):
        worker._validate_private_cache()
    cache_root.chmod(0o700)

    monkeypatch.setattr(os.path, "ismount", lambda value: False)
    with pytest.raises(
        worker.SessionWorkerError,
        match="not an owned writable 0700 mount",
    ):
        worker._validate_private_cache()


def test_runtime_directories_must_be_absolute_and_not_symlinks(
    cache_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_CACHE_PATH", "relative/cache")
    with pytest.raises(
        worker.SessionWorkerError,
        match="CUDA_CACHE_PATH is not one absolute runtime directory",
    ):
        worker._validate_private_cache()
    monkeypatch.setenv("CUDA_CACHE_PATH", str(cache_root / "cuda_cache_path"))

    monkeypatch.setenv("HF_HOME", "")
    with pytest.raises(
        worker.SessionWorkerError,
        match="HF_HOME is not one absolute runtime directory",
    ):
        worker._validate_private_cache()
    monkeypatch.setenv("HF_HOME", str(cache_root / "hf_home"))

    real = cache_root / "real-triton"
    real.mkdir(mode=0o700)
    link = cache_root / "triton-link"
    link.symlink_to(real)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(link))
    with pytest.raises(
        worker.SessionWorkerError,
        match="TRITON_CACHE_DIR is not one absolute runtime directory",
    ):
        worker._validate_private_cache()


def test_runtime_directories_may_not_escape_the_cache(
    cache_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "outside"))
    with pytest.raises(
        worker.SessionWorkerError,
        match="XDG_CACHE_HOME escapes the private runtime cache",
    ):
        worker._validate_private_cache()
    assert not (tmp_path / "outside").exists()


def test_unwritable_runtime_directory_fails_the_probe(
    cache_root: Path,
) -> None:
    home = cache_root / "home"
    home.mkdir(mode=0o500)
    try:
        with pytest.raises(
            worker.SessionWorkerError,
            match="HOME write/rename/delete probe failed",
        ):
            worker._validate_private_cache()
    finally:
        home.chmod(0o700)
