from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "optima" / "vendor_provenance.json"

EXPECTED_ASSETS = (
    "optima/arena_assets/minimax_m3/sglang_patch/flashinfer_trtllm.py",
    "optima/arena_assets/minimax_m3/sglang_patch/modelopt_quant.py",
    "optima/eval/seccomp_moby_v0_2_1.json",
)

EXPECTED_LICENSE_HASHES = {
    "LICENSES/MINIMAX_COMMUNITY_LICENSE.txt": (
        "b53f2fdda3049b0e9013207be51efc2d372cda1fcfdd8bb4bb8b22658ca5db9c"
    ),
    "LICENSES/SGLANG.txt": (
        "1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747"
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load() -> dict[str, object]:
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


def _assets() -> list[dict[str, object]]:
    return list(_load()["assets"])


def _wheel_license_member(names: set[str], relative: str) -> str:
    suffixes = {
        f".dist-info/{PurePosixPath(relative).name}",
        f".dist-info/{relative}",
        f".dist-info/licenses/{relative}",
    }
    matches = [name for name in names if name.endswith(tuple(suffixes))]
    assert len(matches) == 1
    return matches[0]


def test_vendor_manifest_is_canonical_and_closed() -> None:
    data = _load()
    expected = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    assert PROVENANCE.read_text(encoding="utf-8") == expected
    assert set(data) == {"assets", "schema_version"}
    assert data["schema_version"] == 1

    paths = tuple(asset["packaged"]["path"] for asset in data["assets"])
    assert paths == EXPECTED_ASSETS


def test_packaged_vendor_assets_are_safe_regular_hash_bound_files() -> None:
    for asset in _assets():
        packaged = asset["packaged"]
        relative = PurePosixPath(packaged["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts

        path = ROOT.joinpath(*relative.parts)
        mode = path.lstat().st_mode
        assert stat.S_ISREG(mode)
        assert not path.is_symlink()

        raw = path.read_bytes()
        assert len(raw) == packaged["size"]
        assert _sha256(raw) == packaged["sha256"]

        distribution = asset["license"]["distribution"]
        license_path = ROOT / distribution["file"]
        assert license_path.is_file()
        assert not license_path.is_symlink()


def test_provenance_uses_immutable_https_sources() -> None:
    for asset in _assets():
        source = asset["source_distribution"]
        assert source["repository"].startswith("https://")
        assert len(source["revision"]) == 40
        int(source["revision"], 16)
        assert len(source["sha256"]) == 64
        int(source["sha256"], 16)

        for key in ("derivation_base", "runtime_target"):
            if key not in asset:
                continue
            row = asset[key]
            assert row["repository"].startswith("https://")
            assert len(row["revision"]) == 40
            int(row["revision"], 16)
            assert len(row["sha256"]) == 64
            int(row["sha256"], 16)


def test_moby_profile_declares_its_only_transformation() -> None:
    asset = _assets()[2]
    packaged = asset["packaged"]
    source = asset["source_distribution"]
    raw = (ROOT / packaged["path"]).read_bytes()

    assert source["tag"] == "seccomp/v0.2.1"
    assert source["size"] == len(raw) - 1
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert _sha256(raw[:-1]) == source["sha256"]
    assert asset["transformations"] == [
        {
            "input_sha256": source["sha256"],
            "name": "append-terminal-lf-v1",
            "output_sha256": packaged["sha256"],
        }
    ]

    profile = json.loads(raw)
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    assert isinstance(profile["syscalls"], list) and profile["syscalls"]


def test_overlay_distribution_and_runtime_preimage_are_not_conflated() -> None:
    flashinfer, modelopt = _assets()[:2]

    for asset in (flashinfer, modelopt):
        assert asset["transformations"] == []
        assert asset["packaged"]["sha256"] == asset["source_distribution"]["sha256"]
        assert asset["packaged"]["size"] == asset["source_distribution"]["size"]
        assert asset["derivation_base"]["insertions"] > 0
        assert asset["derivation_base"]["deletions"] == 0
        assert asset["runtime_target"]["package_version"] == "0.0.0.dev1+g56e290315"

    assert flashinfer["derivation_base"]["revision"] != flashinfer["runtime_target"]["revision"]
    assert flashinfer["packaged"]["sha256"] != flashinfer["runtime_target"]["sha256"]
    assert modelopt["derivation_base"]["sha256"] == modelopt["runtime_target"]["sha256"]


def test_third_party_license_copies_are_hash_bound() -> None:
    for relative, expected_hash in EXPECTED_LICENSE_HASHES.items():
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink()
        assert _sha256(path.read_bytes()) == expected_hash


def test_clean_wheel_and_sdist_include_vendor_record(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for relative in ("LICENSE", "NOTICE", "README.md", "pyproject.toml"):
        shutil.copy2(ROOT / relative, source / relative)
    shutil.copytree(ROOT / "LICENSES", source / "LICENSES")
    shutil.copytree(ROOT / "optima", source / "optima")
    shutil.copytree(ROOT / "optima_kernels", source / "optima_kernels")

    output = source / "dist"
    output.mkdir()
    env = {
        **os.environ,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = (
        "from setuptools import build_meta; "
        "build_meta.build_sdist('dist'); build_meta.build_wheel('dist')"
    )
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=source,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(output.glob("*.whl"))
    sdist = next(output.glob("*.tar.gz"))
    required_package_files = {
        *EXPECTED_ASSETS,
        "optima/vendor_provenance.json",
        "optima_kernels/collective/fused_ar_rmsnorm_sm103.cu",
    }

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert required_package_files <= names
        for relative in ("LICENSE", "NOTICE", *EXPECTED_LICENSE_HASHES):
            member = _wheel_license_member(names, relative)
            assert archive.read(member) == (ROOT / relative).read_bytes()

    with tarfile.open(sdist, "r:gz") as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
        assert required_package_files <= names
        assert {"LICENSE", "NOTICE", *EXPECTED_LICENSE_HASHES} <= names
