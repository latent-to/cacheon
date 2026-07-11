from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from optima.arenas import ARENAS, verify_model_content_seal
from optima.model_provision import (
    MODEL_SEAL_NAME,
    ModelProvisionError,
    generate_model_content_seal,
    install_runtime_overlay_assets,
    main,
)
from optima.runtime_overlay import RuntimeFileOverlay, verify_runtime_overlays


def _digest(rows: dict[str, bytes]) -> str:
    payload = "".join(
        f"{path}\0{hashlib.sha256(value).hexdigest()}\n"
        for path, value in sorted(rows.items())
    ).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_model_seal_is_deterministic_and_compatible_with_arena_verifier(tmp_path):
    model = tmp_path / "model"
    (model / "weights").mkdir(parents=True)
    (model / "config.json").write_bytes(b'{"model":"test"}\n')
    (model / "weights" / "b.bin").write_bytes(b"bbb")
    (model / "weights" / "a.bin").write_bytes(b"aaa")
    (model / ".cache" / "huggingface").mkdir(parents=True)
    (model / ".cache" / "huggingface" / "ignored").write_bytes(b"receipt")
    expected = _digest(
        {
            "config.json": b'{"model":"test"}\n',
            "weights/a.bin": b"aaa",
            "weights/b.bin": b"bbb",
        }
    )

    assert generate_model_content_seal(model, expected_digest=expected, workers=3) == expected
    first = (model / MODEL_SEAL_NAME).read_bytes()
    assert generate_model_content_seal(model, expected_digest=expected, workers=1) == expected
    assert (model / MODEL_SEAL_NAME).read_bytes() == first
    seal = json.loads(first)
    assert [row["path"] for row in seal["files"]] == [
        "config.json",
        "weights/a.bin",
        "weights/b.bin",
    ]
    verify_model_content_seal(model, expected_digest=expected, verify_bytes=True)


def test_wrong_expected_digest_does_not_replace_existing_seal(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    target = model / "config.json"
    target.write_bytes(b"one")
    old = generate_model_content_seal(model, workers=1)
    published = (model / MODEL_SEAL_NAME).read_bytes()
    target.write_bytes(b"two")

    with pytest.raises(ModelProvisionError, match="model content digest mismatch"):
        generate_model_content_seal(model, expected_digest=old, workers=1)

    assert (model / MODEL_SEAL_NAME).read_bytes() == published


def test_model_seal_rejects_symlink_and_hardlink_aliases(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (model / "link").symlink_to(outside)
    with pytest.raises(ModelProvisionError, match="symlink"):
        generate_model_content_seal(model, workers=1)

    (model / "link").unlink()
    original = model / "original"
    original.write_bytes(b"shared")
    os.link(original, model / "alias")
    with pytest.raises(ModelProvisionError, match="hard links"):
        generate_model_content_seal(model, workers=1)


def _overlay_fixture(tmp_path: Path):
    payload = b"# SPDX-License-Identifier: Apache-2.0\nVALUE = 7\n"
    assets = tmp_path / "assets"
    source = assets / "sglang_patch" / "overlay.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    overlay = RuntimeFileOverlay(
        source="sglang_patch/overlay.py",
        target="/sgl-workspace/sglang/python/sglang/overlay.py",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    return assets, overlay, payload


def test_runtime_overlay_assets_install_exact_bytes_and_join_model_seal(tmp_path):
    assets, overlay, payload = _overlay_fixture(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"{}\n")

    installed = install_runtime_overlay_assets(
        model, (overlay,), assets_root=assets
    )
    assert installed == (model / overlay.source,)
    assert installed[0].read_bytes() == payload
    verify_runtime_overlays(model, (overlay,))
    expected = _digest({"config.json": b"{}\n", overlay.source: payload})
    assert generate_model_content_seal(
        model, expected_digest=expected, workers=2
    ) == expected
    verify_model_content_seal(model, expected_digest=expected, verify_bytes=True)


def test_registered_provision_command_installs_overlays_and_seals(
    tmp_path, monkeypatch, capsys
):
    from optima import arenas

    assets, overlay, payload = _overlay_fixture(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"{}\n")
    expected = _digest({"config.json": b"{}\n", overlay.source: payload})

    def verify_receipt(root, *, verify_bytes):
        assert verify_bytes is False
        verify_model_content_seal(root, expected_digest=expected, verify_bytes=True)

    fake = SimpleNamespace(
        model_path=str(model),
        model_id="test-model",
        model_content_digest=expected,
        runtime_overlays=(overlay,),
        verify_model_receipt=verify_receipt,
    )
    monkeypatch.setattr(arenas, "get_arena", lambda name: fake if name == "test" else None)

    assert main(
        [
            "provision",
            "--arena",
            "test",
            "--model-root",
            str(model),
            "--assets-root",
            str(assets),
            "--workers",
            "2",
        ]
    ) == 0
    assert capsys.readouterr().out.strip() == expected
    assert (model / overlay.source).read_bytes() == payload


def test_packaged_m3_overlays_match_every_registered_m3_arena():
    assets = (
        Path(__file__).resolve().parents[1]
        / "optima"
        / "arena_assets"
        / "minimax_m3"
    )
    m3 = [arena for arena in ARENAS.values() if arena.model_id == "MiniMax-M3-NVFP4"]
    assert m3
    expected = None
    for arena in m3:
        current = tuple(
            (overlay.source, overlay.sha256, overlay.size)
            for overlay in arena.runtime_overlays
        )
        expected = current if expected is None else expected
        assert current == expected
        for relative, sha256, size in current:
            path = assets / relative
            assert path.stat().st_size == size
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256
