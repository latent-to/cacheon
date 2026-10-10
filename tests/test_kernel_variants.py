"""Shape-specialized implementations sharing one semantic slot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cacheon.manifest import DEFAULT_VARIANT, ManifestError, load_manifest


SLOT = "activation.silu_and_mul"


def _write_bundle(
    root: Path,
    rows: list[dict[str, str]],
) -> Path:
    root.mkdir(parents=True)
    lines = ['bundle_id = "variants-test"', 'abi_version = "cacheon-op-abi-v0"', ""]
    for index, row in enumerate(rows):
        source = row.get("source", f"kernels/k{index}.py")
        entry = row.get("entry", f"entry_{index}")
        lines += ["[[ops]]", f'slot = "{row.get("slot", SLOT)}"']
        if "variant" in row:
            lines.append(f'variant = "{row["variant"]}"')
        lines += [f'source = "{source}"', f'entry = "{entry}"']
        if "metadata" in row:
            lines.append(f'metadata = "{row["metadata"]}"')
        lines.append("")
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"def {entry}(*args):\n    return None\n")
    (root / "manifest.toml").write_text("\n".join(lines))
    return root


def test_legacy_manifest_gets_stable_default_variant(tmp_path):
    manifest = load_manifest(_write_bundle(tmp_path / "bundle", [{}]))

    assert manifest.ops[0].variant == DEFAULT_VARIANT


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{}, {"variant": "h128"}], "every row.*explicit"),
        (
            [{"variant": "h128"}, {"variant": "h128"}],
            "duplicate variant.*h128",
        ),
        ([{"variant": "bad id"}], "simple identifier"),
    ],
)
def test_duplicate_slot_requires_unique_explicit_variant_ids(tmp_path, rows, message):
    with pytest.raises(ManifestError, match=message):
        load_manifest(_write_bundle(tmp_path / "bundle", rows))


def test_manifest_rejects_non_string_variant(tmp_path):
    bundle = _write_bundle(tmp_path / "bundle", [{}])
    text = (bundle / "manifest.toml").read_text()
    (bundle / "manifest.toml").write_text(
        text.replace('slot = "activation.silu_and_mul"',
                     'slot = "activation.silu_and_mul"\nvariant = 7')
    )

    with pytest.raises(ManifestError, match="variant.*string"):
        load_manifest(bundle)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("dtypes", '"float16"', "array of strings"),
        ("architectures", '["sm120", ""]', "non-empty strings"),
    ],
)
def test_manifest_eligibility_requires_string_arrays(
    tmp_path, field, value, message
):
    bundle = _write_bundle(tmp_path / "bundle", [{}])
    text = (bundle / "manifest.toml").read_text()
    (bundle / "manifest.toml").write_text(
        text.replace(
            'entry = "entry_0"', f'entry = "entry_0"\n{field} = {value}'
        )
    )

    with pytest.raises(ManifestError, match=message):
        load_manifest(bundle)


def test_seam_loader_registers_every_variant(tmp_path):
    from cacheon.registry import REGISTRY
    from cacheon.seam import _load_bundle_into_registry

    bundle = _write_bundle(
        tmp_path / "bundle",
        [
            {"variant": "h64", "metadata": "metadata/h64.json"},
            {"variant": "h128", "metadata": "metadata/h128.json"},
        ],
    )
    metadata = bundle / "metadata"
    metadata.mkdir()
    (metadata / "h64.json").write_text(
        json.dumps({"capabilities": {"head_dim": 64}})
    )
    (metadata / "h128.json").write_text(
        json.dumps({"capabilities": {"head_dim": 128}})
    )

    REGISTRY.clear()
    try:
        _load_bundle_into_registry(str(bundle))
        assert [impl.variant for impl in REGISTRY.variants(SLOT)] == ["h64", "h128"]
    finally:
        REGISTRY.clear()


def test_shared_setup_runs_once_across_variants(tmp_path, monkeypatch):
    from cacheon import sandbox
    from cacheon.registry import REGISTRY
    from cacheon.seam import _load_bundle_into_registry

    bundle = tmp_path / "bundle"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "metadata").mkdir()
    (bundle / "metadata" / "a.json").write_text(
        json.dumps({"capabilities": {"head_dim": 64}})
    )
    (bundle / "metadata" / "b.json").write_text(
        json.dumps({"capabilities": {"head_dim": 128}})
    )
    (bundle / "kernels" / "shared.py").write_text(
        "def entry_a(*args):\n    return None\n"
        "def entry_b(*args):\n    return None\n"
        "def setup_engine():\n    return None\n"
    )
    (bundle / "manifest.toml").write_text(
        'bundle_id = "setup-variants"\n'
        'abi_version = "cacheon-op-abi-v0"\n'
        '[[ops]]\nslot = "activation.silu_and_mul"\nvariant = "a"\n'
        'source = "kernels/shared.py"\nentry = "entry_a"\nsetup = "setup_engine"\n'
        'metadata = "metadata/a.json"\n'
        '[[ops]]\nslot = "activation.silu_and_mul"\nvariant = "b"\n'
        'source = "kernels/shared.py"\nentry = "entry_b"\nsetup = "setup_engine"\n'
        'metadata = "metadata/b.json"\n'
    )
    calls = []
    load_calls = []
    original = sandbox.callable_from
    original_load = sandbox.load_module

    def observed(module, name):
        fn = original(module, name)
        if name != "setup_engine":
            return fn

        def wrapped():
            calls.append(name)
            return fn()

        return wrapped

    def observed_load(path):
        load_calls.append(path)
        return original_load(path)

    monkeypatch.setattr(sandbox, "callable_from", observed)
    monkeypatch.setattr(sandbox, "load_module", observed_load)
    REGISTRY.clear()
    try:
        monkeypatch.delenv("CACHEON_FRAMEWORK_MODE", raising=False)
        with pytest.raises(RuntimeError, match="CACHEON_FRAMEWORK_MODE is not armed"):
            _load_bundle_into_registry(str(bundle))
        assert load_calls == []
        assert calls == []

        monkeypatch.setenv("CACHEON_FRAMEWORK_MODE", "1")
        _load_bundle_into_registry(str(bundle))
        assert len(load_calls) == 1
        assert calls == ["setup_engine"]
    finally:
        REGISTRY.clear()
