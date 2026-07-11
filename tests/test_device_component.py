from __future__ import annotations

import ctypes
import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from optima.competition import CompetitionError, SLOT_MODE, resolve_competition
from optima.device_component import (
    DEVICE_ABIS,
    SILU_CUDA_ABI,
    UNTRUSTED_HOST_SYSTEM_TARGET,
    DeviceComponentError,
    _SiluEntry,
    _validate_artifact,
    component_crown_rejection,
    device_manifest_rejection,
    prepare_device_artifacts,
    untrusted_host_product_fingerprints,
    untrusted_host_system_rejection,
)
from optima.manifest import (
    CompetitionEntry,
    UNTRUSTED_HOST_EXECUTION,
    VALIDATOR_DEVICE_EXECUTION,
    ManifestError,
    load_manifest,
)
from optima.sandbox import scan_tree


SLOT = "activation.silu_and_mul"


def _host_bundle(root: Path, source: str | None = None) -> Path:
    root.mkdir(parents=True)
    source = source or "def silu_and_mul(x, out):\n    out.copy_(x[..., :out.shape[-1]])\n"
    (root / "kernel.py").write_text(source)
    (root / "manifest.toml").write_text(
        'bundle_id = "host"\n'
        'abi_version = "optima-op-abi-v0"\n'
        "[competition]\n"
        f'target = "{SLOT}"\n'
        'mode = "slot"\n'
        "[[ops]]\n"
        f'slot = "{SLOT}"\n'
        'source = "kernel.py"\n'
        'entry = "silu_and_mul"\n'
        'dtypes = ["bfloat16"]\n'
    )
    return root


def _device_bundle(
    root: Path,
    *,
    slot: str = SLOT,
    device_abi: str = SILU_CUDA_ABI,
    extra: str = "",
) -> Path:
    root.mkdir(parents=True)
    (root / "kernel.cu").write_text(
        'extern "C" __global__ void silu_bf16(const void* x, void* out, '
        "unsigned long long n, unsigned int d) {}\n"
    )
    (root / "manifest.toml").write_text(
        'bundle_id = "device"\n'
        'abi_version = "optima-op-abi-v0"\n'
        "[competition]\n"
        f'target = "{slot}"\n'
        'mode = "slot"\n'
        "[[ops]]\n"
        f'slot = "{slot}"\n'
        'source = "kernel.cu"\n'
        'entry = "silu_bf16"\n'
        f'execution_class = "{VALIDATOR_DEVICE_EXECUTION}"\n'
        f'device_abi = "{device_abi}"\n'
        'dtypes = ["bfloat16"]\n'
        f"{extra}"
    )
    return root


def _host_system_bundle(root: Path, slots: tuple[str, ...]) -> Path:
    root.mkdir(parents=True)
    (root / "kernel.py").write_text("def entry(*args):\n    return None\n")
    rows = [
        'bundle_id = "host-system"',
        'abi_version = "optima-op-abi-v0"',
        "[competition]",
        f'target = "{UNTRUSTED_HOST_SYSTEM_TARGET}"',
        'mode = "system"',
    ]
    for slot in slots:
        rows.extend(
            [
                "[[ops]]",
                f'slot = "{slot}"',
                'source = "kernel.py"',
                'entry = "entry"',
            ]
        )
    (root / "manifest.toml").write_text("\n".join(rows) + "\n")
    return root


def test_legacy_python_is_explicitly_untrusted_and_not_component_crownable(tmp_path):
    manifest = load_manifest(_host_bundle(tmp_path / "host"))

    assert manifest.ops[0].execution_class == UNTRUSTED_HOST_EXECUTION
    resolved = resolve_competition(manifest)
    assert resolved.target == SLOT and resolved.mode == SLOT_MODE
    assert not resolved.crownable
    assert "scheduler Python" in (resolved.reason or "")
    assert UNTRUSTED_HOST_SYSTEM_TARGET in (resolved.reason or "")
    assert "mode='system'" in (resolved.reason or "")
    with pytest.raises(
        CompetitionError,
        match=r"untrusted_host scheduler Python.*isolated system lane",
    ):
        resolve_competition(manifest, for_settlement=True)


def test_scanner_clean_sglang_monkeypatch_is_still_untrusted_host(tmp_path):
    bundle = _host_bundle(
        tmp_path / "monkeypatch",
        "import sglang\n"
        "def replacement(*args):\n    return None\n"
        "def silu_and_mul(x, out):\n"
        "    setattr(sglang, 'miner_replacement', replacement)\n"
        "    out.copy_(x[..., :out.shape[-1]])\n",
    )
    manifest = load_manifest(bundle)
    # This intentionally demonstrates why AST cleanliness is not authority.
    assert scan_tree(bundle, declared_cuda_sources=frozenset()).ok
    assert component_crown_rejection(manifest) is not None
    with pytest.raises(CompetitionError, match="untrusted_host"):
        resolve_competition(manifest, for_settlement=True)


def test_scanner_clean_sglang_monkeypatch_really_executes_in_legacy_loader(
    tmp_path, monkeypatch
):
    from optima.registry import REGISTRY
    from optima.seam import _load_bundle_into_registry

    bundle = _host_bundle(
        tmp_path / "loader-monkeypatch",
        "import sglang\n"
        "def replacement(*args):\n    return 'miner-controlled'\n"
        "sglang.forward = replacement\n"
        "def silu_and_mul(x, out):\n"
        "    out.copy_(x[..., :out.shape[-1]])\n",
    )
    fake_sglang = types.ModuleType("sglang")
    fake_sglang.forward = lambda *_a: "stock"
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    manifest = load_manifest(bundle)
    assert scan_tree(bundle, declared_cuda_sources=frozenset()).ok

    REGISTRY.clear()
    try:
        _load_bundle_into_registry(str(bundle))
        assert fake_sglang.forward() == "miner-controlled"
        assert REGISTRY.variants(SLOT)  # it looked like a normal component load
        assert not resolve_competition(manifest).crownable
    finally:
        REGISTRY.clear()


def test_validator_device_manifest_is_valid_for_development_but_not_crownable(
    tmp_path,
):
    manifest = load_manifest(_device_bundle(tmp_path / "device"))

    op = manifest.ops[0]
    assert op.execution_class == VALIDATOR_DEVICE_EXECUTION
    assert op.device_abi == SILU_CUDA_ABI
    assert op.source == "kernel.cu"
    assert device_manifest_rejection(manifest) is None
    assert "raw CUDA device pointers" in component_crown_rejection(manifest)
    resolved = resolve_competition(manifest)
    assert not resolved.crownable and resolved.target == SLOT
    with pytest.raises(CompetitionError, match="raw CUDA device pointers"):
        resolve_competition(manifest, for_settlement=True)
    assert DEVICE_ABIS[SILU_CUDA_ABI].slot == SLOT


@pytest.mark.parametrize(
    "rewrite, match",
    [
        (("validator_device", "anything"), "execution_class.*one of"),
        (("source = \"kernel.cu\"", "source = \"kernel.py\""), "must be a .cu"),
        ((f'device_abi = "{SILU_CUDA_ABI}"\n', ""), "requires.*device_abi"),
        (("dtypes =", 'setup = "evil"\ndtypes ='), "cannot declare prepare/setup"),
    ],
)
def test_validator_device_manifest_fails_closed_on_host_surface(
    tmp_path, rewrite, match
):
    bundle = _device_bundle(tmp_path / "device")
    text = (bundle / "manifest.toml").read_text().replace(*rewrite)
    if "kernel.py" in text:
        (bundle / "kernel.py").write_text("def silu_bf16(): pass\n")
    (bundle / "manifest.toml").write_text(text)
    with pytest.raises(ManifestError, match=match):
        load_manifest(bundle)


def test_untrusted_host_cannot_claim_device_abi(tmp_path):
    bundle = _host_bundle(tmp_path / "host")
    manifest = bundle / "manifest.toml"
    manifest.write_text(
        manifest.read_text().replace(
            'entry = "silu_and_mul"',
            f'entry = "silu_and_mul"\ndevice_abi = "{SILU_CUDA_ABI}"',
        )
    )
    with pytest.raises(ManifestError, match="untrusted_host may not claim"):
        load_manifest(bundle)


def test_validator_device_rejects_rebuild_plan_and_dependency_patch(tmp_path):
    rebuild = _device_bundle(tmp_path / "rebuild")
    (rebuild / "rebuild.json").write_text('{"steps": []}')
    with pytest.raises(ManifestError, match="may not select rebuild.json"):
        load_manifest(rebuild)

    dep = _device_bundle(tmp_path / "dep")
    (dep / "p.patch").write_text(
        "--- a/x.cu\n+++ b/x.cu\n@@ -1 +1 @@\n-a\n+b\n"
    )
    manifest = dep / "manifest.toml"
    manifest.write_text(
        manifest.read_text().replace(
            "[[ops]]",
            '[[dep_patches]]\ntarget = "flashinfer"\npath = "p.patch"\n\n[[ops]]',
        )
    )
    with pytest.raises(ManifestError, match="may not patch host dependencies"):
        load_manifest(dep)


def test_device_builder_refuses_undeclared_compilation_inputs_before_cuda(tmp_path):
    bundle = _device_bundle(tmp_path / "device")
    (bundle / "hidden.cuh").write_text("#define STOLEN_BUILD_INPUT 1\n")
    with pytest.raises(DeviceComponentError, match="declared-source scan"):
        prepare_device_artifacts(bundle, phase="build")


def test_unknown_or_cross_slot_device_abi_cannot_crown(tmp_path):
    unknown = load_manifest(
        _device_bundle(tmp_path / "unknown", device_abi="miner.chosen.abi.v1")
    )
    with pytest.raises(CompetitionError, match="unknown validator-owned device_abi"):
        resolve_competition(unknown, for_settlement=True)

    wrong = load_manifest(
        _device_bundle(
            tmp_path / "wrong",
            slot="norm.rmsnorm",
            device_abi=SILU_CUDA_ABI,
        )
    )
    with pytest.raises(CompetitionError, match="belongs to slot"):
        resolve_competition(wrong, for_settlement=True)


def test_device_artifact_stamp_fails_closed_on_cubin_tamper(tmp_path):
    from optima.device_component import _stamp

    cubin = tmp_path / "kernel.cubin"
    stamp_path = tmp_path / "artifact.json"
    cubin.write_bytes(b"one")
    identity = {"schema": "identity"}
    stamp_path.write_text(json.dumps(_stamp(identity, "abc", cubin)))
    assert _validate_artifact(
        cubin, stamp_path, identity=identity, artifact_id="abc"
    )[0]

    cubin.write_bytes(b"two")
    ok, why = _validate_artifact(
        cubin, stamp_path, identity=identity, artifact_id="abc"
    )
    assert not ok and "hash differs" in why


def test_seam_device_lane_never_imports_bundle_host_code(tmp_path, monkeypatch):
    from optima import device_component, rebuild, sandbox
    from optima.registry import REGISTRY
    from optima.seam import _load_bundle_into_registry

    bundle = _device_bundle(tmp_path / "device")
    trusted_entry = lambda x, out: None
    monkeypatch.setattr(rebuild, "apply_rebuild_plan", lambda *_a, **_k: True)
    monkeypatch.setattr(
        device_component, "load_device_entry", lambda *_a, **_k: trusted_entry
    )
    monkeypatch.setattr(
        sandbox,
        "load_module",
        lambda *_a, **_k: pytest.fail("validator_device imported a bundle module"),
    )
    REGISTRY.clear()
    try:
        _load_bundle_into_registry(str(bundle))
        impl = REGISTRY.variants(SLOT)[0]
        assert impl.entry is trusted_entry
    finally:
        REGISTRY.clear()


def test_trusted_silu_launcher_owns_grid_block_and_argument_marshalling(monkeypatch):
    import torch

    launches = []

    class Driver:
        def launch(self, function, **kwargs):
            params = kwargs["params"]
            values = (
                ctypes.cast(params[0], ctypes.POINTER(ctypes.c_uint64)).contents.value,
                ctypes.cast(params[1], ctypes.POINTER(ctypes.c_uint64)).contents.value,
                ctypes.cast(params[2], ctypes.POINTER(ctypes.c_uint64)).contents.value,
                ctypes.cast(params[3], ctypes.POINTER(ctypes.c_uint32)).contents.value,
            )
            launches.append((function, kwargs, values))

    class Tensor:
        is_cuda = True
        dtype = torch.bfloat16
        device = "cuda:0"

        def __init__(self, shape, pointer):
            self.shape = shape
            self.ndim = len(shape)
            self._pointer = pointer

        def is_contiguous(self):
            return True

        def numel(self):
            n = 1
            for dim in self.shape:
                n *= dim
            return n

        def data_ptr(self):
            return self._pointer

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _device: type("S", (), {"cuda_stream": 77})()
    )
    entry = _SiluEntry(Path("never-loaded.cubin"), "miner_symbol")
    entry._loaded = (Driver(), ctypes.c_void_p(1), ctypes.c_void_p(2))

    entry(Tensor((3, 2048), 0x1000), Tensor((3, 1024), 0x2000))

    assert len(launches) == 1
    function, launch, values = launches[0]
    assert function.value == 2
    assert launch["grid_x"] == 12  # ceil((3*1024) / 256)
    assert launch["block_x"] == 256
    assert launch["stream"] == 77
    assert values == (0x1000, 0x2000, 3072, 1024)


def test_untrusted_host_bundles_are_explicit_system_products(tmp_path):
    bundles = (
        _host_system_bundle(
            tmp_path / "deep",
            (
                "collective.ar_residual_rmsnorm",
                "collective.moe_finalize_ar_rmsnorm",
            ),
        ),
        _host_system_bundle(
            tmp_path / "blockscore", ("attention.msa_prefill_block_score",)
        ),
    )
    for bundle in bundles:
        manifest = load_manifest(bundle)
        assert {op.execution_class for op in manifest.ops} == {
            UNTRUSTED_HOST_EXECUTION
        }
        component_reason = component_crown_rejection(manifest)
        assert component_reason is not None
        assert "isolated system lane" in component_reason
        assert untrusted_host_system_rejection(manifest) is None
        assert untrusted_host_product_fingerprints(bundle)
        system = resolve_competition(
            manifest, for_settlement=True, warn_legacy=False
        )
        assert system.target == UNTRUSTED_HOST_SYSTEM_TARGET
        assert system.mode == "system"
        assert system.members == ()
        assert system.crownable and not system.legacy


def test_cli_verify_routes_cuda_source_to_device_verifier(tmp_path, monkeypatch):
    import torch

    from optima import cli
    from optima.eval import _launch

    bundle = _device_bundle(tmp_path / "device")
    seen = {}

    def fake_call(fn, *args, **kwargs):
        seen.update(fn=fn, args=args, kwargs=kwargs)
        return SimpleNamespace(passed=True)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(_launch, "call_in_subprocess", fake_call)
    monkeypatch.setattr(
        cli,
        "scan_path",
        lambda *_a, **_k: pytest.fail(".cu source was routed through Python scan"),
    )
    monkeypatch.setattr(
        __import__("optima.verify", fromlist=["format_verify"]),
        "format_verify",
        lambda _result: "device verify passed",
    )

    rc = cli.cmd_verify(
        SimpleNamespace(
            bundle=str(bundle),
            world_size=2,
            device="cuda",
            model=None,
            dtype="bfloat16",
            seed=7,
        )
    )

    from optima.device_component import verify_device_entry_from_bundle

    assert rc == 0
    assert seen["fn"] is verify_device_entry_from_bundle
    assert seen["args"] == (str(bundle), SLOT, "default")
    assert seen["kwargs"] == {
        "dtype_name": "bfloat16",
        "device": "cuda",
        "seed": 7,
        "jitter_seed": 7,
        "model_key": None,
    }


def test_whole_serving_system_target_is_exact_and_rejects_device_products(tmp_path):
    host = load_manifest(_host_bundle(tmp_path / "host"))
    wrong = replace(
        host,
        competition=CompetitionEntry(target="miner.system.target", mode="system"),
    )
    with pytest.raises(CompetitionError, match="whole-serving system target"):
        resolve_competition(wrong, for_settlement=True)

    device = load_manifest(_device_bundle(tmp_path / "device"))
    device = replace(
        device,
        competition=CompetitionEntry(
            target=UNTRUSTED_HOST_SYSTEM_TARGET, mode="system"
        ),
    )
    with pytest.raises(CompetitionError, match="only untrusted_host"):
        resolve_competition(device, for_settlement=True)
