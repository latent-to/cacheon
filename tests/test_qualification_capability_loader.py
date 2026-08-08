from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cacheon.eval import b300_remote_worker_adapter as adapter
from cacheon.eval import qualification_capability_loader as loader
from cacheon.eval.b300_qualification_commission import (
    B300QualificationCapabilities,
)
from cacheon.eval.qualification_runner import HiddenJudgeBinding


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _capabilities() -> B300QualificationCapabilities:
    class _Judge:
        binding = HiddenJudgeBinding(
            _sha256("hidden-corpus"),
            _sha256("hidden-judge"),
            _sha256("hidden-policy"),
        )

        def __call__(self, **_kwargs):
            raise AssertionError("loader tests do not execute the hidden judge")

    class _Resolver:
        def resolve_proposal(self, *_args, **_kwargs):
            raise AssertionError("loader tests do not resolve candidate sources")

        def resolve_integrated(self, *_args, **_kwargs):
            raise AssertionError("loader tests do not resolve integrated sources")

    return B300QualificationCapabilities(
        secret_loader=lambda _reference: b"s" * 32,
        entropy_provider=lambda *_args: None,
        hidden_judge=_Judge(),
        source_resolver=_Resolver(),
        source_resolver_digest=_sha256("source-resolver"),
        graph_facts_builder=lambda *_args: None,
        graph_facts_builder_digest=_sha256("graph-facts"),
        resident_count_quality_builder=lambda *_args: None,
        resident_count_quality_builder_digest=_sha256("resident-count-builder"),
    )


@pytest.fixture
def factory_support(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(imports=0, calls=0)
    support = ModuleType("qualification_loader_test_support")

    def record_import() -> None:
        state.imports += 1

    def build() -> B300QualificationCapabilities:
        state.calls += 1
        return _capabilities()

    def build_wrong() -> object:
        state.calls += 1
        return object()

    support.record_import = record_import
    support.build = build
    support.build_wrong = build_wrong
    monkeypatch.setitem(sys.modules, support.__name__, support)
    return state


def _source(*, attribute: str = "factory", expression: str = "build()") -> bytes:
    return (
        "from qualification_loader_test_support import build, build_wrong, "
        "record_import\n"
        "record_import()\n"
        f"def {attribute}():\n"
        f"    return {expression}\n"
    ).encode("utf-8")


def _write_module(root: Path, module_name: str, source: bytes) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{module_name}.py"
    path.write_bytes(source)
    importlib.invalidate_caches()
    return path


def _digest(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def test_verified_source_loads_once_and_returns_exact_typed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    module_name = "qualification_factory_correct"
    source = _source()
    source_path = _write_module(tmp_path, module_name, source)
    monkeypatch.syspath_prepend(str(tmp_path.resolve()))

    receipt = loader.load_qualification_capabilities(
        f"{module_name}:factory", _digest(source)
    )

    assert type(receipt) is loader.QualificationCapabilityLoadReceipt
    assert type(receipt.capabilities) is B300QualificationCapabilities
    assert receipt.source_path == source_path
    assert receipt.source_sha256 == _digest(source)
    assert receipt.private_module_name == (
        f"_cacheon_qualification_capability_{_digest(source)}"
    )
    assert factory_support.imports == 1
    assert factory_support.calls == 1
    assert module_name not in sys.modules
    assert receipt.private_module_name not in sys.modules


@pytest.mark.parametrize(
    "specifier",
    (
        "",
        "module_only",
        ":attribute",
        "module:",
        "package.module:factory",
        "module:factory:extra",
        "not-a-module:factory",
        "module:not-an-attribute",
    ),
)
def test_operand_requires_one_top_level_module_and_attribute(specifier: str) -> None:
    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="top-level MODULE:ATTRIBUTE|MODULE:ATTRIBUTE",
    ):
        loader.load_qualification_capabilities(specifier, "0" * 64)


@pytest.mark.parametrize(
    "source_sha256",
    ("", "0" * 63, "0" * 65, "G" * 64, "A" * 64, object()),
)
def test_source_digest_must_be_exact_lowercase_sha256(source_sha256: object) -> None:
    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="lowercase SHA-256",
    ):
        loader.load_qualification_capabilities(
            "qualification_factory_digest:factory", source_sha256  # type: ignore[arg-type]
        )


def test_wrong_digest_refuses_before_import_or_factory_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    module_name = "qualification_factory_wrong_digest"
    _write_module(tmp_path, module_name, _source())
    monkeypatch.syspath_prepend(str(tmp_path.resolve()))

    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="source digest differs",
    ):
        loader.load_qualification_capabilities(f"{module_name}:factory", "0" * 64)

    assert factory_support.imports == 0
    assert factory_support.calls == 0


def test_source_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    root = tmp_path.resolve()
    target = _write_module(root, "qualification_factory_symlink_target", _source())
    module_name = "qualification_factory_symlink"
    link = root / f"{module_name}.py"
    link.symlink_to(target)
    importlib.invalidate_caches()
    monkeypatch.syspath_prepend(str(root))

    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="path identity is unsafe|opened safely",
    ):
        loader.load_qualification_capabilities(
            f"{module_name}:factory", _digest(target.read_bytes())
        )

    assert factory_support.imports == 0
    assert factory_support.calls == 0


@pytest.mark.parametrize("kind", ("package", "namespace"))
def test_packages_and_namespace_packages_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    root = tmp_path.resolve()
    module_name = f"qualification_factory_{kind}"
    package = root / module_name
    package.mkdir()
    if kind == "package":
        package.joinpath("__init__.py").write_bytes(_source())
    importlib.invalidate_caches()
    monkeypatch.syspath_prepend(str(root))

    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="not one ordinary source file",
    ):
        loader.load_qualification_capabilities(f"{module_name}:factory", "0" * 64)


def test_extension_module_spec_is_rejected_without_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "qualification_factory_extension"
    extension_path = tmp_path.resolve() / "qualification_factory_extension.so"
    extension_path.write_bytes(b"not a native extension")
    specification = importlib.machinery.ModuleSpec(
        module_name,
        importlib.machinery.ExtensionFileLoader(module_name, str(extension_path)),
        origin=str(extension_path),
    )
    monkeypatch.setattr(
        loader.importlib.util,
        "find_spec",
        lambda observed: specification if observed == module_name else None,
    )

    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="not one ordinary source file",
    ):
        loader.load_qualification_capabilities(f"{module_name}:factory", "0" * 64)


def test_source_replacement_between_lstat_and_open_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    module_name = "qualification_factory_race"
    original = _source()
    source_path = _write_module(tmp_path, module_name, original)
    replacement = tmp_path.resolve() / "replacement.py"
    replacement.write_bytes(original.replace(b"build()", b"object()"))
    monkeypatch.syspath_prepend(str(tmp_path.resolve()))
    real_open = os.open
    swapped = False

    def swap_before_open(path, flags):
        nonlocal swapped
        if Path(path) == source_path and not swapped:
            replacement.replace(source_path)
            swapped = True
        return real_open(path, flags)

    monkeypatch.setattr(loader.os, "open", swap_before_open)
    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="changed before open",
    ):
        loader.load_qualification_capabilities(
            f"{module_name}:factory", _digest(original)
        )

    assert factory_support.imports == 0
    assert factory_support.calls == 0


def test_missing_attribute_and_non_callable_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    root = tmp_path.resolve()
    missing_name = "qualification_factory_missing_attribute"
    missing_source = _source(attribute="present")
    _write_module(root, missing_name, missing_source)
    non_callable_name = "qualification_factory_non_callable"
    non_callable_source = (
        "from qualification_loader_test_support import record_import\n"
        "record_import()\n"
        "factory = 7\n"
    ).encode("utf-8")
    _write_module(root, non_callable_name, non_callable_source)
    monkeypatch.syspath_prepend(str(root))

    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="attribute is unavailable",
    ):
        loader.load_qualification_capabilities(
            f"{missing_name}:factory", _digest(missing_source)
        )
    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="not callable",
    ):
        loader.load_qualification_capabilities(
            f"{non_callable_name}:factory", _digest(non_callable_source)
        )

    assert factory_support.imports == 2
    assert factory_support.calls == 0


def test_factory_must_have_no_parameters_and_return_exact_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    root = tmp_path.resolve()
    parameter_name = "qualification_factory_parameter"
    parameter_source = (
        "from qualification_loader_test_support import build, record_import\n"
        "record_import()\n"
        "def factory(optional=None):\n"
        "    return build()\n"
    ).encode("utf-8")
    _write_module(root, parameter_name, parameter_source)
    wrong_type_name = "qualification_factory_wrong_type"
    wrong_type_source = _source(expression="build_wrong()")
    _write_module(root, wrong_type_name, wrong_type_source)
    monkeypatch.syspath_prepend(str(root))

    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="exactly zero arguments",
    ):
        loader.load_qualification_capabilities(
            f"{parameter_name}:factory", _digest(parameter_source)
        )
    assert factory_support.calls == 0
    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="did not return exact capabilities",
    ):
        loader.load_qualification_capabilities(
            f"{wrong_type_name}:factory", _digest(wrong_type_source)
        )

    assert factory_support.imports == 2
    assert factory_support.calls == 1


def test_same_name_in_two_directories_cannot_substitute_for_selected_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    module_name = "qualification_factory_duplicate"
    selected_root = tmp_path.resolve() / "selected"
    shadowed_root = tmp_path.resolve() / "shadowed"
    selected_source = _source()
    shadowed_source = _source().replace(b"build()", b"object()")
    selected_path = _write_module(selected_root, module_name, selected_source)
    _write_module(shadowed_root, module_name, shadowed_source)
    monkeypatch.syspath_prepend(str(shadowed_root))
    monkeypatch.syspath_prepend(str(selected_root))

    with pytest.raises(
        loader.QualificationCapabilityLoadError,
        match="source digest differs",
    ):
        loader.load_qualification_capabilities(
            f"{module_name}:factory", _digest(shadowed_source)
        )
    assert factory_support.imports == 0
    assert factory_support.calls == 0

    receipt = loader.load_qualification_capabilities(
        f"{module_name}:factory", _digest(selected_source)
    )
    assert receipt.source_path == selected_path
    assert factory_support.imports == 1
    assert factory_support.calls == 1


def _required_adapter_argv(tmp_path: Path) -> list[str]:
    return [
        "--registration",
        str(tmp_path / "registration.json"),
        "--ready-receipt",
        str(tmp_path / "ready.json"),
        "--credential",
        str(tmp_path / "credential.secret"),
        "--publication-root",
        str(tmp_path / "publications"),
        "--processing-root",
        str(tmp_path / "processing"),
        "--results-root",
        str(tmp_path / "results"),
        "--continuation-root",
        str(tmp_path / "continuation"),
    ]


def test_screen_only_cli_does_not_probe_capability_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("screen-only startup must not probe capabilities")

    monkeypatch.setattr(adapter, "_load_qualification_capabilities", forbidden_loader)
    observed: list[object] = []

    def serve(_paths, capabilities):
        observed.append(capabilities)
        return 17

    monkeypatch.setattr(adapter, "_serve", serve)
    assert adapter.main(["--serve", *_required_adapter_argv(tmp_path)]) == 17
    assert observed == [None]


@pytest.mark.parametrize(
    "qualification_argv",
    (
        ["--qualification-capabilities", "private_factory:build"],
        ["--qualification-capabilities-sha256", "0" * 64],
    ),
)
def test_cli_requires_capability_specifier_and_digest_as_one_pair(
    tmp_path: Path,
    qualification_argv: list[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        adapter.main(
            ["--serve", *_required_adapter_argv(tmp_path), *qualification_argv]
        )
    assert captured.value.code == 2


def test_cli_passes_digest_bound_capabilities_to_persistent_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _capabilities()
    observed_load: list[tuple[str, str]] = []
    observed_serve: list[object] = []

    def load(specifier: str, source_sha256: str):
        observed_load.append((specifier, source_sha256))
        return SimpleNamespace(capabilities=capabilities)

    monkeypatch.setattr(adapter, "_load_qualification_capabilities", load)

    def serve(_paths, observed_capabilities):
        observed_serve.append(observed_capabilities)
        return 19

    monkeypatch.setattr(adapter, "_serve", serve)
    argv = [
        "--serve",
        *_required_adapter_argv(tmp_path),
        "--qualification-capabilities",
        "private_factory:build",
        "--qualification-capabilities-sha256",
        "a" * 64,
    ]
    assert adapter.main(argv) == 19
    assert observed_load == [("private_factory:build", "a" * 64)]
    assert observed_serve == [capabilities]
