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
from tests.support.b300 import qualification_capabilities as _capabilities


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        loader.importlib.machinery.PathFinder,
        "find_spec",
        staticmethod(
            lambda observed, path=None, target=None: (
                specification if observed == module_name else None
            )
        ),
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


def test_a_defaulted_parameter_still_satisfies_the_zero_argument_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    """The loader calls the factory with no arguments, so a parameter carrying
    a default is already satisfied. Rejecting it is not a safety property: it
    costs a re-commission of a sealed capability source whose signature is
    correct. The deployed B300 factory is exactly this shape."""

    root = tmp_path.resolve()
    name = "qualification_factory_defaulted"
    source = (
        "from qualification_loader_test_support import build, record_import\n"
        "record_import()\n"
        "def factory(inputs=None):\n"
        "    return build()\n"
    ).encode("utf-8")
    _write_module(root, name, source)
    monkeypatch.syspath_prepend(str(root))

    receipt = loader.load_qualification_capabilities(
        f"{name}:factory", _digest(source)
    )

    assert receipt.attribute_name == "factory"
    assert factory_support.calls == 1


def test_factory_must_have_no_required_parameters_and_return_exact_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    root = tmp_path.resolve()
    parameter_name = "qualification_factory_parameter"
    parameter_source = (
        "from qualification_loader_test_support import build, record_import\n"
        "record_import()\n"
        "def factory(required):\n"
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


def test_an_already_imported_module_is_reused_with_its_operator_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    """The pod adapter binds calibration entropy and seeded managers onto this
    module before serving. Executing a second private copy of the same bytes
    would drop those bindings and evaluate against the wrong authorities, so a
    live import is reused -- after its bytes still verify."""

    module_name = "qualification_factory_preimported"
    source = _source()
    source_path = _write_module(tmp_path, module_name, source)
    monkeypatch.syspath_prepend(str(tmp_path.resolve()))

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    preimported = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, preimported)
    spec.loader.exec_module(preimported)
    preimported.OPERATOR_BINDING = "installed"

    receipt = loader.load_qualification_capabilities(
        f"{module_name}:factory", _digest(source)
    )

    assert type(receipt.capabilities) is B300QualificationCapabilities
    # The live module is the one that answered, and it is left intact for the
    # rest of the serving process.
    assert receipt.private_module_name == module_name
    assert sys.modules[module_name] is preimported
    assert preimported.OPERATOR_BINDING == "installed"
    # Reused, not re-executed: the module body ran once, at the operator's
    # import, and only the factory was called by the loader.
    assert factory_support.imports == 1
    assert factory_support.calls == 1


def test_a_preimported_module_from_another_file_is_still_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    """Reuse is pinned to the exact resolved file. A module occupying the name
    from anywhere else is an identity substitution, not an operator binding."""

    module_name = "qualification_factory_impostor"
    source = _source()
    _write_module(tmp_path, module_name, source)
    monkeypatch.syspath_prepend(str(tmp_path.resolve()))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    other_path = _write_module(elsewhere, "qualification_factory_other", source)
    spec = importlib.util.spec_from_file_location(module_name, other_path)
    impostor = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, impostor)
    spec.loader.exec_module(impostor)

    with pytest.raises(
        loader.QualificationCapabilityLoadError, match="imported from another file"
    ):
        loader.load_qualification_capabilities(
            f"{module_name}:factory", _digest(source)
        )


def test_a_preimported_module_whose_bytes_changed_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_support: SimpleNamespace,
) -> None:
    """Reuse never skips the seal: the file still has to match the digest."""

    module_name = "qualification_factory_drifted"
    source = _source()
    source_path = _write_module(tmp_path, module_name, source)
    monkeypatch.syspath_prepend(str(tmp_path.resolve()))

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    preimported = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, preimported)
    spec.loader.exec_module(preimported)

    with pytest.raises(
        loader.QualificationCapabilityLoadError, match="source digest differs"
    ):
        loader.load_qualification_capabilities(
            f"{module_name}:factory", _sha256("a different sealed declaration")
        )
