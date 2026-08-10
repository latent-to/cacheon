"""Digest-bound loader for one validator-private qualification factory.

The worker adapter is allowed to execute a private capabilities factory only
when the operator names one top-level source module and seals the exact source
bytes by SHA-256.  Resolution does not import the requested module.  The file
is reopened without following links, read once under a stable file identity,
and executed from those verified in-memory bytes under a private deterministic
module name.

This seam intentionally does not search for an alternate copy when normal
Python resolution selects the wrong file.  The selected file either matches
the supplied digest or startup fails closed.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import inspect
import os
import stat
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from cacheon.eval.b300_qualification_commission import (
    B300QualificationCapabilities,
)


MAX_CAPABILITY_SOURCE_BYTES = 4 * 1024 * 1024

_LOAD_LOCK = threading.RLock()
_LOWER_HEX = frozenset("0123456789abcdef")


class QualificationCapabilityLoadError(RuntimeError):
    """A private qualification factory could not be verified and loaded."""


@dataclass(frozen=True)
class QualificationCapabilityLoadReceipt:
    """Exact source identity and typed result of one factory invocation."""

    specifier: str
    module_name: str
    attribute_name: str
    source_path: Path
    source_sha256: str
    private_module_name: str
    capabilities: B300QualificationCapabilities


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    uid: int
    gid: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            links=value.st_nlink,
            uid=value.st_uid,
            gid=value.st_gid,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
        )


def _parse_request(specifier: str, source_sha256: str) -> tuple[str, str]:
    if type(specifier) is not str:
        raise QualificationCapabilityLoadError(
            "qualification capabilities operand must be MODULE:ATTRIBUTE"
        )
    module_name, separator, attribute_name = specifier.partition(":")
    if (
        separator != ":"
        or not module_name
        or not attribute_name
        or ":" in attribute_name
        or "." in module_name
        or not module_name.isidentifier()
        or not attribute_name.isidentifier()
    ):
        raise QualificationCapabilityLoadError(
            "qualification capabilities operand must be one top-level "
            "MODULE:ATTRIBUTE"
        )
    if (
        type(source_sha256) is not str
        or len(source_sha256) != 64
        or any(character not in _LOWER_HEX for character in source_sha256)
    ):
        raise QualificationCapabilityLoadError(
            "qualification capabilities source digest must be one lowercase SHA-256"
        )
    return module_name, attribute_name


def _resolved_source(module_name: str) -> Path:
    if module_name in sys.modules:
        raise QualificationCapabilityLoadError(
            "qualification capabilities module was already imported"
        )
    try:
        specification = importlib.util.find_spec(module_name)
    except Exception as exc:
        raise QualificationCapabilityLoadError(
            f"qualification capabilities module cannot be resolved: {exc}"
        ) from None
    if specification is None:
        raise QualificationCapabilityLoadError(
            "qualification capabilities module cannot be resolved"
        )
    if (
        specification.name != module_name
        or not specification.has_location
        or specification.submodule_search_locations is not None
        or type(specification.loader) is not importlib.machinery.SourceFileLoader
        or type(specification.origin) is not str
        or specification.origin in {"built-in", "frozen"}
    ):
        raise QualificationCapabilityLoadError(
            "qualification capabilities module is not one ordinary source file"
        )
    loader = specification.loader
    if loader.name != module_name or loader.path != specification.origin:
        raise QualificationCapabilityLoadError(
            "qualification capabilities module has an ambiguous source identity"
        )
    source_path = Path(specification.origin)
    if (
        not source_path.is_absolute()
        or source_path.suffix != ".py"
        or os.path.normpath(specification.origin) != specification.origin
        or os.path.realpath(specification.origin) != specification.origin
    ):
        raise QualificationCapabilityLoadError(
            "qualification capabilities source path identity is unsafe"
        )
    return source_path


def _stable_source_bytes(source_path: Path) -> bytes:
    try:
        before = os.lstat(source_path)
    except OSError as exc:
        raise QualificationCapabilityLoadError(
            f"qualification capabilities source cannot be inspected: {exc}"
        ) from None
    before_identity = _FileIdentity.from_stat(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_size < 0
        or before.st_size > MAX_CAPABILITY_SOURCE_BYTES
    ):
        raise QualificationCapabilityLoadError(
            "qualification capabilities source is not one bounded regular file"
        )
    no_follow = getattr(os, "O_NOFOLLOW", None)
    close_on_exec = getattr(os, "O_CLOEXEC", None)
    if type(no_follow) is not int or type(close_on_exec) is not int:
        raise QualificationCapabilityLoadError(
            "qualification capabilities source requires O_NOFOLLOW and O_CLOEXEC"
        )
    try:
        descriptor = os.open(
            source_path,
            os.O_RDONLY | no_follow | close_on_exec,
        )
    except OSError as exc:
        raise QualificationCapabilityLoadError(
            f"qualification capabilities source cannot be opened safely: {exc}"
        ) from None
    try:
        opened_identity = _FileIdentity.from_stat(os.fstat(descriptor))
        if opened_identity != before_identity or not stat.S_ISREG(opened_identity.mode):
            raise QualificationCapabilityLoadError(
                "qualification capabilities source changed before open"
            )
        chunks: list[bytes] = []
        remaining = MAX_CAPABILITY_SOURCE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        source_bytes = b"".join(chunks)
        after_descriptor_identity = _FileIdentity.from_stat(os.fstat(descriptor))
        try:
            after_path_identity = _FileIdentity.from_stat(os.lstat(source_path))
        except OSError as exc:
            raise QualificationCapabilityLoadError(
                f"qualification capabilities source disappeared during read: {exc}"
            ) from None
        if (
            len(source_bytes) > MAX_CAPABILITY_SOURCE_BYTES
            or len(source_bytes) != before_identity.size
            or after_descriptor_identity != before_identity
            or after_path_identity != before_identity
        ):
            raise QualificationCapabilityLoadError(
                "qualification capabilities source changed during read"
            )
        return source_bytes
    finally:
        os.close(descriptor)


def _invoke_verified_factory(
    *,
    module_name: str,
    attribute_name: str,
    source_path: Path,
    source_sha256: str,
    source_bytes: bytes,
) -> tuple[str, B300QualificationCapabilities]:
    private_module_name = f"_cacheon_qualification_capability_{source_sha256}"
    if private_module_name in sys.modules or module_name in sys.modules:
        raise QualificationCapabilityLoadError(
            "qualification capabilities module identity is already occupied"
        )
    try:
        code = compile(
            source_bytes,
            str(source_path),
            "exec",
            dont_inherit=True,
            optimize=0,
        )
    except (SyntaxError, UnicodeError, ValueError) as exc:
        raise QualificationCapabilityLoadError(
            f"qualification capabilities source cannot be compiled: {exc}"
        ) from None
    module = ModuleType(private_module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        private_module_name,
        loader=None,
        origin=str(source_path),
    )
    sys.modules[private_module_name] = module
    sys.modules[module_name] = module
    try:
        try:
            exec(code, module.__dict__)
        except Exception as exc:
            raise QualificationCapabilityLoadError(
                "qualification capabilities source failed during verified import: "
                f"{type(exc).__name__}: {exc}"
            ) from None
        if (
            sys.modules.get(private_module_name) is not module
            or sys.modules.get(module_name) is not module
        ):
            raise QualificationCapabilityLoadError(
                "qualification capabilities source replaced its module identity"
            )
        try:
            factory = vars(module)[attribute_name]
        except KeyError:
            raise QualificationCapabilityLoadError(
                "qualification capabilities factory attribute is unavailable"
            ) from None
        if not callable(factory):
            raise QualificationCapabilityLoadError(
                "qualification capabilities factory is not callable"
            )
        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError) as exc:
            raise QualificationCapabilityLoadError(
                f"qualification capabilities factory signature is unavailable: {exc}"
            ) from None
        # The invariant is that the loader invokes the factory with no
        # arguments, so what disqualifies a factory is a parameter it cannot
        # supply itself. A parameter carrying a default -- or a *args/**kwargs
        # that binds to nothing -- still satisfies the zero-argument call and
        # must not be rejected: refusing it costs a live commission for a
        # signature that is already correct.
        required = tuple(
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )
        if required:
            raise QualificationCapabilityLoadError(
                "qualification capabilities factory must accept exactly zero"
                f" arguments; {', '.join(required)} has no default"
            )
        try:
            capabilities = factory()
        except Exception as exc:
            raise QualificationCapabilityLoadError(
                "qualification capabilities factory failed: "
                f"{type(exc).__name__}: {exc}"
            ) from None
        if (
            sys.modules.get(private_module_name) is not module
            or sys.modules.get(module_name) is not module
        ):
            raise QualificationCapabilityLoadError(
                "qualification capabilities factory replaced its module identity"
            )
        if type(capabilities) is not B300QualificationCapabilities:
            raise QualificationCapabilityLoadError(
                "qualification capabilities factory did not return exact capabilities"
            )
        return private_module_name, capabilities
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.pop(private_module_name, None)


def load_qualification_capabilities(
    specifier: str,
    source_sha256: str,
) -> QualificationCapabilityLoadReceipt:
    """Verify, execute, and invoke one explicitly digest-bound factory once."""

    module_name, attribute_name = _parse_request(specifier, source_sha256)
    with _LOAD_LOCK:
        source_path = _resolved_source(module_name)
        source_bytes = _stable_source_bytes(source_path)
        actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if actual_sha256 != source_sha256:
            raise QualificationCapabilityLoadError(
                "qualification capabilities source digest differs"
            )
        private_module_name, capabilities = _invoke_verified_factory(
            module_name=module_name,
            attribute_name=attribute_name,
            source_path=source_path,
            source_sha256=source_sha256,
            source_bytes=source_bytes,
        )
    return QualificationCapabilityLoadReceipt(
        specifier=specifier,
        module_name=module_name,
        attribute_name=attribute_name,
        source_path=source_path,
        source_sha256=source_sha256,
        private_module_name=private_module_name,
        capabilities=capabilities,
    )
