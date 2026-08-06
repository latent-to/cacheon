"""Exact-typed JSON codec for durable qualification continuation payloads.

Continuation checkpoints must survive a process crash, so the raw evidence
dataclasses (crossover spans, engine execution evidence, pristine reference
sessions) need one faithful byte representation.  Hand-writing one closed
serializer per class across that graph invites drift, so this codec derives
the closed shape mechanically from the frozen dataclass definitions
themselves and reconstructs values only through each class's own constructor
-- every ``__post_init__`` fail-closed validation still runs on reopen.

The codec is deliberately narrow.  A codec is built from an explicit tuple of
root classes; every type reachable from their resolved field hints must be
one of: ``str``, ``int``, ``float``, ``bool``, ``pathlib.Path``, ``None``,
``Enum`` subclasses, frozen dataclasses, homogeneous ``tuple[X, ...]``,
fixed-arity tuples, or optional unions of exactly one such type and ``None``.
Anything else (callables, live handles, open unions, dictionaries, byte
strings) is rejected when the codec is *built*, not when a checkpoint is
first needed in production.

Durable records in this repository never carry JSON floats (the canonical
identity encoder rejects them), so float fields encode as the canonical
``.17g`` decimal string and decode only from that exact spelling.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from enum import Enum
from pathlib import Path


class ContinuationCodecError(RuntimeError):
    """The payload or schema cannot be represented exactly; fail closed."""


_PRIMITIVES = (str, int, float, bool)


def _type_key(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


class ContinuationCodec:
    """One closed, mechanically derived codec over an explicit class registry."""

    def __init__(self, roots: tuple[type, ...]) -> None:
        if type(roots) is not tuple or not roots:
            raise ContinuationCodecError("codec roots must be a non-empty tuple")
        self._dataclasses: dict[str, type] = {}
        self._enums: dict[str, type] = {}
        self._hints: dict[str, dict[str, object]] = {}
        for root in roots:
            self._register(root)
        self.roots = roots

    # -- registration ------------------------------------------------------

    def _register(self, cls: object) -> None:
        if isinstance(cls, type) and issubclass(cls, Enum):
            key = _type_key(cls)
            existing = self._enums.get(key)
            if existing is not None and existing is not cls:
                raise ContinuationCodecError(f"enum name collision for {key}")
            self._enums[key] = cls
            return
        if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
            raise ContinuationCodecError(
                f"codec class {cls!r} is not a dataclass or enum"
            )
        key = _type_key(cls)
        if key in self._dataclasses:
            if self._dataclasses[key] is not cls:
                raise ContinuationCodecError(f"class name collision for {key}")
            return
        self._dataclasses[key] = cls
        try:
            hints = typing.get_type_hints(cls)
        except Exception as exc:
            raise ContinuationCodecError(
                f"cannot resolve field types for {key}: {exc}"
            ) from None
        rows: dict[str, object] = {}
        for field in dataclasses.fields(cls):
            if not field.init:
                raise ContinuationCodecError(
                    f"{key}.{field.name} is not constructor-settable"
                )
            hint = hints.get(field.name)
            if hint is None:
                raise ContinuationCodecError(f"{key}.{field.name} has no type hint")
            self._validate_hint(hint, f"{key}.{field.name}")
            rows[field.name] = hint
        self._hints[key] = rows

    def _validate_hint(self, hint: object, where: str) -> None:
        if hint in _PRIMITIVES or hint is Path or hint is type(None):
            return
        origin = typing.get_origin(hint)
        if origin is tuple:
            arguments = typing.get_args(hint)
            if not arguments:
                raise ContinuationCodecError(f"{where} is an open tuple")
            if len(arguments) == 2 and arguments[1] is Ellipsis:
                self._validate_hint(arguments[0], where)
                return
            for argument in arguments:
                self._validate_hint(argument, where)
            return
        if origin in (typing.Union, types.UnionType):
            arguments = tuple(typing.get_args(hint))
            others = tuple(
                argument for argument in arguments if argument is not type(None)
            )
            if len(others) != len(arguments) - 1 or len(others) != 1:
                raise ContinuationCodecError(f"{where} is an open union")
            self._validate_hint(others[0], where)
            return
        if isinstance(hint, type) and issubclass(hint, Enum):
            self._register(hint)
            return
        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            self._register(hint)
            return
        raise ContinuationCodecError(f"{where} has unsupported type {hint!r}")

    # -- encoding ----------------------------------------------------------

    def encode(self, value: object) -> dict[str, object]:
        """Encode one registered root instance into a closed JSON object."""

        key = _type_key(type(value))
        if key not in self._dataclasses or type(value) not in self.roots:
            raise ContinuationCodecError(f"{key} is not a registered codec root")
        return {"type": key, "value": self._encode_dataclass(value)}

    def _encode_dataclass(self, value: object) -> dict[str, object]:
        key = _type_key(type(value))
        hints = self._hints.get(key)
        if hints is None or self._dataclasses.get(key) is not type(value):
            raise ContinuationCodecError(f"{key} is not exactly registered")
        return {
            name: self._encode_value(getattr(value, name), hint, f"{key}.{name}")
            for name, hint in hints.items()
        }

    def _encode_value(self, value: object, hint: object, where: str) -> object:
        origin = typing.get_origin(hint)
        if origin in (typing.Union, types.UnionType):
            if value is None:
                return None
            (inner,) = tuple(
                argument
                for argument in typing.get_args(hint)
                if argument is not type(None)
            )
            return self._encode_value(value, inner, where)
        if hint is type(None):
            if value is not None:
                raise ContinuationCodecError(f"{where} must be null")
            return None
        if hint is bool:
            if type(value) is not bool:
                raise ContinuationCodecError(f"{where} is not exactly bool")
            return value
        if hint is int:
            if type(value) is not int:
                raise ContinuationCodecError(f"{where} is not exactly int")
            return value
        if hint is float:
            if type(value) is bool or type(value) not in (int, float):
                raise ContinuationCodecError(f"{where} is not a real number")
            number = float(value)
            if not math.isfinite(number):
                raise ContinuationCodecError(f"{where} is not finite")
            return format(number, ".17g")
        if hint is str:
            if type(value) is not str:
                raise ContinuationCodecError(f"{where} is not exactly str")
            return value
        if hint is Path:
            if not isinstance(value, Path):
                raise ContinuationCodecError(f"{where} is not a path")
            return str(value)
        if origin is tuple:
            if type(value) is not tuple:
                raise ContinuationCodecError(f"{where} is not a tuple")
            arguments = typing.get_args(hint)
            if len(arguments) == 2 and arguments[1] is Ellipsis:
                return [
                    self._encode_value(row, arguments[0], where) for row in value
                ]
            if len(value) != len(arguments):
                raise ContinuationCodecError(f"{where} has the wrong tuple arity")
            return [
                self._encode_value(row, argument, where)
                for row, argument in zip(value, arguments, strict=True)
            ]
        if isinstance(hint, type) and issubclass(hint, Enum):
            if type(value) is not hint:
                raise ContinuationCodecError(f"{where} is not exactly {hint!r}")
            return value.value
        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            if type(value) is not hint:
                raise ContinuationCodecError(f"{where} is not exactly {hint!r}")
            return self._encode_dataclass(value)
        raise ContinuationCodecError(f"{where} has unsupported type {hint!r}")

    # -- decoding ----------------------------------------------------------

    def decode(self, payload: object) -> object:
        """Reconstruct one root instance; every constructor validation runs."""

        if (
            type(payload) is not dict
            or set(payload) != {"type", "value"}
            or type(payload.get("type")) is not str
        ):
            raise ContinuationCodecError("continuation payload envelope is not closed")
        key = payload["type"]
        cls = self._dataclasses.get(key)
        if cls is None or cls not in self.roots:
            raise ContinuationCodecError(f"{key} is not a registered codec root")
        return self._decode_dataclass(payload["value"], cls)

    def _decode_dataclass(self, value: object, cls: type) -> object:
        key = _type_key(cls)
        hints = self._hints[key]
        if type(value) is not dict or set(value) != set(hints):
            raise ContinuationCodecError(f"{key} payload fields are not closed")
        kwargs = {
            name: self._decode_value(value[name], hint, f"{key}.{name}")
            for name, hint in hints.items()
        }
        try:
            return cls(**kwargs)
        except ContinuationCodecError:
            raise
        except Exception as exc:
            raise ContinuationCodecError(
                f"{key} failed its own construction validation: {exc}"
            ) from None

    def _decode_value(self, value: object, hint: object, where: str) -> object:
        origin = typing.get_origin(hint)
        if origin in (typing.Union, types.UnionType):
            if value is None:
                return None
            (inner,) = tuple(
                argument
                for argument in typing.get_args(hint)
                if argument is not type(None)
            )
            return self._decode_value(value, inner, where)
        if hint is type(None):
            if value is not None:
                raise ContinuationCodecError(f"{where} must be null")
            return None
        if hint is bool:
            if type(value) is not bool:
                raise ContinuationCodecError(f"{where} is not exactly bool")
            return value
        if hint is int:
            if type(value) is not int:
                raise ContinuationCodecError(f"{where} is not exactly int")
            return value
        if hint is float:
            if type(value) is not str:
                raise ContinuationCodecError(
                    f"{where} is not a canonical decimal string"
                )
            try:
                number = float(value)
            except ValueError:
                raise ContinuationCodecError(
                    f"{where} is not a canonical decimal string"
                ) from None
            if not math.isfinite(number) or format(number, ".17g") != value:
                raise ContinuationCodecError(f"{where} is not canonical")
            return number
        if hint is str:
            if type(value) is not str:
                raise ContinuationCodecError(f"{where} is not exactly str")
            return value
        if hint is Path:
            if type(value) is not str or not value:
                raise ContinuationCodecError(f"{where} is not a path string")
            return Path(value)
        if origin is tuple:
            if type(value) is not list:
                raise ContinuationCodecError(f"{where} is not a sequence")
            arguments = typing.get_args(hint)
            if len(arguments) == 2 and arguments[1] is Ellipsis:
                return tuple(
                    self._decode_value(row, arguments[0], where) for row in value
                )
            if len(value) != len(arguments):
                raise ContinuationCodecError(f"{where} has the wrong tuple arity")
            return tuple(
                self._decode_value(row, argument, where)
                for row, argument in zip(value, arguments, strict=True)
            )
        if isinstance(hint, type) and issubclass(hint, Enum):
            try:
                return hint(value)
            except ValueError as exc:
                raise ContinuationCodecError(f"{where}: {exc}") from None
        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            return self._decode_dataclass(value, hint)
        raise ContinuationCodecError(f"{where} has unsupported type {hint!r}")


__all__ = ["ContinuationCodec", "ContinuationCodecError"]
