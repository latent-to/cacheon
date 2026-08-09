"""Shared fail-closed validators for sealed operator configuration inputs.

One implementation of the checks the standing supervisor, the mainnet screen
dispatcher, and the lease operator each carried as private copies. Every
function takes the caller's error type so refusal exception classes and
messages stay exactly what each loader's contract tests pin. The path
validator applies the strictest historical variant: absolute, NUL-free, and
canonical.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def absolute_path(
    value: object, label: str, *, error: type[Exception]
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not Path(value).is_absolute()
    ):
        raise error(f"{label} must be an absolute path")
    path = Path(value)
    if path != Path(os.path.normpath(path)):
        raise error(f"{label} must be canonical")
    return path


def authority_file(
    path: Path, label: str, *, secret: bool = False, error: type[Exception]
) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise error(f"{label} is unavailable: {exc}") from None
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
        or (secret and stat.S_IMODE(info.st_mode) & 0o077)
        or (not secret and stat.S_IMODE(info.st_mode) & 0o022)
    ):
        qualifier = (
            "owner-only regular file" if secret else "owner-controlled regular file"
        )
        raise error(f"{label} must be an {qualifier}")


def private_directory(
    path: Path, label: str, *, error: type[Exception]
) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise error(f"{label} is unavailable: {exc}") from None
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
    ):
        raise error(f"{label} must be an owner-controlled mode-0700 directory")


def positive_int(
    value: object,
    label: str,
    *,
    maximum: int | None = None,
    allow_zero: bool = False,
    error: type[Exception],
) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise error(f"{label} is outside its integer bounds")
    return value


__all__ = [
    "absolute_path",
    "authority_file",
    "positive_int",
    "private_directory",
]
