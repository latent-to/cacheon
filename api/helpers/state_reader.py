"""Thin readers for on-disk state files. All reads are best-effort and
never raise on missing or corrupt data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def safe_json_load(path: Path, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return default


def safe_jsonl_load(path: Path) -> list[dict]:
    entries: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        pass
    return entries


def sanitize_floats(obj: Any) -> Any:
    """Replace NaN/Inf with None so json.dumps doesn't blow up."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    return obj
