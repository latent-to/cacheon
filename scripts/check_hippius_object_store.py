#!/usr/bin/env python3
"""Smoke-test Hippius (or any S3-compatible) object-store put/get via Optima.

Requires a Master Token from https://console.hippius.com (Access Key starts
with hip_) and an existing bucket. Wallets / account_ss58 are not used for S3.

Examples:

  export OPTIMA_OBJECT_STORE_ACCESS_KEY_ID=hip_...
  export OPTIMA_OBJECT_STORE_SECRET_ACCESS_KEY=...
  export OPTIMA_OBJECT_STORE_BUCKET=optima-weights

  python scripts/check_hippius_object_store.py

  python scripts/check_hippius_object_store.py \\
    --bucket optima-weights \\
    --prefix smoke \\
    --endpoint https://us-central-1.hippius.com
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _env(name: str) -> str | None:
    value = os.environ.get(f"OPTIMA_OBJECT_STORE_{name}")
    if value is None or value == "":
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        default="",
        help="hippius|s3|minio (default: hippius or OPTIMA_OBJECT_STORE_PROVIDER)",
    )
    parser.add_argument("--bucket", default="", help="bucket name")
    parser.add_argument("--prefix", default="", help="optional key prefix")
    parser.add_argument("--endpoint", default="", help="override endpoint URL")
    parser.add_argument("--region", default="", help="override region")
    parser.add_argument("--access-key", default="", help="S3 access key id")
    parser.add_argument("--secret-key", default="", help="S3 secret access key")
    parser.add_argument(
        "--key",
        default="",
        help="object key (default: optima-smoke/<uuid>.txt)",
    )
    args = parser.parse_args(argv)

    from optima.object_store import (
        ObjectStoreConfig,
        ObjectStoreError,
        open_configured_object_store,
    )

    provider = (
        args.provider or _env("PROVIDER") or "hippius"
    ).strip().lower()
    bucket = args.bucket or _env("BUCKET") or ""
    if not bucket:
        print(
            "error: set --bucket or OPTIMA_OBJECT_STORE_BUCKET",
            file=sys.stderr,
        )
        return 2

    cfg = ObjectStoreConfig(
        provider=provider,
        bucket=bucket,
        key_prefix=args.prefix if args.prefix else (_env("KEY_PREFIX") or ""),
        endpoint_url=args.endpoint or _env("ENDPOINT_URL"),
        region_name=args.region or _env("REGION"),
        access_key_id=args.access_key
        or _env("ACCESS_KEY_ID")
        or os.environ.get("AWS_ACCESS_KEY_ID"),
        secret_access_key=args.secret_key
        or _env("SECRET_ACCESS_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    key = args.key or f"optima-smoke/{uuid.uuid4().hex}.txt"
    payload = (
        f"optima hippius smoke {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"{uuid.uuid4().hex}\n"
    ).encode()

    print(f"provider={cfg.provider}")
    print(f"bucket={cfg.bucket}")
    print(f"prefix={cfg.key_prefix!r}")
    print(f"endpoint={cfg.endpoint_url or '(preset default)'}")
    print(f"region={cfg.region_name or '(preset default)'}")
    print(f"key={key}")
    ak = cfg.access_key_id or ""
    print(f"access_key={ak[:8]}…" if len(ak) > 8 else f"access_key={ak or '(missing)'}")

    try:
        store = open_configured_object_store(cfg)
        store.put_bytes(key, payload, content_type="text/plain")
        print("put: ok")
        got = store.get_bytes(key)
        print(f"get: ok ({len(got)} bytes)")
    except ObjectStoreError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if got != payload:
        print("FAIL: round-trip bytes mismatch", file=sys.stderr)
        return 1

    print("PASS: put/get round-trip matches")
    print(f"(object left at s3://{cfg.bucket}/{cfg.resolve_key(key)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
