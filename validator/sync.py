"""Hippius S3 state sync for the Cacheon validator.

Upload and download the ``state/`` directory to/from Hippius S3-compatible
storage so state survives across ephemeral GPU pods.

Can be used as a library (``from validator.sync import download, upload``)
or as a standalone CLI (``python -m validator.sync download``).

Requires ``boto3`` and the following env vars:
    HIPPIUS_ACCESS_KEY, HIPPIUS_SECRET_KEY, CACHEON_S3_BUCKET
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENDPOINT_URL: str = os.environ.get("HIPPIUS_ENDPOINT_URL", "https://s3.hippius.com")
ACCESS_KEY: str = os.environ.get("HIPPIUS_ACCESS_KEY", "")
SECRET_KEY: str = os.environ.get("HIPPIUS_SECRET_KEY", "")
BUCKET: str = os.environ.get("CACHEON_S3_BUCKET", "cacheon-validator")
S3_PREFIX: str = os.environ.get("CACHEON_S3_PREFIX", "state")

SKIP_PATTERNS: set[str] = {
    ".tmp",
    ".corrupt.",
    # Legacy KV-cache era (do not sync)
    "policy-cache",
    # Pre-timestamped single-file log at state root
    "validator.log",
}


def _client():
    """Build a boto3 S3 client pointing at Hippius."""
    import boto3
    from botocore.config import Config

    if not ACCESS_KEY or not SECRET_KEY:
        raise RuntimeError(
            "HIPPIUS_ACCESS_KEY and HIPPIUS_SECRET_KEY must be set "
            "for S3 sync. See .env.gpu.example."
        )

    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="decentralized",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def _should_skip(rel_path: str) -> bool:
    return any(pat in rel_path for pat in SKIP_PATTERNS)


def upload(
    state_dir: str | Path,
    bucket: str = "",
    prefix: str = "",
    only: list[str] | None = None,
) -> int:
    """Upload files from ``state_dir`` to S3. Returns file count.

    When *only* is provided, only relative paths (files or directories)
    matching those prefixes are uploaded. E.g.
    ``only=["eval_job.json", "logs/"]`` uploads ``eval_job.json`` and
    everything under ``logs/``.
    """
    bucket = bucket or BUCKET
    prefix = prefix or S3_PREFIX
    state_path = Path(state_dir)
    if not state_path.is_dir():
        logger.warning("State dir %s does not exist, nothing to upload", state_path)
        return 0

    if only:
        logger.info(
            "⏳ S3 upload starting: %s -> s3://%s/%s (filter: %s)",
            state_path.resolve(),
            bucket,
            prefix,
            ", ".join(only),
        )
    else:
        logger.info(
            "⏳S3 upload starting: %s -> s3://%s/%s",
            state_path.resolve(),
            bucket,
            prefix,
        )

    s3 = _client()
    count = 0
    for local_file in sorted(state_path.rglob("*")):
        if not local_file.is_file():
            continue
        rel = local_file.relative_to(state_path).as_posix()
        if _should_skip(rel):
            continue
        if only and not any(rel == o or rel.startswith(o) for o in only):
            continue
        key = f"{prefix}/{rel}" if prefix else rel
        logger.debug("Uploading %s -> s3://%s/%s", local_file, bucket, key)
        s3.upload_file(str(local_file), bucket, key)
        count += 1

    logger.info("☁️  S3 upload: %d file(s) -> s3://%s/%s", count, bucket, prefix)
    return count


def download(
    state_dir: str | Path,
    bucket: str = "",
    prefix: str = "",
) -> int:
    """Download all files from S3 prefix into ``state_dir``. Returns file count."""
    bucket = bucket or BUCKET
    prefix = prefix or S3_PREFIX
    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "⏳ S3 download starting: s3://%s/%s -> %s",
        bucket,
        prefix,
        state_path.resolve(),
    )

    s3 = _client()
    paginator = s3.get_paginator("list_objects_v2")
    count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if prefix:
                rel = key[len(prefix) :].lstrip("/")
            else:
                rel = key
            if not rel or _should_skip(rel):
                continue
            local_file = state_path / rel
            local_file.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Downloading s3://%s/%s -> %s", bucket, key, local_file)
            s3.download_file(bucket, key, str(local_file))
            count += 1

    logger.info("☁️  S3 download: %d file(s) <- s3://%s/%s", count, bucket, prefix)
    return count


def _cli() -> None:
    """Minimal CLI: ``python -m validator.sync {download|upload}``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Hippius S3 state sync")
    p.add_argument("action", choices=["download", "upload"])
    p.add_argument("--state-dir", default="state")
    p.add_argument("--bucket", default="")
    p.add_argument("--prefix", default="")
    args = p.parse_args()

    if args.action == "download":
        download(args.state_dir, bucket=args.bucket, prefix=args.prefix)
    else:
        upload(args.state_dir, bucket=args.bucket, prefix=args.prefix)


if __name__ == "__main__":
    _cli()
