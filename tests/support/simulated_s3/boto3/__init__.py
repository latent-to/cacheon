"""Simulated boto3 S3 client for end-to-end operator subprocesses.

Only the snapshot children in ``tests/test_validator_operator.py`` put this
directory on their ``PYTHONPATH``; it is never on the normal import path, so it
cannot shadow an installed SDK. ``--object-store-endpoint`` names the bucket
service as a ``file://`` directory, one subdirectory per bucket, so the
production ``S3CompatibleObjectStore`` drives it through the SDK calls it makes
on a real endpoint: ``put_object`` and ``get_object``, with a missing key raised
as the SDK's ``NoSuchKey`` client error.
"""

import io
import os
from pathlib import Path
from urllib.parse import urlparse

from botocore import ClientError


class _Client:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, bucket: str, key: str) -> Path:
        return self._root / bucket / key

    def put_object(self, *, Bucket, Key, Body, ContentType):  # noqa: N803 - the SDK's keywords
        path = self._path(Bucket, Key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(Body)
        temporary.replace(path)
        return {"ContentType": ContentType}

    def get_object(self, *, Bucket, Key):  # noqa: N803 - the SDK's keywords
        path = self._path(Bucket, Key)
        if not path.is_file():
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": Key}}, "GetObject")
        body = path.read_bytes()
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}


def client(service_name, *, endpoint_url=None, **_options):
    if service_name != "s3" or not endpoint_url or urlparse(endpoint_url).scheme != "file":
        raise ValueError("the simulated SDK serves only s3 at a file:// endpoint")
    return _Client(Path(urlparse(endpoint_url).path))
