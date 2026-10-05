"""Read upstream SGLang build labels from the commissioned immutable image."""

from functools import lru_cache
import json
import logging
import re
import shlex
import subprocess
import time

LOG = logging.getLogger(__name__)


def sglang_build(runtime: str, source) -> dict:
    """Resolve only the submission's runtime, never relabel history after a cutover."""
    if not runtime or source is None:
        return {}
    try:
        registration = json.loads(source.values["REGISTRATION_PATH"].read_text())
        if registration["worker_readiness"]["runtime_digest"] != runtime:
            return {}
        stage = source.values["STAGE_ROOT"]
        for path in (stage / "ready-receipt.json", *stage.glob("*/ready-receipt.json")):
            if not path.is_file():
                continue
            ready = json.loads(path.read_text())
            if ready["receipt_digest"] != registration["ready_receipt_digest"]:
                continue
            return _image_build(registration["pod_host"], registration["pod_port"],
                                registration["pod_user"], registration["known_hosts_path"],
                                ready["worker_image"], int(time.time() // 300))
        LOG.warning("No matching READY receipt for SGLang metadata in %s", stage)
    except (OSError, ValueError, KeyError) as exc:
        LOG.warning("Cannot resolve SGLang runtime %s: %s", runtime, exc)
    return {}


@lru_cache(maxsize=64)
def _image_build(host, port, user, known_hosts, image, interval):
    # Cache by immutable image and refresh failed lookups after five minutes.
    try:
        if not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}", image):
            raise ValueError("READY image is not an immutable image reference")
        command = "docker image inspect " + shlex.quote(image) + " --format " + shlex.quote("{{json .Config.Labels}}")
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=yes",
             "-o", "UserKnownHostsFile=" + known_hosts, "-p", str(port), f"{user}@{host}", command],
            capture_output=True, text=True, check=True, timeout=5)
        labels = json.loads(result.stdout)
        commit = labels["ai.sglang.build.commit"]
        version = labels["ai.sglang.image.tag"].rsplit(":", 1)[-1]
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"v[0-9][0-9A-Za-z.+-]*", version):
            raise ValueError("Invalid upstream SGLang build labels")
        return {"commit": commit, "version": version}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        LOG.warning("Cannot read SGLang labels for %s: %s", image, exc)
        return {}
