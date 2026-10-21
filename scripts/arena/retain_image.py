"""Retain a published worker image in the validator's registry by digest and write its durable image proof.

The published reference is a Docker Hub image by index digest; skopeo copies it with its digests
preserved into the validator's local registry under a repository tag, the manifest the registry then
serves is read back by both OCI and Docker media types (a published OCI manifest answers 404 to a
Docker-only Accept), and the proof names the retained manifest digest as the image every launcher,
commissioner and connect step checks. The proof shape is the one ``launch_node.py`` and the
rendered operator consume; every field is an argument or read from the registry, none is typed by
hand into the file.
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
ACCEPT = "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", required=True, help="docker.io/<org>/<repo>@sha256:<index digest>")
    parser.add_argument("--repository", required=True, help="repository name in the validator registry")
    parser.add_argument("--tag", required=True, help="registry tag, conventionally the source revision prefix")
    parser.add_argument("--registry", default="127.0.0.1:5098", help="the registry endpoint skopeo pushes to")
    parser.add_argument("--worker-registry", default="127.0.0.1:5099",
                        help="the endpoint the pod pulls through (the launch tunnel)")
    parser.add_argument("--registry-root", required=True, help="the registry's storage path on this host")
    parser.add_argument("--sglang-image", required=True, help="the base engine image reference, by digest")
    parser.add_argument("--source-revision", required=True, help="full cacheon commit the image installs")
    parser.add_argument("--out", required=True, type=Path, help="durable-images.json to write (must not exist)")
    parser.add_argument("--manifest-out", required=True, type=Path, help="raw manifest bytes as served")
    parser.add_argument("--log", required=True, type=Path, help="skopeo output")
    args = parser.parse_args()
    assert DIGEST.search(args.published) and DIGEST.search(args.sglang_image), "references must carry digests"
    assert len(args.source_revision) == 40
    for path in (args.out, args.manifest_out):
        assert not path.exists(), path
    source = f"docker://{args.published}"
    target = f"docker://{args.registry}/{args.repository}:{args.tag}"
    with args.log.open("w") as log:
        code = subprocess.run(["docker", "run", "--rm", "--network=host", "quay.io/skopeo/stable:latest", "copy",
                               "--preserve-digests", "--dest-tls-verify=false", source, target],
                              stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"RETAIN_EXIT={code}\n")
    assert code == 0, f"skopeo copy failed ({code}); see {args.log}"
    head = subprocess.run(["curl", "-s", "-D", "-", "-o", str(args.manifest_out), "-H", "Accept: " + ACCEPT,
                           f"http://{args.registry}/v2/{args.repository}/manifests/{args.tag}"],
                          capture_output=True, text=True, check=True).stdout
    digest = re.search(r"(?i)docker-content-digest:\s*(sha256:[0-9a-f]{64})", head).group(1)
    manifest = json.loads(args.manifest_out.read_bytes())
    config = manifest["config"]["digest"]
    proof = {
        "images": [{"image": f"{args.registry}/{args.repository}@{digest}",
                    "worker_image": f"{args.worker_registry}/{args.repository}@{digest}", "image_id": config}],
        "manifest_media_type": manifest.get("mediaType"), "published": args.published,
        "registry_root": args.registry_root, "retained_at": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
        "sglang_image": args.sglang_image, "source_revision": args.source_revision,
        "worker_image": f"{args.worker_registry}/{args.repository}@{digest}", "worker_image_id": config,
    }
    args.out.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"manifest_digest": digest, "image_id": config, "media_type": manifest.get("mediaType"),
                      "layers": len(manifest["layers"]), "proof": str(args.out)}))


if __name__ == "__main__":
    main()
