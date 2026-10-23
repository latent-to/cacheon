"""Write the weights stage's external-share file for the Open Kernel Challenge arena.

Cacheon miners enter the challenge (openkernelchallenge.dev) under their own accounts and also submit
their code to Cacheon with the ordinary unpaid ``chain-submit``: the bundle holds the exact file they
submitted to the challenge, and the committed URL ends in ``?okc=<board>:<job>``, naming the board and
their job on it. Unpaid commits are never evaluated; intake keeps them as reservations, which this reads.

Each refresh pays the share equally to every claim whose job is in its board's top 3 and whose bundle
fetched and matched its committed hash. The first such claim of a job, in chain order, owns it, and code
identical to an earlier claim's (ignoring whitespace and file names) pays only that earlier claimant, so
copies cannot take paid places. With no paid claim the whole share goes to the default hotkey. A bundle
that fails to fetch is not retried; its miner submits a new claim. Every claim's standing is public for
the dashboard; its code stays in the operator's private file until its track closes, then is public too.
See docs/miner-guide/open-kernel-challenge.md.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cacheon.chain.bundle_privacy import validator_key
from cacheon.chain.fetch import FetchError, fetch_bundle

OKC = "https://openkernelchallenge.dev/api/v1"
BOARDS = ("muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer",
          "muse_prefill_layer", "serving")
TOP = 3
WHOLE = 1_000_000
_JOB = re.compile(r"[A-Za-z0-9_-]{1,64}")
_MAX_FILE_BYTES = 1 << 20


def get(path: str):
    request = urllib.request.Request(f"{OKC}/{path}", headers={"User-Agent": "cacheon-okc-shares"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def track_closes() -> dict[str, float]:
    """Each board's track close time (unix seconds), as the challenge publishes it."""
    tracks = get("problems")["tracks"]
    closes = {name: datetime.fromisoformat(tracks[name]["closes"]).timestamp() for name in ("kernels", "serving")}
    return {board: closes["serving" if board == "serving" else "kernels"] for board in BOARDS}


def top_jobs(board: str) -> dict[str, dict]:
    """The board's top rows by job, best first, as the public leaderboard orders them."""
    rows = get("serving/leaderboard" if board == "serving" else f"leaderboard/{board}")["rows"]
    scored = [(row, row.get("score", row.get("speedup"))) for row in rows if row.get("job") and not row.get("reference")]
    scored = sorted(((row, value) for row, value in scored if isinstance(value, (int, float))), key=lambda item: -item[1])
    return {row["job"]: {"rank": rank, "team": row.get("team"), "value": value}
            for rank, (row, value) in enumerate(scored[:TOP], 1)}


def read_claims(intake_db: Path) -> list[dict]:
    """Every reservation whose URL names a board and job, in chain order."""
    db = sqlite3.connect(f"file:{intake_db}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT reservation_id, hotkey, content_hash, url, block FROM reservations "
            "WHERE url LIKE '%okc=%' ORDER BY block, event_index, event_subindex").fetchall()
    finally:
        db.close()
    claims = []
    for reservation_id, hotkey, content_hash, url, block in rows:
        values = parse_qs(urlparse(url).query).get("okc", [])
        board, _, job = values[0].partition(":") if len(values) == 1 else ("", "", "")
        if board in BOARDS and _JOB.fullmatch(job):
            claims.append({"reservation_id": reservation_id, "hotkey": hotkey, "content_hash": content_hash,
                           "url": url, "block": block, "board": board, "job": job})
    return claims


def bundle_files(claim: dict, cache: Path) -> dict[str, str]:
    """The claim's bundle as text by path; FetchError when it is unreachable or its hash differs."""
    root = fetch_bundle(claim["url"], claim["content_hash"], cache)
    return {path.relative_to(root).as_posix(): path.read_bytes()[:_MAX_FILE_BYTES].decode("utf-8", "replace")
            for path in sorted(root.rglob("*")) if path.is_file()}


def fingerprint(files: dict[str, str]) -> str:
    """Code identity for the copy rule: the files' contents without whitespace, names ignored."""
    return hashlib.sha256("\0".join(sorted(re.sub(r"\s+", "", text) for text in files.values())).encode()).hexdigest()


def write(path: Path, body: dict, mode: int = 0o644) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    temporary.chmod(mode)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-db", required=True, type=Path, help="the intake database miners submit into")
    parser.add_argument("--default-hotkey", required=True, help="paid the whole share when no claim is")
    parser.add_argument("--cache", required=True, type=Path, help="private directory for fetched bundles")
    parser.add_argument("--out", required=True, type=Path, help="the weights stage's external_share_path")
    parser.add_argument("--public", required=True, type=Path, help="claims, and code once its track closes")
    parser.add_argument("--private", required=True, type=Path, help="claims with their code, for the operator")
    args = parser.parse_args()
    validator_key()  # an unprovisioned key is the operator's fault: stop before judging any claim
    standing, closes = {board: top_jobs(board) for board in BOARDS}, track_closes()
    previous = json.loads(args.private.read_text())["claims"] if args.private.exists() else []
    failed = {claim["reservation_id"]: claim["status"] for claim in previous
              if claim["status"].startswith("code not fetched")}
    claims, owned, first = read_claims(args.intake_db), set(), {}
    for claim in claims:
        top = standing[claim["board"]].get(claim["job"])
        claim.update(top or {}, status="not in the top 3", files={})
        # Earlier claims already fetched still own their code even after leaving the top 3.
        if top is None and not (args.cache / claim["content_hash"]).is_dir():
            continue
        if claim["reservation_id"] in failed:
            claim["status"] = failed[claim["reservation_id"]]
            continue
        try:
            claim["files"] = bundle_files(claim, args.cache)
        except FetchError as exc:
            claim["status"] = f"code not fetched: {exc}"
            continue
        claim["fingerprint"] = fingerprint(claim["files"])
        owner = first.setdefault(claim["fingerprint"], claim)
        if owner["hotkey"] != claim["hotkey"]:
            claim["status"] = f"same code as earlier claim {owner['reservation_id']}"
        elif top is None:
            claim["status"] = "not in the top 3"
        elif (claim["board"], claim["job"]) in owned:
            claim["status"] = "job already claimed"
        else:
            owned.add((claim["board"], claim["job"]))
            claim["status"] = "paid"
    paid = [claim for claim in claims if claim["status"] == "paid"]
    parts: dict[str, int] = {}
    if paid:
        base, extra = divmod(WHOLE, len(paid))
        for index, claim in enumerate(paid):
            claim["parts"] = base + (index < extra)
            parts[claim["hotkey"]] = parts.get(claim["hotkey"], 0) + claim["parts"]
    else:
        parts[args.default_hotkey] = WHOLE
    refreshed = int(time.time())
    write(args.out, {"schema": "cacheon-external-share-v1",
                     "shares": [{"hotkey": hotkey, "parts": value} for hotkey, value in sorted(parts.items())],
                     "detail": {"refreshed": refreshed, "paid": [claim["reservation_id"] for claim in paid]}})
    summary = {"refreshed": refreshed, "default_hotkey": args.default_hotkey, "top": standing, "shares": parts}
    write(args.private, {**summary, "claims": claims}, 0o600)
    public = [{**claim, "files": claim["files"] if refreshed >= closes[claim["board"]] else {},
               "code_public_after": int(closes[claim["board"]])} for claim in claims]
    write(args.public, {**summary, "claims": public})
    print(json.dumps({"refreshed": refreshed, "claims": len(claims), "paid": len(paid), "shares": parts},
                     sort_keys=True))


if __name__ == "__main__":
    main()
