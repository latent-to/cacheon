"""Write the weights stage's external-share file for the Open Kernel Challenge arena.

Cacheon miners enter the challenge (openkernelchallenge.dev) under their own accounts and also submit
their code to Cacheon with the ordinary unpaid ``chain-submit``: the bundle holds the exact file they
submitted to the challenge, and the committed URL ends in ``?okc=<board>:<job>``, naming the board and
their job on it, plus ``&okc_user_block=<registration block>``. The exact OKC username must be bound
one-to-one to the claiming hotkey by its oldest accepted on-chain registration in the round. Unpaid commits are
never evaluated; intake keeps them as reservations, which this reads.

Each refresh pays the share equally to every claim whose job is in its board's top 3 and whose bundle
fetched and matched its committed hash. The first such claim of a job, in chain order, owns it, and code
identical to an earlier claim's (ignoring whitespace and file names) pays only that earlier claimant, so
copies cannot take paid places. With no paid claim the whole share goes to the default hotkey. A bundle
that fails to fetch is not retried; its miner submits a new claim. Every claim's standing is public for
the dashboard; its code stays in the operator's private file until its track closes, then is public too.
See docs/miner-guide/open-kernel-challenge.md.
"""

import argparse
import fcntl
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

from cacheon import chain
from cacheon.chain.bundle_privacy import validator_key
from cacheon.chain.fetch import FetchError, fetch_bundle
from cacheon.chain.okc_history import sync_registrations
from cacheon.chain.okc_username import InvalidRegistration, check_claim_registration

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


def read_claims(intake_db: Path, start_block: int) -> list[dict]:
    """Read round claims with an in-round registration pointer, in chain order."""
    db = sqlite3.connect(f"file:{intake_db}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT reservation_id, hotkey, content_hash, url, block FROM reservations "
            "WHERE url LIKE '%okc=%' AND block >= ? ORDER BY block, event_index, event_subindex",
            (start_block,)).fetchall()
    finally:
        db.close()
    claims = []
    for reservation_id, hotkey, content_hash, url, block in rows:
        query = parse_qs(urlparse(url).query, keep_blank_values=True)
        values = query.get("okc", [])
        board, _, job = values[0].partition(":") if len(values) == 1 else ("", "", "")
        registration = query.get("okc_user_block", [])
        user_block = (int(registration[0]) if len(registration) == 1
                      and re.fullmatch(r"[1-9][0-9]{0,15}", registration[0]) else None)
        if board in BOARDS and _JOB.fullmatch(job) and user_block is not None and user_block >= start_block:
            claims.append({"reservation_id": reservation_id, "hotkey": hotkey, "content_hash": content_hash,
                           "url": url, "block": block, "board": board, "job": job,
                           "okc_user_block": user_block})
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
    parser.add_argument("--network", required=True, help="archive chain endpoint used by intake")
    parser.add_argument("--netuid", required=True, type=int, help="subnet used by intake and username registrations")
    parser.add_argument("--start-block", required=True, type=int,
                        help="first included finalized block of this fresh OKC round")
    parser.add_argument("--username-db", required=True, type=Path, help="persistent oldest-username history index")
    parser.add_argument("--history-batch-blocks", type=int, default=5000,
                        help="maximum finalized history blocks indexed per run; no shares until caught up")
    args = parser.parse_args()
    with args.private.with_name(args.private.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = chain.connect(args.network)
        try:
            refresh(args, st)
        finally:
            st.close()


def refresh(args, st) -> None:
    validator_key()  # an unprovisioned key is the operator's fault: stop before judging any claim
    finalized, owners = sync_registrations(st, args.netuid, args.username_db,
                                          max_blocks=args.history_batch_blocks, start_block=args.start_block)
    standing, closes = {board: top_jobs(board) for board in BOARDS}, track_closes()
    previous = json.loads(args.private.read_text())["claims"] if args.private.exists() else []
    previous_by_id = {claim["reservation_id"]: claim for claim in previous}
    failed = {claim["reservation_id"]: claim.get("fetch_error", claim["status"]) for claim in previous
              if claim.get("fetch_error") or claim["status"].startswith("code not fetched")}
    claims, owned, first = read_claims(args.intake_db, args.start_block), set(), {}
    for claim in claims:
        top = standing[claim["board"]].get(claim["job"])
        claim.update(top or {}, status="not in the top 3", files={})
        if claim["reservation_id"] in failed:
            claim["fetch_error"] = failed[claim["reservation_id"]]
        if claim["block"] > finalized:
            raise RuntimeError("claim intake is ahead of the finalized username history")
        earlier = previous_by_id.get(claim["reservation_id"], {})
        verified_team = earlier.get("verified_team") or (earlier.get("team") if earlier.get("username_verified") else None)
        if top is None:
            if verified_team is None:
                continue
            claim["team"] = verified_team
        try:
            owner = check_claim_registration(st, args.netuid, {**claim, "team": verified_team or claim.get("team")},
                                             owners.get(claim["hotkey"]), finalized)
        except InvalidRegistration as exc:
            claim["status"] = f"username rejected: {exc}"
            continue
        username_matches = claim.get("team") == owner["username"]
        claim.update(username=owner["username"], username_block=owner["block"],
                     verified_team=owner["username"], username_verified=username_matches)
        if not username_matches:
            claim["status"] = "username rejected: OKC team does not exactly match the hotkey's committed username"
        if claim.get("fetch_error"):
            if username_matches:
                claim["status"] = claim["fetch_error"]
            continue
        # Earlier claims already fetched still own their code even after leaving the top 3.
        if (top is None or not username_matches) and not (args.cache / claim["content_hash"]).is_dir():
            continue
        try:
            claim["files"] = bundle_files(claim, args.cache)
        except FetchError as exc:
            claim["status"] = f"code not fetched: {exc}"
            claim["fetch_error"] = claim["status"]
            continue
        claim["fingerprint"] = fingerprint(claim["files"])
        owner = first.setdefault(claim["fingerprint"], claim)
        if not username_matches:
            continue
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
                     "detail": {"refreshed": refreshed, "start_block": args.start_block,
                                "paid": [claim["reservation_id"] for claim in paid]}})
    summary = {"refreshed": refreshed, "start_block": args.start_block,
               "default_hotkey": args.default_hotkey, "top": standing, "shares": parts}
    write(args.private, {**summary, "claims": claims}, 0o600)
    public = [{**claim, "files": claim["files"] if refreshed >= closes[claim["board"]] else {},
               "code_public_after": int(closes[claim["board"]])} for claim in claims]
    write(args.public, {**summary, "claims": public})
    print(json.dumps({"refreshed": refreshed, "claims": len(claims), "paid": len(paid), "shares": parts},
                     sort_keys=True))


if __name__ == "__main__":
    main()
