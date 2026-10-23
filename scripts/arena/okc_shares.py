"""Write the weights stage's external-share file from Open Kernel Challenge crowns held by Cacheon miners.

Every miner enters the challenge (openkernelchallenge.dev) under their own GitHub account and claims a
board by publishing, from that same account, a public gist holding the exact code of their official
job beside ``cacheon-okc-claim.json``: ``{"board", "job", "hotkey", "code_sha256", "signature"}``. The
signature is the hotkey's over ``cacheon-okc-claim:v1:<board>:<job>:<github login>:<code_sha256>``, so
the gist proves the account and the signature the hotkey; the gists are the shared code miners build on.
A claim counts only while its job is still that account's row on the public board: superseding it
without republishing, a withdrawal, or an organizer removal stops it at the next refresh.

A board's crown passes to a claim only when it beats the holder's official result by the margin (shared
code must not win on judge noise), or when the holder's claim lapses. Until a crown exists on the serving
board, the four boards that count toward the challenge's serving priority pay a quarter of the share each;
then the serving board pays all of it. Unheld parts go to the burn hotkey in the weights stage. The output
file doubles as the crown state; see docs/miner-guide/open-kernel-challenge.md.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from cacheon.chain.weight_share import default_verify_fn

OKC = "https://openkernelchallenge.dev/api/v1"
# The challenge's decode-step problems: the boards its serving-queue priority ranks.
COUNTED = ("muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer")
CLAIM_FILE = "cacheon-okc-claim.json"


def get(url: str, *, api: bool = False) -> bytes:
    headers = {"User-Agent": "cacheon-okc-shares"}
    if api and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return response.read()


def board_rows(board: str) -> dict[str, tuple[str | None, float]]:
    """Each account's official row on one board: its job id and the board's score (higher is better)."""
    if board == "serving":
        rows = json.loads(get(f"{OKC}/serving/leaderboard"))["rows"]
        return {row["team"]: (row.get("job"), row["score"]) for row in rows
                if not row.get("reference") and isinstance(row.get("score"), (int, float))}
    rows = json.loads(get(f"{OKC}/leaderboard/{board}"))["rows"]
    return {row["team"]: (row["job"], row["speedup"]) for row in rows}


def claims(login: str, cache: dict) -> dict[str, dict]:
    """A login's verified claims by board, newest gist first; a malformed claim is skipped, not trusted."""
    if login not in cache:
        found = {}
        try:
            gists = json.loads(get(f"https://api.github.com/users/{login}/gists?per_page=100", api=True))
        except urllib.error.HTTPError as exc:
            if exc.code != 404:  # a renamed or deleted account has no claims; rate limits stay loud
                raise
            gists = []
        for gist in gists:
            files = gist["files"]
            if CLAIM_FILE not in files or gist["owner"]["login"] != login:
                continue
            try:
                claim = json.loads(get(files[CLAIM_FILE]["raw_url"]))
                board, job, hotkey, digest = (claim[key] for key in ("board", "job", "hotkey", "code_sha256"))
                if board in found or not any(name != CLAIM_FILE and hashlib.sha256(get(meta["raw_url"])).hexdigest()
                                             == digest for name, meta in files.items()):
                    continue
                message = f"cacheon-okc-claim:v1:{board}:{job}:{login}:{digest}".encode()
                if default_verify_fn(hotkey, message, bytes.fromhex(claim["signature"].removeprefix("0x"))):
                    found[board] = {"github": login, "job": job, "hotkey": hotkey, "gist": gist["html_url"]}
            except (KeyError, TypeError, ValueError, AttributeError, OSError):
                continue
        cache[login] = found
    return cache[login]


def crown(board: str, holder: dict | None, margin: float, cache: dict) -> dict | None:
    """Keep or pass one board's crown against the current public rows."""
    rows = board_rows(board)

    def current(login: str) -> dict | None:
        claim = claims(login, cache).get(board)
        job, value = rows.get(login, (None, None))
        if claim is None or value is None or (job is not None and claim["job"] != job):
            return None
        return {**claim, "value": value}

    held = current(holder["github"]) if holder else None
    for login, (_job, value) in sorted(rows.items(), key=lambda row: -row[1][1]):
        if held is not None and (login == held["github"] or value < held["value"] * (1 + margin)):
            break
        challenger = current(login)
        if challenger is not None:
            return {**challenger, "since": int(time.time())}
    return held and {**held, "since": holder["since"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="the stage's external_share_path (also the state)")
    parser.add_argument("--public", type=Path, help="also write the crowns and their gists here, for the dashboard")
    parser.add_argument("--kernel-margin", type=float, default=0.01, help="fraction a new kernel crown must clear")
    parser.add_argument("--serving-margin", type=float, default=0.02, help="fraction a new serving crown must clear")
    args = parser.parse_args()
    held = json.loads(args.out.read_text())["detail"]["crowns"] if args.out.exists() else {}
    cache: dict = {}
    crowns = {board: crown(board, held.get(board), args.serving_margin if board == "serving" else args.kernel_margin,
                           cache) for board in (*COUNTED, "serving")}
    paying = {"serving": 1_000_000} if crowns["serving"] else dict.fromkeys(COUNTED, 250_000)
    parts: dict[str, int] = {}
    for board, value in paying.items():
        if crowns[board]:
            parts[crowns[board]["hotkey"]] = parts.get(crowns[board]["hotkey"], 0) + value
    document = {"schema": "cacheon-external-share-v1",
                "shares": [{"hotkey": hotkey, "parts": value} for hotkey, value in sorted(parts.items())],
                "detail": {"crowns": crowns, "paying": sorted(paying), "refreshed": int(time.time())}}
    for path, body in ((args.out, document), (args.public, document["detail"])):
        if path is not None:
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
            temporary.chmod(0o644)
            os.replace(temporary, path)
    print(json.dumps(document["detail"], sort_keys=True))


if __name__ == "__main__":
    main()
