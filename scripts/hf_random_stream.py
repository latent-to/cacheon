#!/usr/bin/env python3
"""
HF Dataset Explorer — Randomly discover and stream datasets from Hugging Face.

Usage:
    python hf_random_stream.py              # Random dataset, show 5 rows
    python hf_random_stream.py -n 10        # Random dataset, show 10 rows
    python hf_random_stream.py -t text      # Random dataset tagged with "text"
    python hf_random_stream.py -s           # Continuous stream mode (new random dataset every round)
    python hf_random_stream.py --search code # Search for datasets matching "code"
"""

import argparse
import json
import random
import sys
import textwrap
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


API_BASE = "https://huggingface.co/api"
VIEWER_BASE = "https://datasets-server.huggingface.co"


def fetch_json(url: str, timeout: int = 15) -> dict | list | None:
    """Fetch JSON from a URL, return None on failure."""
    try:
        req = Request(url, headers={"User-Agent": "hf-random-stream/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"  ⚠ Request failed: {e}", file=sys.stderr)
        return None


MIN_DOWNLOADS = 50_000
MIN_LIKES = 10


def get_random_datasets(limit: int = 500, tag: str | None = None, search: str | None = None) -> list[dict]:
    """Fetch a batch of high-quality datasets from the HF Hub API.

    Sorts by downloads descending and samples from the top tier so results
    are large, well-maintained datasets rather than obscure one-offs.
    """
    # Stay within the top-downloaded datasets; randomise within that window
    # so we don't always return the same handful of famous datasets.
    offset = random.randint(0, 500)
    url = f"{API_BASE}/datasets?limit={limit}&offset={offset}&sort=downloads&direction=-1"
    if tag:
        url += f"&tags=task_categories:{tag}"
    if search:
        url += f"&search={search}"
    data = fetch_json(url)
    if not isinstance(data, list):
        return []

    # Keep only datasets that clear the quality bar
    filtered = [
        ds for ds in data
        if ds.get("downloads", 0) >= MIN_DOWNLOADS and ds.get("likes", 0) >= MIN_LIKES
    ]
    # Fall back to the full list if the filter is too aggressive (e.g. niche tag)
    return filtered if filtered else data


def get_dataset_info(dataset_id: str) -> dict | None:
    """Fetch metadata for a specific dataset."""
    url = f"{API_BASE}/datasets/{dataset_id}"
    return fetch_json(url)


def get_dataset_splits(dataset_id: str) -> dict | None:
    """Get available configs and splits from the dataset viewer API."""
    url = f"{VIEWER_BASE}/splits?dataset={dataset_id}"
    return fetch_json(url)


def get_rows(dataset_id: str, config: str, split: str, offset: int = 0, length: int = 5) -> dict | None:
    """Fetch rows from a dataset via the viewer API."""
    url = (
        f"{VIEWER_BASE}/rows?dataset={dataset_id}"
        f"&config={config}&split={split}&offset={offset}&length={length}"
    )
    return fetch_json(url)


def format_value(val, max_len: int = 120) -> str:
    """Format a cell value for display."""
    if val is None:
        return "null"
    if isinstance(val, str):
        val = val.replace("\n", " ").strip()
        if len(val) > max_len:
            return val[:max_len] + "…"
        return val
    if isinstance(val, dict):
        # Could be an image reference, audio, etc.
        if "src" in val:
            return f"[media: {val['src'][:60]}…]"
        s = json.dumps(val, ensure_ascii=False)
        if len(s) > max_len:
            return s[:max_len] + "…"
        return s
    if isinstance(val, list):
        s = json.dumps(val, ensure_ascii=False)
        if len(s) > max_len:
            return s[:max_len] + "…"
        return s
    return str(val)


def print_header(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_dataset_meta(info: dict):
    """Print dataset metadata."""
    dataset_id = info.get("id", "unknown")
    desc = info.get("description", "No description available.")
    downloads = info.get("downloads", "?")
    likes = info.get("likes", "?")
    tags = info.get("tags", [])
    license_tag = next((t.split(":")[-1] for t in tags if t.startswith("license:")), "unknown")
    task_tags = [t.split(":")[-1] for t in tags if t.startswith("task_categories:")]

    print(f"  📦 Dataset:   {dataset_id}")
    print(f"  🔗 URL:       https://huggingface.co/datasets/{dataset_id}")
    print(f"  📥 Downloads: {downloads:,}" if isinstance(downloads, int) else f"  📥 Downloads: {downloads}")
    print(f"  ❤️  Likes:     {likes}")
    print(f"  📜 License:   {license_tag}")
    if task_tags:
        print(f"  🏷️  Tasks:     {', '.join(task_tags)}")

    # Truncate description
    if desc:
        desc_clean = desc.replace("\n", " ").strip()
        wrapped = textwrap.fill(desc_clean[:300], width=66, initial_indent="  ", subsequent_indent="  ")
        print(f"\n  📝 Description:")
        print(wrapped)
        if len(desc_clean) > 300:
            print("  ...")


def show_rows(dataset_id: str, num_rows: int = 5) -> bool:
    """Fetch and display rows from a random dataset. Returns True on success."""
    # Get splits
    splits_data = get_dataset_splits(dataset_id)
    if not splits_data or "splits" not in splits_data or not splits_data["splits"]:
        print("  ⚠ Could not retrieve splits (dataset may be gated or private).")
        return False

    # Pick first available split (usually train)
    split_info = splits_data["splits"][0]
    config = split_info["config"]
    split = split_info["split"]

    print(f"\n  📂 Config: {config} | Split: {split}")

    # Fetch rows
    rows_data = get_rows(dataset_id, config, split, offset=0, length=num_rows)
    if not rows_data or "rows" not in rows_data:
        print("  ⚠ Could not fetch rows.")
        return False

    rows = rows_data["rows"]
    if not rows:
        print("  ⚠ Dataset appears empty.")
        return False

    # Get column names from features or first row
    features = rows_data.get("features", [])
    columns = [f["name"] for f in features] if features else list(rows[0].get("row", {}).keys())

    print(f"  📊 Columns: {', '.join(columns)}")
    print(f"  📋 Showing {len(rows)} row(s):\n")

    # Print rows
    for i, row_obj in enumerate(rows):
        row = row_obj.get("row", {})
        print(f"  ── Row {i} {'─'*55}")
        for col in columns:
            val = row.get(col)
            formatted = format_value(val)
            label = f"    {col}:"
            if len(formatted) > 100:
                print(f"{label}")
                print(f"      {formatted}")
            else:
                print(f"{label} {formatted}")
    print()
    return True


def explore_random(num_rows: int = 5, tag: str | None = None, search: str | None = None) -> bool:
    """Pick a random dataset and display it. Returns True on success."""
    print("\n  🎲 Rolling the dice...")

    datasets = get_random_datasets(limit=200, tag=tag, search=search)
    if not datasets:
        print("  ⚠ Could not fetch dataset list from HF.")
        return False

    # Shuffle and try up to 10 datasets (some may be gated/broken)
    random.shuffle(datasets)
    for ds in datasets[:10]:
        dataset_id = ds.get("id")
        if not dataset_id:
            continue

        dl = ds.get("downloads", "?")
        likes = ds.get("likes", "?")
        dl_str = f"{dl:,}" if isinstance(dl, int) else str(dl)
        print_header(f"🎰  Random Dataset  |  📥 {dl_str} downloads  ❤️  {likes} likes")

        # Fetch full metadata
        info = get_dataset_info(dataset_id)
        if info:
            print_dataset_meta(info)
        else:
            print(f"  📦 Dataset: {dataset_id}")

        if show_rows(dataset_id, num_rows):
            return True
        else:
            print("  ↻ Trying another dataset...\n")

    print("  ✗ Couldn't find a viewable dataset after 10 attempts. Try again!")
    return False


def stream_mode(num_rows: int = 5, tag: str | None = None, search: str | None = None):
    """Continuous stream — press Enter for a new random dataset, q to quit."""
    print_header("🌊  HF Dataset Stream Mode")
    print("  Press Enter for a new random dataset, 'q' to quit.\n")

    while True:
        explore_random(num_rows=num_rows, tag=tag, search=search)
        try:
            user_in = input("  ⏎ Enter for next, 'q' to quit: ").strip().lower()
            if user_in == "q":
                print("\n  👋 Done exploring. Happy building!\n")
                break
        except (KeyboardInterrupt, EOFError):
            print("\n\n  👋 Bye!\n")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Randomly discover and stream datasets from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python hf_random_stream.py              # Random dataset, 5 rows
              python hf_random_stream.py -n 10        # Show 10 rows
              python hf_random_stream.py -t text-classification  # Filter by task tag
              python hf_random_stream.py --search medical        # Search for 'medical'
              python hf_random_stream.py -s            # Stream mode (continuous)
        """)
    )
    parser.add_argument("-n", "--num-rows", type=int, default=5, help="Number of rows to display (default: 5)")
    parser.add_argument("-t", "--tag", type=str, default=None,
                        help="Filter by task category tag (e.g. text-classification, question-answering)")
    parser.add_argument("--search", type=str, default=None, help="Search datasets by keyword")
    parser.add_argument("-s", "--stream", action="store_true", help="Continuous stream mode")

    args = parser.parse_args()

    if args.stream:
        stream_mode(num_rows=args.num_rows, tag=args.tag, search=args.search)
    else:
        explore_random(num_rows=args.num_rows, tag=args.tag, search=args.search)


if __name__ == "__main__":
    main()