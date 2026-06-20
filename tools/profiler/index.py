#!/usr/bin/env python3
"""Build a static profiler run registry.

The registry is intentionally boring: one self-contained ``index.html`` with a
table of generated profiler datasets and links to their reports. It does not
require a web server or JavaScript dependencies.

    python3 tools/profiler/index.py profiler_runs/ -o profiler_index/
    python3 tools/profiler/index.py profiler_runs/ -o profiler_index/ --build-missing
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_dataset(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
        if "dataset" in data and isinstance(data["dataset"], dict):
            data = data["dataset"]
        return data
    except (OSError, ValueError):
        return None


def _datasets(root: Path) -> list[tuple[Path, dict]]:
    out = []
    for p in sorted(root.rglob("dataset.json")):
        if ".profiler_cache" in p.parts or any(part.startswith("profiler_index") for part in p.parts):
            continue
        ds = _load_dataset(p)
        if ds:
            out.append((p, ds))
    return out


def _maybe_build_missing(root: Path, outdir: Path) -> None:
    import build as build_mod  # local import keeps normal index scans cheap

    existing = {p.parent.resolve() for p, _ in _datasets(root)}
    candidates = [root] + [p for p in sorted(root.iterdir()) if p.is_dir()]
    for d in candidates:
        if d.resolve() in existing:
            continue
        names = {p.name for p in d.iterdir() if p.is_file()}
        if not any(n.endswith((".trace.json.gz", "_kernsum.txt", "_raw.csv", "_details.txt", ".sqlite")) or
                   n.startswith(("e2e_", "ceil_")) for n in names):
            continue
        target = outdir / "runs" / d.name
        build_mod.build(d, target)


def _row(path: Path, ds: dict) -> dict:
    f = ds.get("findings", {})
    peak = f.get("peak") or {}
    a = f.get("amdahl") or {}
    model = ds.get("model") or {}
    runtime = ds.get("runtime") or {}
    report = path.parent / "report.html"
    return {
        "dataset": path,
        "report": report if report.exists() else None,
        "run": (ds.get("run") or {}).get("name") or path.parent.name,
        "model": model.get("id") or "unknown",
        "family": model.get("family") or "unknown",
        "decoder": model.get("decoder_type") or "unknown",
        "attention": model.get("attention") or "unknown",
        "engine": runtime.get("engine") or "unknown",
        "graph": runtime.get("graph_mode") or "unknown",
        "peak": peak.get("tok_s"),
        "winnable": a.get("winnable_pct"),
        "rewrite": a.get("rewritable_pct"),
        "floor": a.get("floor_pct"),
        "unknown": a.get("unknown_pct"),
        "headline": f.get("headline") or "",
    }


def render(rows: list[dict]) -> str:
    def esc(x):
        return html.escape("" if x is None else str(x))

    def num(x, digits: int = 1):
        if x is None:
            return ""
        return f"{float(x):.{digits}f}"

    body = []
    for r in rows:
        report = f'<a href="{esc(r["report"].resolve().as_uri())}">report</a>' if r.get("report") else "report missing"
        dataset = f'<a href="{esc(r["dataset"].resolve().as_uri())}">dataset</a>'
        body.append(
            "<tr>"
            f"<td>{report} · {dataset}</td>"
            f"<td>{esc(r['run'])}</td>"
            f"<td>{esc(r['model'])}<div class=muted>{esc(r['family'])}</div></td>"
            f"<td>{esc(r['decoder'])}<div class=muted>{esc(r['attention'])}</div></td>"
            f"<td>{esc(r['engine'])}<div class=muted>graphs {esc(r['graph'])}</div></td>"
            f"<td class=num>{num(r['peak'], 0)}</td>"
            f"<td class=num>{num(r['rewrite'], 1)}</td>"
            f"<td class=num>{num(r['winnable'], 1)}</td>"
            f"<td class=num>{num(r['floor'], 1)}</td>"
            f"<td class=num>{num(r['unknown'], 1)}</td>"
            f"<td class=headline>{esc(r['headline'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset=utf-8><title>Profiler Runs</title>
<style>
body{{margin:0;background:#0d1117;color:#e6edf3;font:14px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif}}
header{{padding:20px 26px;border-bottom:1px solid #30363d}} h1{{font-size:20px;margin:0}}
main{{padding:18px 26px}} table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{padding:7px 9px;border-bottom:1px solid #30363d;text-align:left;vertical-align:top}}
th{{color:#8b949e;position:sticky;top:0;background:#0d1117}} a{{color:#58a6ff;text-decoration:none}}
.num{{text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}} .muted{{color:#8b949e;font-size:11.5px}}
.headline{{max-width:460px;color:#8b949e}}
</style></head><body><header><h1>Profiler Runs</h1><div class=muted>{len(rows)} dataset(s)</div></header>
<main><table><tr><th>links</th><th>run</th><th>model</th><th>architecture</th><th>runtime</th>
<th class=num>peak tok/s</th><th class=num>rewrite%</th><th class=num>fuse%</th><th class=num>floor%</th><th class=num>unknown%</th><th>headline</th></tr>
{''.join(body)}
</table></main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("profiler_index"))
    ap.add_argument("--build-missing", action="store_true", help="build direct child profile dirs that do not already have dataset.json")
    args = ap.parse_args()
    root = args.root.expanduser()
    outdir = args.outdir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.build_missing:
        _maybe_build_missing(root, outdir)
    rows = [_row(path, ds) for path, ds in _datasets(root) + _datasets(outdir / "runs")]
    (outdir / "index.html").write_text(render(rows))
    (outdir / "index.json").write_text(json.dumps([
        {k: str(v) if isinstance(v, Path) else v for k, v in r.items()} for r in rows
    ], indent=2))
    print(f"wrote {outdir/'index.html'} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
