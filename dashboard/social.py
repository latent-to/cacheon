"""Submission share metadata from the dashboard's retained evaluation readers."""

from contextlib import closing
from dataclasses import dataclass
from html import escape
import json
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from dashboard.forensics import retained_speed
from dashboard.sources import selected
from dashboard.winners import candidate_measurement, result_summary


@dataclass(frozen=True)
class SubmissionCard:
    """Public display fields; stock numbers belong to the same retained attempt."""

    reservation: str
    model: str
    target: str
    status: str
    gain: float | None = None
    submission: float | None = None
    stock: float | None = None
    metric: str = "OUTPUT THROUGHPUT"
    commit: str = ""

    @property
    def description(self):
        """Text alternative for crawlers and screen readers."""
        gain = (f"{self.gain:+.2f}% over stock SGLang" if self.gain is not None
                else "Stock comparison unavailable")
        rate = lambda n: f"{n:,.1f} tok/s" if n is not None else "unavailable"
        runtime = f"SGLang @ {self.commit[:7]}" if self.commit else "SGLang commit unavailable"
        return (f"{self.model} · {self.target} · {self.status}. {gain}. "
                f"{self.metric.title()}: submission {rate(self.submission)}; "
                f"stock SGLang {rate(self.stock)}. {runtime}.")


def submission_card(reservation, api):
    """Bind the displayed score and rates to one retained qualification."""
    detail = api["submission_detail"](reservation, Response())
    rid, target = detail["reservation_id"], detail["target_id"]
    fields = dict(reservation=rid, model=detail["competition"], target=target,
                  status=" · ".join(filter(None, (detail["status"], detail["decision"]))).upper())
    with closing(api["intake_conn"]()) as con:
        rows = con.execute("SELECT qualification_json, attempt_ref_json FROM settlement_qualifications "
                           "WHERE reservation_id=? ORDER BY reproduction_index", (rid,)).fetchall()
        if not rows:
            result = detail.get("result")
            return SubmissionCard(**fields, submission=result["decode_tps"][1] if result else detail.get("tokens_per_second"),
                                  metric="DECODE THROUGHPUT" if result else "OUTPUT THROUGHPUT")
        # Historical paired qualifications retain the lower accepted score.
        row = min(rows, key=lambda r: float(json.loads(r["qualification_json"])["speedup"]))
        qualification = json.loads(row["qualification_json"])
        speed = retained_speed(row["attempt_ref_json"], api["evidence_roots"](con), target)
    incumbent = qualification.get("incumbent_manifest") or {}
    runtime = (qualification.get("candidate_manifest") or {}).get("runtime_digest")
    stock = bool(runtime) and incumbent.get("entries") == {} and incumbent.get("runtime_digest") == runtime
    source = selected.get()
    commit = source.values.get("SGLANG_COMMITS", {}).get(runtime, "") if source else ""
    fields.update(commit=commit, gain=(float(qualification["speedup"]) - 1) * 100 if stock else None)
    if speed:
        result = result_summary(speed)
        if result:
            baseline, candidate = result["decode_tps"]
            fields.update(metric="DECODE THROUGHPUT", submission=candidate, stock=baseline if stock else None)
        else:
            baseline = [lane["tokens_per_second"] for lane in speed["lanes"]
                        if lane["role"] in ("B", "B_prime", "B_double_prime")
                        and lane.get("tokens_per_second") is not None]
            fields.update(submission=candidate_measurement([speed])["tokens_per_second"],
                          stock=max(baseline) if stock and baseline else None)
    return SubmissionCard(**fields)


def submission_page(request: Request, api):
    """Serve crawler-visible metadata in the same HTML used by the interactive app."""
    path = api["STATIC_DIR"] / "index.html"
    reservation = request.query_params.get("submission")
    if not reservation:
        return FileResponse(path)
    card = submission_card(reservation, api)
    source = selected.get()
    slug = source.public()["slug"] if source else ""
    canonical = "https://dash.cacheon.ai/" + slug + "?" + urlencode({"submission": card.reservation})
    query = "?" + urlencode({"arena": source.key}) if source else ""
    image = f"https://dash.cacheon.ai/api/submissions/{card.reservation}/preview.png{query}"
    title = f"Cacheon · {card.model} · {card.target}"
    tags = {"og:type": "website", "og:site_name": "Cacheon", "og:title": title,
            "og:description": card.description, "og:url": canonical, "og:image": image,
            "og:image:type": "image/png", "og:image:width": "1200", "og:image:height": "630",
            "og:image:alt": card.description, "twitter:card": "summary_large_image",
            "twitter:title": title, "twitter:description": card.description,
            "twitter:image": image, "twitter:image:alt": card.description}
    markup = "\n".join(f'<meta {"name" if k.startswith("twitter:") else "property"}="{k}" '
                       f'content="{escape(v, quote=True)}"/>' for k, v in tags.items())
    return HTMLResponse(path.read_text().replace("</head>", markup + "\n</head>"),
                        headers={"Cache-Control": "no-store"})


def install_social(app, api):
    """Register PNGs alongside the existing source-scoped submission API."""
    from dashboard.social_image import render_card

    @app.get("/api/submissions/{reservation_id}/preview.png", include_in_schema=False)
    def preview(reservation_id: str):
        card = submission_card(reservation_id, api)
        return Response(render_card(card), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=60"})
