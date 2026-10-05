"""Render Cacheon's branded 1200×630 submission cards without a browser."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from dashboard.social import SubmissionCard

ASSETS = Path(__file__).parent / "static"
FG, MUTED, ACCENT, LINE = "#f4f8f7", "#9aaaa7", "#30d3be", "#193b36"


@lru_cache(maxsize=128)
def render_card(card: SubmissionCard) -> bytes:
    """Cache only fully specified cards so a new evaluation changes the image."""
    image = Image.new("RGB", (1200, 630), "#030a09")
    draw = ImageDraw.Draw(image)

    def text(x, y, value, size=24, color=FG, width=1092, anchor=None):
        font = ImageFont.truetype(str(ASSETS / "Ubuntu.ttf"), size)
        while draw.textlength(value, font=font) > width and size > 12:
            size -= 1
            font = ImageFont.truetype(str(ASSETS / "Ubuntu.ttf"), size)
        if anchor == "mm":
            left, top, right, bottom = draw.textbbox((0, 0), value, font=font)
            x, y, anchor = x - (left + right) / 2, y - (top + bottom) / 2, None
        draw.text((x, y), value, font=font, fill=color, anchor=anchor)

    draw.rectangle((0, 0, 1199, 7), fill=ACCENT)
    with Image.open(ASSETS / "icon-192.png") as original:
        logo = original.convert("RGBA").resize((44, 44), Image.Resampling.LANCZOS)
    image.paste(logo, (54, 36), logo)
    text(110, 35, "Cacheon", 34)
    text(280, 46, "COMPETITIVE INFERENCE OPTIMIZATION", 19, MUTED, width=650)
    draw.rounded_rectangle((958, 38, 1146, 78), 20, fill="#0c302a")
    text(1052, 60, card.status, 18, ACCENT, width=148, anchor="mm")
    text(54, 118, card.model, 53)
    text(57, 186, "TARGET  /  " + card.target.replace("_", " ").upper(), 21, MUTED)
    draw.line((54, 242, 1146, 242), fill=LINE)
    def ttft(value):
        return f"TTFT {value * 1000:,.1f} ms" if value is not None else "TTFT unavailable"

    if card.comparison:
        text(600, 328, f"{card.gain:+.2f}%", 106, ACCENT, width=1000, anchor="mm")
        text(600, 401, "throughput improvement over stock SGLang", 23, anchor="mm")
        for x, label, value, latency in ((325, "SUBMISSION", card.submission, card.ttft),
                                         (875, "STOCK SGLANG", card.stock, card.stock_ttft)):
            text(x, 451, label + " · " + card.metric, 16, MUTED, width=510, anchor="mm")
            text(x, 483, f"{value:,.1f} tok/s · {ttft(latency)}", 22, width=510, anchor="mm")
        if card.stock_reference_date:
            text(600, 515, f"Stock reference · {card.stock_reference_date} · separate runs", 16, MUTED, anchor="mm")
    else:
        text(600, 292, card.metric, 21, MUTED, anchor="mm")
        if card.submission is not None:
            text(600, 377, f"{card.submission:,.1f}", 106, ACCENT, width=1000, anchor="mm")
            text(600, 452, "tok/s", 25, MUTED, anchor="mm")
        else:
            text(600, 377, "Awaiting measurement", 42, MUTED, anchor="mm")
        text(600, 491, ttft(card.ttft), 22, MUTED, anchor="mm")
    draw.line((54, 531, 1146, 531), fill=LINE)
    text(54, 560, card.runtime_label, 22, MUTED, width=620)
    text(720, 563, "SUBMISSION  " + card.reservation[:12], 19, MUTED)
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
