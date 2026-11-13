"""Visual review of the selected cross-gap pairs (the slide's "visually validate"
step). For each scored clip in an eval JSON, lay out the GT pair and the SR pair
with the tracked box drawn, a short GT "pose strip" of the frames either side of
each selected frame (so the reviewer can see the pose is stable, not a fluke
frame), the exact object crops the scorers compared, and a caption of the
per-metric c_gt / c_sr / s_M, fidelity, the temporal separation, and the
reliability breakdown. Pairs that fail a clarity gate (motion blur, object on the
frame edge, weak anchor) are routed to an "Excluded" appendix with the reason, so
the headline gallery stays clean. Pure image work (cv2 + PIL): no torch.

    python -m scripts.lr_vcc.adr.export_pairs --json out.json \
        --gt-dir data/starter --sr-dir data/flashvsr_results_flat \
        --ann-dir data/starter/annotations --out-dir data/review/flashvsr
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .annotations import load_annotations
from .crops import bbox_crop

STRIP_RADIUS = 1          # GT frames shown either side of each selected frame
_QUALITY_FLOOR = 0.6      # edge / anchor factor below this excludes the pair
_SIZE_FLOOR = 0.35        # size factor (area/BIG_AREA_FRAC) below this = too tiny to review
DIVERGE_BAND = 0.90       # consistency below this = SR changed the GT cross-gap relationship
PCT_GAIN_MIN_CGT = 0.05   # below this GT self-similarity is in the noise -> %gain is uninformative


def consistency_band(cons: float, threshold: float = DIVERGE_BAND) -> str:
    """Presentation band: 'diverges' when the SR cross-gap relationship departs from
    GT's (lower consistency), else 'preserves'."""
    return "diverges" if cons < threshold else "preserves"


def pct_gain(c_gt: float, c_sr: float, min_cgt: float = PCT_GAIN_MIN_CGT):
    """Percentage change of the SR's cross-gap self-similarity relative to the GT's,
    per scorer: ``(c_sr - c_gt) / c_gt * 100``. Negative = SR less consistent than GT
    (drift); positive = SR *more* self-similar than GT (which, with real GT, usually
    means fabricated over-consistency, not an improvement).

    Reference-only: it is handed to a reader alongside the raw ``c_sr`` but is NOT the
    operative metric -- the experiments rank on the GT-anchored ``s_M = 1 - |c_sr -
    c_gt|``, because %gain is (a) unbounded and unstable as ``c_gt`` -> 0 and (b) signs
    over-consistency as a gain. Returns ``None`` (render as 'n/a') when ``c_gt`` is in
    the noise (e.g. SIFT on a low-texture object: leopard/fox ``c_gt ~ 0``), where the
    ratio is meaningless."""
    if c_gt is None or c_sr is None:
        return None
    if abs(c_gt) < min_cgt:
        return None
    return (c_sr - c_gt) / c_gt * 100.0


def exclusion_reasons(factors: dict) -> list[str]:
    """Human-readable reasons a pair should be routed to the appendix, from its
    reliability factors. The gates are the signals that proved reliable on real
    review: object on the frame edge, a weak GT anchor (the GT pair itself barely
    matches), and a too-tiny object. Laplacian *sharpness* is deliberately NOT a
    gate -- it proved an unreliable proxy for perceived motion blur (the crops the
    reviewer called blurry had the highest variance), so motion blur stays an
    eye-call (the slide's 'visually validate' step)."""
    reasons = []
    if factors.get("edge", 1.0) < _QUALITY_FLOOR:
        reasons.append("object on frame edge")
    if factors.get("anchor", 1.0) < _QUALITY_FLOOR:
        reasons.append("weak pose/anchor")
    if factors.get("size", 1.0) < _SIZE_FLOOR:
        reasons.append("object too small")
    return reasons


def strip_indices(center: int, lo: int, hi: int, radius: int = STRIP_RADIUS) -> list[int]:
    """Frame indices for the pose strip around ``center``, clamped to [lo, hi]."""
    return list(range(max(lo, center - radius), min(hi, center + radius) + 1))


def grab_frames(video_path: str, indices) -> dict:
    """Decode ``video_path`` once and return {idx: RGB frame} for the wanted indices
    (sequential decode matches the eval's decode order exactly)."""
    want = sorted({int(i) for i in indices})
    out: dict[int, np.ndarray] = {}
    if not want:
        return out
    cap = cv2.VideoCapture(str(video_path))
    try:
        wanted, last, pos = set(want), want[-1], 0
        while pos <= last:
            if not cap.grab():
                break
            if pos in wanted:
                ok, bgr = cap.retrieve()
                if ok:
                    out[pos] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pos += 1
    finally:
        cap.release()
    return out


def draw_box(img: np.ndarray, box, color=(0, 220, 0), width=4) -> np.ndarray:
    """Draw an (x, y, w, h) box on a copy of an RGB frame."""
    x, y, w, h = (int(round(v)) for v in box)
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, width)
    return out


def _fit(img: np.ndarray, w: int, h: int) -> Image.Image:
    im = Image.fromarray(img)
    im.thumbnail((w, h))
    canvas = Image.new("RGB", (w, h), (20, 20, 20))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def _crop_panel(img: np.ndarray, box, side: int = 180) -> Image.Image:
    c = bbox_crop(img, tuple(float(v) for v in box))
    if c is None or c.size == 0:
        return Image.new("RGB", (side, side), (60, 20, 20))
    return _fit(c, side, side)


def _font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _paste_strip(panel, draw, x0, y, idxs, center, frames, boxes, side, font):
    """A horizontal filmstrip of GT frames (box drawn), the selected frame boxed in
    green and its neighbours dimmer, for pose-stability context."""
    x = x0
    for k in idxs:
        f = frames.get(k)
        if f is None:
            x += side + 6
            continue
        color = (0, 220, 0) if k == center else (120, 120, 120)
        panel.paste(_fit(draw_box(f, boxes[k], color=color, width=3), side, int(side * 9 / 16)), (x, y))
        draw.text((x + 3, y + 2), str(k), fill=(200, 200, 60), font=font)
        x += side + 6
    return x


def compose_panel(rec, gt_frames: dict, sr_frames: dict, boxes, strip_radius: int = STRIP_RADIUS) -> Image.Image:
    """Panel: GT_i|GT_j and SR_i|SR_j full frames (box drawn), a GT pose strip around
    each selected frame, the four scorer crops, then a scores + reliability caption."""
    i, j = rec["i"], rec["j"]
    box_i, box_j = tuple(float(v) for v in boxes[i]), tuple(float(v) for v in boxes[j])
    gt_i, gt_j = gt_frames[i], gt_frames[j]
    sr_i, sr_j = sr_frames[i], sr_frames[j]

    fw, fh = 560, 315
    pad, cside, cap_h, sside = 12, 180, 176, 150
    sfh = int(sside * 9 / 16)
    W = fw * 2 + pad * 3
    H = fh * 2 + sfh + cside + cap_h + pad * 7

    panel = Image.new("RGB", (W, H), (15, 15, 18))
    font, small = _font(16), _font(14)
    d = ImageDraw.Draw(panel)

    frames = [(gt_i, box_i, "GT  i"), (gt_j, box_j, "GT  j"),
              (sr_i, box_i, "SR  i"), (sr_j, box_j, "SR  j")]
    for k, (img, box, label) in enumerate(frames):
        r, c = divmod(k, 2)
        x = pad + c * (fw + pad)
        y = pad + r * (fh + pad)
        panel.paste(_fit(draw_box(img, box), fw, fh), (x, y))
        d.text((x + 6, y + 4), label, fill=(0, 255, 120), font=font)

    # GT pose strips: frames either side of i (left half) and j (right half)
    sy = pad * 3 + fh * 2
    hi = boxes.shape[0] - 1
    d.text((pad, sy - 16), "GT pose strip  (selected frame boxed green)", fill=(170, 170, 170), font=small)
    _paste_strip(panel, d, pad, sy, strip_indices(i, 0, hi, strip_radius), i, gt_frames, boxes, sside, small)
    _paste_strip(panel, d, pad + fw + pad, sy, strip_indices(j, 0, hi, strip_radius), j, gt_frames, boxes, sside, small)

    cy = sy + sfh + pad
    crops = [(_crop_panel(gt_i, box_i), "GT i"), (_crop_panel(gt_j, box_j), "GT j"),
             (_crop_panel(sr_i, box_i), "SR i"), (_crop_panel(sr_j, box_j), "SR j")]
    for k, (cim, lab) in enumerate(crops):
        x = pad + k * (cside + pad)
        panel.paste(cim, (x, cy))
        d.text((x + 4, cy + 2), lab, fill=(200, 200, 60), font=small)

    cap_y = cy + cside + pad
    c_gt, c_sr, s_M = rec.get("c_gt", {}), rec.get("c_sr", {}), rec.get("s_M", {})
    fid, fac = rec.get("fidelity", {}), rec.get("reliability_factors", {})
    is_lr = rec.get("pair_type") == "long_range"
    span_label = "span" if is_lr else "gap"
    reasons = exclusion_reasons(fac)
    lines = [
        f"{rec.get('video','?')}   status={rec.get('status')}   type={rec.get('pair_type','adr')}   "
        f"{span_label}={rec.get('gap_s')}s   separation={abs(j - i)} fr   i={i}  j={j}   "
        f"consistency={rec.get('consistency'):.3f}   reliability={rec.get('reliability'):.3f}"
        if rec.get("consistency") is not None else "",
        "  ".join(f"{m}: c_gt={c_gt.get(m,float('nan')):.3f} c_sr={c_sr.get(m,float('nan')):.3f} "
                  f"s_M={s_M.get(m,float('nan')):.3f}" for m in s_M),
        ("reliability  " + "  ".join(f"{k}={fac[k]:.3f}" for k in ("size", "sharp", "edge", "anchor") if k in fac))
        if fac else "",
        ("EXCLUDED: " + ", ".join(reasons)) if reasons else "",
        "fidelity  " + "  ".join(f"{k}={v:.3f}" for k, v in fid.items()) if fid else "",
    ]
    for n, line in enumerate(lines):
        fill = (255, 170, 90) if line.startswith("EXCLUDED") else (235, 235, 235)
        d.text((pad, cap_y + n * 22), line, fill=fill, font=font)
    return panel


def _panel_label(rec, stem):
    """Unique, informative panel basename (per-gap runs emit several pairs/clip)."""
    if "pair_id" not in rec:
        return stem
    gs = rec.get("gap_s")
    kind = "lr" if rec.get("pair_type") == "long_range" else "gap"
    return f"{stem}_p{rec['pair_id']}_{kind}" + (f"{gs:.0f}s" if gs is not None else "")


def _row(rec, stem):
    """One index row (label, consistency, png_name, reasons, gap_s) -- no decode."""
    label = _panel_label(rec, stem)
    reasons = exclusion_reasons(rec.get("reliability_factors", {}))
    return (label, rec.get("consistency"), f"{label}.png", reasons, rec.get("gap_s"))


def _write_index(out: Path, keep: list, excluded: list) -> None:
    """Grouped gallery: Keep split into SR-diverges (worst first) vs SR-preserves,
    with a headline stat line, then the excluded appendix with reasons."""
    import statistics as st

    cons = [c for _, c, _, _, _ in keep if c is not None]
    mean = st.mean(cons) if cons else float("nan")
    med = st.median(cons) if cons else float("nan")
    diverge = sorted([r for r in keep if r[1] is not None and consistency_band(r[1]) == "diverges"],
                     key=lambda r: r[1])
    preserve = sorted([r for r in keep if r[1] is None or consistency_band(r[1]) == "preserves"],
                      key=lambda r: (r[1] is None, r[1]))
    excluded = sorted(excluded, key=lambda r: (", ".join(r[3]), r[1] is None, r[1]))

    def block(rows, border):
        h = []
        for label, c, name, reasons, gap_s in rows:
            sep = f" &nbsp; &Delta;t={gap_s}s" if gap_s is not None else ""
            tag = f" &mdash; <span style='color:#f9a'>{', '.join(reasons)}</span>" if reasons else ""
            cc = f"{c:.3f}" if c is not None else "n/a"
            h.append(f"<h3>{label} &mdash; consistency={cc}{sep}{tag}</h3>")
            h.append(f"<img src='{name}' style='max-width:100%;border:1px solid {border}'><hr>")
        return h

    html = ["<html><body style='background:#111;color:#eee;font-family:sans-serif'>",
            "<h1>ADR cross-gap consistency &mdash; FlashVSR review</h1>",
            f"<p>{len(keep)} clean pairs &middot; mean consistency {mean:.3f}, median {med:.3f} "
            f"&middot; {len(diverge)} diverge, {len(preserve)} preserve &middot; "
            f"{len(excluded)} excluded (edge / weak-anchor / too-small). "
            "Lower consistency = SR departs from the GT cross-gap relationship; motion "
            "blur is a reviewer eye-call, not an automatic gate.</p>",
            "<h2 style='color:#f66'>SR diverges &mdash; cross-gap relationship changed (worst first)</h2>"]
    html += block(diverge, "#733")
    html.append("<h2 style='color:#6c6'>SR preserves &mdash; cross-gap relationship kept</h2>")
    html += block(preserve, "#363")
    html.append("<h2 style='color:#f9a'>Excluded &mdash; weak pairs (reason shown)</h2>")
    html += block(excluded, "#533")
    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html))
    print(f"index: {out/'index.html'}  ({len(keep)} keep [{len(diverge)} diverge/"
          f"{len(preserve)} preserve], {len(excluded)} excluded)")


def main(argv=None):
    import argparse
    from collections import defaultdict

    p = argparse.ArgumentParser(description="Export selected-pair review panels")
    p.add_argument("--json", required=True)
    p.add_argument("--gt-dir")
    p.add_argument("--sr-dir")
    p.add_argument("--ann-dir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--strip-radius", type=int, default=STRIP_RADIUS)
    p.add_argument("--html-only", action="store_true",
                   help="rebuild index.html from the JSON + existing panels (no video decode)")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored = [r for r in json.loads(Path(args.json).read_text()) if r.get("status") in ("ok", "weak")]

    keep, excluded = [], []       # (label, consistency, filename, reasons, gap_s)
    if args.html_only:
        for rec in scored:
            row = _row(rec, rec["video"].rsplit(".", 1)[0])
            (excluded if row[3] else keep).append(row)
    else:
        by_clip = defaultdict(list)
        for rec in scored:
            by_clip[rec["video"]].append(rec)
        for clip, recs in by_clip.items():
            stem = clip.rsplit(".", 1)[0]
            boxes = load_annotations(f"{args.ann_dir}/{stem}").boxes
            hi = boxes.shape[0] - 1
            gt_need, sr_need = set(), set()
            for rec in recs:
                for c in (rec["i"], rec["j"]):
                    sr_need.add(c)
                    gt_need.update(strip_indices(c, 0, hi, args.strip_radius))
            gt_frames = grab_frames(f"{args.gt_dir}/{stem}.mp4", gt_need)
            sr_frames = grab_frames(f"{args.sr_dir}/{stem}.mp4", sr_need)
            for rec in recs:
                panel = compose_panel(rec, gt_frames, sr_frames, boxes, args.strip_radius)
                row = _row(rec, stem)
                panel.save(out / row[2])
                (excluded if row[3] else keep).append(row)
                print(f"wrote {out/row[2]}{'  [excluded: ' + ', '.join(row[3]) + ']' if row[3] else ''}",
                      flush=True)

    _write_index(out, keep, excluded)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
