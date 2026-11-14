"""Draw what the system sees on a plan: segmented rooms (label + measured area) and detected doors/windows/fixtures.

    python3 viz.py --image plan.png --pred seg_preds/plan.json --labels results/gemini_single_gemini-3.8-flash/plan.json --out plan_viz.png
"""
import argparse, json, colorsys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from seg_post import split_merged
from scale_bar import refine_scale
from answer import polygon_area_px

COL = {"door": (230, 60, 30), "window": (30, 120, 230), "toilet": (150, 40, 200), "washbasin": (0, 160, 160),
       "bathtub": (20, 150, 60), "shower": (240, 150, 0), "stove": (120, 80, 40), "sink": (200, 0, 120)}
BOX_M = {"door": 0.9, "window": 1.0, "toilet": 0.6, "washbasin": 0.55, "bathtub": 1.0, "shower": 0.9, "stove": 0.6, "sink": 0.6}


def font(n):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try: return ImageFont.truetype(f, n)
        except Exception: pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--labels", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args()
    img = Image.open(a.image).convert("RGB"); W, H = img.size
    pred = json.load(open(a.pred)); segs = pred.get("rooms", [])
    rooms, ppm = [], None
    if a.labels:
        s = json.load(open(a.labels)).get("norm_schematic") or {}
        rooms = [dict(r, polygon=[[x * W / 1000, y * H / 1000] for x, y in r.get("polygon", [])]) for r in s.get("rooms", [])]
        ppm = (s.get("scale") or {}).get("px_per_m")
    ppm = refine_scale(np.asarray(img.convert("L")), ppm) or ppm
    segs, _ = split_merged(rooms, segs, ppm)
    from shapely.geometry import Polygon
    lab_for = {}
    for j, sg in enumerate(segs):
        sp = Polygon(sg["polygon"]).buffer(0)
        best = max(((Polygon(r["polygon"]).buffer(0).intersection(sp).area, r) for r in rooms if len(r["polygon"]) >= 3), default=(0, None), key=lambda t: t[0])
        if best[1] is not None and best[0] > 0.3 * sp.area: lab_for[j] = best[1]

    over = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(over)
    for j, sg in enumerate(segs):
        h = (j * 0.618) % 1; c = tuple(int(255 * v) for v in colorsys.hsv_to_rgb(h, 0.45, 0.95))
        d.polygon([tuple(p) for p in sg["polygon"]], fill=c + (90,), outline=c + (255,))
    img = Image.alpha_composite(img.convert("RGBA"), over); d = ImageDraw.Draw(img)
    f1, f2 = font(max(14, W // 120)), font(max(12, W // 150))
    for j, sg in enumerate(segs):
        if not ppm: break
        area = abs(polygon_area_px(sg["polygon"])) / ppm ** 2
        if area < 1.0: continue
        r = lab_for.get(j); cx, cy = np.mean(np.array(sg["polygon"]), 0)
        txt = f"{(r or {}).get('label') or ''} {area:.2f} m²".strip()
        tw = d.textlength(txt, font=f2)
        d.rectangle([cx - tw / 2 - 3, cy + 14, cx + tw / 2 + 3, cy + 14 + f2.size + 4], fill=(255, 255, 255, 230))
        d.text((cx - tw / 2, cy + 15), txt, fill=(0, 0, 0), font=f2)
    counts = {}
    for it in pred.get("items", []):
        if it.get("conf", 1) < a.conf: continue
        t = it["type"]; counts[t] = counts.get(t, 0) + 1
        x, y = it["point"]; s = BOX_M.get(t, 0.7) * (ppm or 60) / 2
        d.rectangle([x - s, y - s, x + s, y + s], outline=COL.get(t, (0, 0, 0)), width=4)
    y = 12
    d.rectangle([10, 8, 10 + 330, 16 + (len(counts) + 2) * (f1.size + 6)], fill=(255, 255, 255, 235), outline=(0, 0, 0))
    d.text((18, y), f"rooms {len([1 for sg in segs])} · scale {ppm:.2f} px/m" if ppm else "rooms", fill=(0, 0, 0), font=f1); y += f1.size + 8
    for t, n in sorted(counts.items(), key=lambda t: -t[1]):
        d.rectangle([18, y + 3, 18 + f1.size - 4, y + f1.size - 1], outline=COL.get(t, (0, 0, 0)), width=4)
        d.text((18 + f1.size + 6, y), f"{t}: {n}", fill=(0, 0, 0), font=f1); y += f1.size + 6
    img.convert("RGB").save(a.out)
    print("saved", a.out, counts)


if __name__ == "__main__":
    main()
