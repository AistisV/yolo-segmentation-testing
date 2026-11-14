"""Where do the segmenter's room areas go wrong?  Uses the true scale and true room outlines, so it isolates geometry.

    python3 area_diag.py --preds seg_preds --gt gt_full            # Swiss eval plans

For every true room it finds the predicted room it overlaps most and reports:
  - within 2% of the true area, for outlines grown/shrunk by k pixels (finds any systematic bias)
  - merged: one predicted room covers 2+ true rooms (an open doorway joined them)
  - split: a true room is covered by 2+ predicted rooms
  - missed: no predicted room overlaps it
"""
import argparse, glob, json, os, statistics
from shapely.geometry import Polygon

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="seg_preds"); ap.add_argument("--gt", default="gt_full")
a = ap.parse_args()


def P(poly):
    try:
        p = Polygon(poly).buffer(0)
        return p if p.area > 0 else None
    except Exception:
        return None


KS = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
ok = {k: 0 for k in KS}; n = 0; merged = split = missed = 0; ratios = []
small_ok = small_n = 0
for f in sorted(glob.glob(os.path.join(a.preds, "*.json"))):
    pred = json.load(open(f)); gp = os.path.join(a.gt, pred["plan"] + ".json")
    if not os.path.exists(gp) or "rooms" not in pred: continue
    g = json.load(open(gp)); ppm = (g.get("scale") or {}).get("px_per_m")
    if not ppm: continue
    segs = [s for s in (P(r["polygon"]) for r in pred["rooms"]) if s is not None]
    truths = [(P(r["polygon"]), r.get("area_m2")) for r in g["rooms"]]
    cover = [0] * len(segs)                                    # how many true rooms each segment mostly covers
    for t, area in truths:
        if t is None or not area: continue
        n += 1
        hits = [(t.intersection(s).area, j) for j, s in enumerate(segs) if s.intersects(t)]
        hits = [(x, j) for x, j in hits if x > 0.1 * t.area]
        if not hits: missed += 1; continue
        if len(hits) >= 2: split += 1
        x, j = max(hits)
        if x > 0.5 * t.area: cover[j] += 1
        s = segs[j]; true_px = area * ppm * ppm
        ratios.append(s.area / true_px)
        for k in KS:
            if abs(s.buffer(k, join_style=2).area / true_px - 1) <= 0.02: ok[k] += 1
        if area < 6:
            small_n += 1; small_ok += abs(s.area / true_px - 1) <= 0.02
    merged += sum(c >= 2 for c in cover)

print(f"{n} true rooms | missed {missed} | split {split} | predicted rooms covering 2+ true rooms: {merged}")
print(f"median predicted/true area {statistics.median(ratios):.4f}  (1.0 = no bias)")
for k in KS: print(f"  grow outline by {k:+.1f} px: {ok[k] / n:6.1%} of rooms within 2%")
if small_n: print(f"rooms under 6 m²: {small_ok / small_n:.1%} within 2% (k=0, {small_n} rooms)")
