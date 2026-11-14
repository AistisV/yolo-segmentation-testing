"""Does classic CV recover room areas? Seeds = a point inside each ground-truth room
(stand-in for a model pointing at the room label). Reports per-room area error."""
import json, os, sys, glob
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from cv_rooms import segment_rooms

IM = os.environ.get("IMAGES_DIR", "data/swiss_dwellings")
plans = sys.argv[1:] or [os.path.basename(p)[:-5] for p in sorted(glob.glob("gt/*.json"))]
errs, rows = [], []
for p in plans:
    g = json.load(open(f"gt/{p}.json")); ppm = g["scale"]["px_per_m"]
    rooms = [r for r in g["rooms"] if len(r["polygon"]) >= 3 and Polygon(r["polygon"]).buffer(0).area / ppm**2 >= 1.0]
    polys = [Polygon(r["polygon"]).buffer(0) for r in rooms]
    seeds = [list(pp.representative_point().coords[0]) for pp in polys]
    res = segment_rooms(Image.open(f"{IM}/{p}.png"), seeds)
    pe = []
    for r, pp, o in zip(rooms, polys, res):
        gt_a = pp.area
        e = (o["area_px"] - gt_a) / gt_a if gt_a else 0
        iou = 0
        if len(o["polygon"]) >= 3:
            q = Polygon(o["polygon"]).buffer(0); iou = q.intersection(pp).area / q.union(pp).area
        pe.append(e); rows.append((p, r.get("number"), r.get("label"), round(gt_a / ppm**2, 2), round(o["area_px"] / ppm**2, 2), round(100*e, 1), round(iou, 3)))
    pe = np.abs(pe); errs += list(pe)
    print(f"{p:32s} ppm={ppm:5.1f} rooms={len(pe):3d}  within2%={np.mean(pe<=.02):.0%}  within5%={np.mean(pe<=.05):.0%}  median={100*np.median(pe):.1f}%")
errs = np.array(errs)
print(f"\nALL {len(errs)} rooms: within 2% {np.mean(errs<=.02):.1%} | within 5% {np.mean(errs<=.05):.1%} | median err {100*np.median(errs):.2f}%")
json.dump(rows, open("cv_rooms_eval.json", "w"))
