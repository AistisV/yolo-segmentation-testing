"""Ceiling test: if a model gets every room's identity right but outlines it only roughly
(each edge off by up to +-R metres), how accurate are areas after snapping to walls?"""
import json, os, sys, glob, random
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from cv_rooms import snap_polygon, _poly_area

IM = os.environ.get("IMAGES_DIR", "data/swiss_dwellings")
R = float(os.environ.get("ROUGH_M", 0.25))
random.seed(0)

def rough(poly, ppm):
    # shift every axis-aligned edge independently by U(-R, R) metres (rectilinear-preserving)
    P = [list(p) for p in poly]; n = len(P)
    for i in range(n):
        (x1, y1), (x2, y2) = P[i], P[(i + 1) % n]
        d = random.uniform(-R, R) * ppm
        if abs(y2 - y1) < 0.2 * abs(x2 - x1):
            P[i][1] += d; P[(i + 1) % n][1] += d
        elif abs(x2 - x1) < 0.2 * abs(y2 - y1):
            P[i][0] += d; P[(i + 1) % n][0] += d
    return P

def main():
    plans = sys.argv[1:] or [os.path.basename(p)[:-5] for p in sorted(glob.glob("gt/*.json"))]
    raw, snp, rows = [], [], []
    for p in plans:
        g = json.load(open(f"gt/{p}.json")); ppm = g["scale"]["px_per_m"]
        img = Image.open(f"{IM}/{p}.png")
        for r in g["rooms"]:
            if len(r["polygon"]) < 3: continue
            # simplify GT polygon (drop collinear/duplicate points) for a cleaner rough outline
            gp = Polygon(r["polygon"]).buffer(0).simplify(1.0)
            if gp.geom_type != "Polygon" or gp.area / ppm**2 < 1: continue
            coords = [list(c) for c in list(gp.exterior.coords)[:-1]]
            ga = gp.area
            rp = rough(coords, ppm)
            s = snap_polygon(img, rp, radius_px=int(0.45 * ppm), T=int(os.environ.get("SNAP_T", 2)))
            e_raw = (_poly_area(rp) - ga) / ga; e_s = (s["area_px"] - ga) / ga
            raw.append(abs(e_raw)); snp.append(abs(e_s))
            rows.append((p, r.get("number"), r.get("label"), round(ga / ppm**2, 2), round(100 * e_raw, 1), round(100 * e_s, 1), sum(s["snapped"]), len(s["snapped"])))
    raw, snp = np.array(raw), np.array(snp)
    print(f"rooms={len(raw)} rough +-{R} m")
    print(f"  rough only : within 2% {np.mean(raw<=.02):.1%} | within 5% {np.mean(raw<=.05):.1%} | median {100*np.median(raw):.2f}%")
    print(f"  snapped    : within 2% {np.mean(snp<=.02):.1%} | within 5% {np.mean(snp<=.05):.1%} | median {100*np.median(snp):.2f}%")
    json.dump(rows, open("ceiling_rows.json", "w"))


if __name__ == "__main__":
    main()
