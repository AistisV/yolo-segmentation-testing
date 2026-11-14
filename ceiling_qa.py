"""End-to-end ceiling: GT schematic but each room outline roughened (+-R m per edge), then optionally
snapped to walls. Areas come from the polygons (printed areas kept: a model can read them).
Reports official-rule accuracy on area questions."""
import json, os, sys, glob, random, copy, collections
from PIL import Image
from shapely.geometry import Polygon
from env import FloorPlanEnv
from cv_rooms import snap_polygon
from ceiling_test import rough

CFG = dict(images_dir=os.environ["IMAGES_DIR"], gt_dir="gt", questions_path=os.environ["QUESTIONS"])
plans = sys.argv[1:] or [os.path.basename(p)[:-5] for p in sorted(glob.glob("gt/*.json"))]
res = collections.defaultdict(lambda: collections.Counter())
for p in plans:
    env = FloorPlanEnv(p, CFG); g = env.gt; ppm = g["scale"]["px_per_m"]
    if not env.questions: continue
    for mode in ("exact", "rough", "snapped"):
        random.seed(hash(p) % 1000)
        s = copy.deepcopy(g)
        for r in s["rooms"]:
            r.pop("area_m2", None)
            if mode == "exact" or len(r["polygon"]) < 3: continue
            gp = Polygon(r["polygon"]).buffer(0).simplify(1.0)
            if gp.geom_type != "Polygon": continue
            rp = rough([list(c) for c in list(gp.exterior.coords)[:-1]], ppm)
            r["polygon"] = snap_polygon(env.img, rp, radius_px=int(0.45 * ppm), T=1)["polygon"] if mode == "snapped" else rp
        info = env.score(s)
        for row in info["rows"]:
            if row["kind"].startswith("area"):
                res[(mode, row["kind"])]["n"] += 1; res[(mode, row["kind"])]["ok"] += bool(row["ok"])
for mode in ("exact", "rough", "snapped"):
    tot = sum(v["ok"] for k, v in res.items() if k[0] == mode); n = sum(v["n"] for k, v in res.items() if k[0] == mode)
    print(f"{mode:8s} area questions {tot}/{n} = {tot/max(n,1):.1%}   " +
          "  ".join(f"{k[1]} {v['ok']}/{v['n']}" for k, v in sorted(res.items()) if k[0] == mode))
