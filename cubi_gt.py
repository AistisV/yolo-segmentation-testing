"""CubiCasa5k SVG annotations -> our ground-truth format (rooms with polygons + type, items with points + boxes).
Checked against Stoa's own CubiCasa answers (counts), like validate_gt.py for Swiss.

    python cubi_gt.py --zip /vol/cubi/cubicasa5k.zip --images /vol/data/data/cubicasa5k --questions /vol/data/questions.jsonl --out /vol/gt_cubi
Our image 'colorful-10052.png' <-> zip entry '.../colorful/10052/model.svg'. SVG coordinates are rescaled to our image size.
"""
import argparse, collections, json, os, re, zipfile
import xml.etree.ElementTree as ET
from PIL import Image

ROOM_TYPE = {"Bedroom": "bedroom", "Kitchen": "kitchen", "Bath": "bathroom", "Toilet": "wc", "Sauna": "sauna",
             "Outdoor": "balcony", "Balcony": "balcony", "Terrace": "terrace", "LivingRoom": "living", "Entry": "entry"}
# fitted against Stoa's answers (cubi_fit.py): Toilet -> 100%, Bathtub + Shower -> 100% (ShowerCab/ShowerPlatform are not counted)
ICON = {"Toilet": "toilet", "Sink": "washbasin", "Bathtub": "bathtub", "Shower": "shower"}
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def parse_matrix(t):
    """SVG transform -> 3x3 affine (supports matrix/translate/scale/rotate; enough for CubiCasa)."""
    import math
    M = [1, 0, 0, 1, 0, 0]
    def mul(A, B):
        a, b, c, d, e, f = A; g, h, i, j, k, l = B
        return [a * g + c * h, b * g + d * h, a * i + c * j, b * i + d * j, a * k + c * l + e, b * k + d * l + f]
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", t or ""):
        v = [float(x) for x in NUM.findall(args)]
        if name == "matrix" and len(v) == 6: T = v
        elif name == "translate": T = [1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0]
        elif name == "scale": T = [v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0]
        elif name == "rotate":
            r = math.radians(v[0]); c, s = math.cos(r), math.sin(r); T = [c, s, -s, c, 0, 0]
            if len(v) == 3: T = mul(mul([1, 0, 0, 1, v[1], v[2]], T), [1, 0, 0, 1, -v[1], -v[2]])
        else: continue
        M = mul(M, T)
    return M


def apply(M, x, y):
    return M[0] * x + M[2] * y + M[4], M[1] * x + M[3] * y + M[5]


def polys(el, M, any_shape=False):
    """All polygons under el, in global coordinates (transforms applied along the way)."""
    out = []
    def walk(e, M):
        M2 = M
        if e.get("transform"):
            A = parse_matrix(e.get("transform")); a, b, c, d, ee, f = M
            M2 = [a * A[0] + c * A[1], b * A[0] + d * A[1], a * A[2] + c * A[3], b * A[2] + d * A[3], a * A[4] + c * A[5] + ee, b * A[4] + d * A[5] + f]
        tag = e.tag.split("}")[-1]
        if tag == "polygon" and e.get("points"):
            v = [float(x) for x in NUM.findall(e.get("points"))]
            pts = [apply(M2, v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
            if len(pts) >= 3: out.append((pts, e))
        elif any_shape and tag == "rect" and e.get("width"):
            x, y, w, h = (float(e.get(k) or 0) for k in ("x", "y", "width", "height"))
            out.append(([apply(M2, x, y), apply(M2, x + w, y), apply(M2, x + w, y + h), apply(M2, x, y + h)], e))
        elif any_shape and tag in ("circle", "ellipse") and (e.get("r") or e.get("rx")):
            cx, cy = float(e.get("cx") or 0), float(e.get("cy") or 0); rx = float(e.get("r") or e.get("rx")); ry = float(e.get("r") or e.get("ry") or rx)
            out.append(([apply(M2, cx - rx, cy - ry), apply(M2, cx + rx, cy - ry), apply(M2, cx + rx, cy + ry), apply(M2, cx - rx, cy + ry)], e))
        elif any_shape and tag in ("path", "polyline", "line"):
            v = [float(x) for x in NUM.findall(e.get("d") or e.get("points") or " ".join(e.get(k) or "" for k in ("x1", "y1", "x2", "y2")))]
            pts = [apply(M2, v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
            if len(pts) >= 2: out.append((pts, e))
        for ch in e: walk(ch, M2)
    walk(el, M)
    return out


def area(p):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(p, p[1:] + p[:1]))) / 2


def parse_svg(data, W, H):
    root = ET.fromstring(data)
    sw = float(NUM.findall(root.get("width") or str(W))[0]); sh = float(NUM.findall(root.get("height") or str(H))[0])
    sx, sy = W / sw, H / sh
    rooms, items = [], []
    parent = {c: p for p in root.iter() for c in p}
    def global_M(e):
        chain = []
        while e is not None: chain.append(e); e = parent.get(e)
        M = [sx, 0, 0, sy, 0, 0]
        for a in reversed(chain[1:]):
            if a.get("transform"):
                A = parse_matrix(a.get("transform")); m = M
                M = [m[0] * A[0] + m[2] * A[1], m[1] * A[0] + m[3] * A[1], m[0] * A[2] + m[2] * A[3], m[1] * A[2] + m[3] * A[3],
                     m[0] * A[4] + m[2] * A[5] + m[4], m[1] * A[4] + m[3] * A[5] + m[5]]
        return M
    for g in root.iter():
        if g.tag.split("}")[-1] != "g": continue
        cls, gid = g.get("class") or "", g.get("id") or ""
        if cls.startswith("Space "):
            name = cls.replace("Space ", "").split(" ")[0]
            pp = [p for p, e in polys(g, global_M(g)) if parent.get(e) is g] or [p for p, _ in polys(g, global_M(g))][:1]
            if pp:
                poly = max(pp, key=area)
                rooms.append({"type": ROOM_TYPE.get(name), "class": name, "label": None, "number": None, "apartment": None,
                              "printed_area_m2": None, "polygon": [[round(x, 1), round(y, 1)] for x, y in poly]})
        elif gid in ("Door", "Window") or cls.startswith("FixedFurniture "):
            if cls.startswith("FixedFurniture "):
                t = ICON.get(cls.replace("FixedFurniture ", "").split(" ")[0])
                if not t: continue
                pp = ([p for p, e in polys(g, global_M(g)) if "BoundaryPolygon" in (parent.get(e).get("class") or "")]
                      or [p for p, _ in polys(g, global_M(g))] or [p for p, _ in polys(g, global_M(g), any_shape=True)])
            else:
                t = gid.lower(); pp = [p for p, _ in polys(g, global_M(g))]
            if not pp: continue
            xs = [x for p in pp for x, _ in p]; ys = [y for p in pp for _, y in p]
            items.append({"type": t, "point": [round((min(xs) + max(xs)) / 2, 1), round((min(ys) + max(ys)) / 2, 1)],
                          "bbox": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]})
    return {"source": "cubicasa5k", "image_size": [W, H], "scale": {"px_per_m": None}, "apartments": [], "rooms": rooms, "items": items}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--questions", required=True); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ours = {os.path.splitext(f)[0] for f in os.listdir(a.images) if f.endswith(".png")}
    zf = zipfile.ZipFile(a.zip)
    svgs = [n for n in zf.namelist() if n.endswith("model.svg")]
    print(len(svgs), "SVGs in zip, e.g.", svgs[:2], "|", len(ours), "of our images", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(32) as ex:                       # image sizes: many small reads, done in parallel
        sizes = dict(zip(sorted(ours), ex.map(lambda nm: Image.open(os.path.join(a.images, nm + ".png")).size, sorted(ours))))
    print("read", len(sizes), "image sizes", flush=True)
    done = 0; miss = 0; bad = 0; inside = tot_items = 0
    for k, n in enumerate(svgs):
        if k % 500 == 0: print(f"svg {k}/{len(svgs)}, converted {done}", flush=True)
        parts = n.strip("/").split("/")
        name = f"{parts[-3]}-{parts[-2]}"
        if name not in ours: miss += 1; continue
        W, H = sizes[name]
        try:
            g = parse_svg(zf.read(n), W, H)
        except Exception as e:
            bad += 1; continue
        for it in g["items"]:
            tot_items += 1; inside += 0 <= it["point"][0] <= W and 0 <= it["point"][1] <= H
        json.dump(g, open(os.path.join(a.out, name + ".json"), "w"))
        done += 1
        if a.limit and done >= a.limit: break
    print(f"converted {done} | zip entries not in our images {miss} | parse errors {bad} | items inside image {inside}/{tot_items}", flush=True)
    # check against Stoa's answers
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from answer import parse, correct
    tot, ok = collections.Counter(), collections.Counter(); diffs = collections.defaultdict(collections.Counter)
    cache = {}
    for line in open(a.questions, encoding="utf-8"):
        if "cubicasa5k" not in line: continue
        q = json.loads(line); p = os.path.basename(q["path"]).rsplit(".", 1)[0]
        f = os.path.join(a.out, p + ".json")
        if p not in cache: cache[p] = json.load(open(f)) if os.path.exists(f) else None
        g = cache[p]
        if g is None: continue
        kind, arg = parse(q["question"])
        if kind == "count_item": pred = sum(1 for i in g["items"] if i["type"] == arg)
        elif kind == "count_items": pred = sum(1 for i in g["items"] if i["type"] in arg)
        elif kind == "count_roomtype": pred = sum(1 for r in g["rooms"] if r["type"] == arg)
        else: continue
        k = kind + ":" + str(arg); tot[k] += 1; ok[k] += correct(pred, q["answer"], "count")
        diffs[k][max(-3, min(3, pred - q["answer"]))] += 1
    for k in sorted(tot): print(f"   {k:32s} {ok[k]:6d}/{tot[k]:6d} {ok[k]/tot[k]:6.1%}   pred-truth: {dict(sorted(diffs[k].items()))}")
