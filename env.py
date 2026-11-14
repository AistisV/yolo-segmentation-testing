"""FloorPlanEnv: the shared environment for every model (frontier baseline and our small model).

Episode = one floor-plan image. The agent may call tools (crop, measure_area) any number of
times, then submits a schematic JSON. The environment answers every Stoa question for that plan
from the schematic with deterministic code (answer.py) and scores it by the official rule.

    env = FloorPlanEnv("site52_bldg298_floor1183", cfg)
    obs = env.reset()
    obs = env.step({"tool": "crop", "args": {"x0": 900, "y0": 200, "x1": 1500, "y1": 700}})
    obs, reward, done, info = env.step({"submit": schematic})
"""
import base64, io, json, math, os
from PIL import Image
import threading
from answer import answer, correct
from policy_io import sanitize

SCHEMA_HELP = open(os.path.join(os.path.dirname(__file__), "schema.md"), encoding="utf-8").read()

# ---------- tools (identical for every model) ----------

def tool_crop(img, x0, y0, x1, y1, max_side=1024):
    """Return a zoomed crop of the ORIGINAL full-resolution image (pixel coords)."""
    W, H = img.size
    x0, x1 = sorted((max(0, int(x0)), min(W, int(x1))))
    y0, y1 = sorted((max(0, int(y0)), min(H, int(y1))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return {"error": "region too small"}
    crop = img.crop((x0, y0, x1, y1))
    s = min(1.0, max_side / max(crop.size)) if max(crop.size) > max_side else max_side / max(crop.size)
    s = min(s, 4.0)
    crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))))
    buf = io.BytesIO(); crop.save(buf, "PNG")
    return {"image_png_b64": base64.b64encode(buf.getvalue()).decode(),
            "note": f"crop of [{x0},{y0}]-[{x1},{y1}] scaled x{s:.2f}; coordinates you report must stay in ORIGINAL image pixels"}

def tool_measure_area(polygon, px_per_m):
    """Deterministic net area in m² of a polygon given in original image pixels."""
    if len(polygon) < 3 or not px_per_m:
        return {"error": "need >=3 points and px_per_m"}
    a = 0.0
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        a += x1 * y2 - x2 * y1
    return {"area_m2": round(abs(a) / 2 / px_per_m ** 2, 2)}

TOOLS_SPEC = [
    {"name": "crop", "description": "Zoom into a region of the original image (pixel coordinates). Returns an image.",
     "parameters": {"type": "object", "properties": {k: {"type": "number"} for k in ("x0", "y0", "x1", "y1")},
                    "required": ["x0", "y0", "x1", "y1"]}},
    {"name": "measure_area", "description": "Exact area in m² of a polygon in original image pixels, given the scale in pixels per metre.",
     "parameters": {"type": "object", "properties": {
         "polygon": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
         "px_per_m": {"type": "number"}}, "required": ["polygon", "px_per_m"]}},
    {"name": "submit_schematic", "description": "Submit the final schematic JSON (see schema). Ends the episode.",
     "parameters": {"type": "object", "properties": {"schematic": {"type": "object"}}, "required": ["schematic"]}},
]

def tool_segment_rooms(img, seeds):
    """Classic-CV room segmentation (cv_rooms.py). seeds: [[x, y], ...] one point inside each room
    (original pixels). Returns one polygon + pixel area per seed, and an overlay image of the regions."""
    from cv_rooms import segment_rooms
    import numpy as np
    from PIL import ImageDraw
    seeds = [s[:2] for s in seeds if isinstance(s, (list, tuple)) and len(s) >= 2][:80]
    if not seeds:
        return {"error": "give seeds as [[x, y], ...]"}
    res = segment_rooms(img, seeds)
    ov = img.copy(); d = ImageDraw.Draw(ov, "RGBA")
    rng = np.random.RandomState(1)
    for i, r in enumerate(res):
        if len(r["polygon"]) >= 3:
            c = tuple(int(v) for v in rng.randint(40, 220, 3))
            d.polygon([tuple(p) for p in r["polygon"]], fill=c + (90,), outline=c + (255,))
        d.text(tuple(seeds[i]), str(i), fill=(0, 0, 0, 255))
    s = 2048 / max(ov.size)
    if s < 1: ov = ov.resize((round(ov.width * s), round(ov.height * s)))
    buf = io.BytesIO(); ov.save(buf, "PNG")
    regions = [{"seed": i, "polygon": [[round(x), round(y)] for x, y in r["polygon"]],
                "area_px": round(r["area_px"]), "closed": r["ok"]} for i, r in enumerate(res)]
    return {"regions": regions, "image_png_b64": base64.b64encode(buf.getvalue()).decode(),
            "note": "overlay of the regions (numbered by seed) attached; areas are in pixels, "
                    "divide by px_per_m**2 for m². Regions can be wrong (open kitchens, stairs, balconies, "
                    "furniture lines): check the overlay and correct polygons yourself where needed."}

TOOLS_SPEC_CV = {"name": "segment_rooms",
    "description": "Classic computer-vision room segmentation. Give one seed point inside each room (e.g. at its label); "
                   "returns each room's polygon and pixel area, split at walls and door openings, plus an overlay image.",
    "parameters": {"type": "object", "properties": {"seeds": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}},
                   "required": ["seeds"]}}

# ---------- geometry check vs ground truth ----------

def item_f1(pred_items, gt_items, tol_px=30):
    """Greedy one-to-one matching by type within tol_px (env passes 0.5 m in pixels). Returns P, R, F1."""
    used, tp = set(), 0
    for p in pred_items:
        best, bd = None, tol_px
        for j, g in enumerate(gt_items):
            if j in used or g["type"] != p.get("type"):
                continue
            d = math.dist(p.get("point", [1e9, 1e9]), g["point"])
            if d <= bd:
                best, bd = j, d
        if best is not None:
            used.add(best); tp += 1
    P = tp / len(pred_items) if pred_items else 0.0
    R = tp / len(gt_items) if gt_items else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def normalise(schematic, image_size):
    """Accept coordinates as pixels (default), [y,x] in 0-1000 ('norm1000') or [x,y] in 0-1000 ('norm1000_xy')."""
    cs = schematic.get("coord_space")
    if cs not in ("norm1000", "norm1000_xy"):
        return schematic
    W, H = image_size
    f = (lambda p: [p[1] / 1000 * W, p[0] / 1000 * H]) if cs == "norm1000" else (lambda p: [p[0] / 1000 * W, p[1] / 1000 * H])
    s = json.loads(json.dumps(schematic))
    for it in s.get("items", []):
        it["point"] = f(it["point"])
    for r in s.get("rooms", []):
        r["polygon"] = [f(p) for p in r.get("polygon", [])]
    s["coord_space"] = "pixel"
    return s

_QCACHE, _QLOCK = {}, threading.Lock()

def questions_for(path, plan):
    """Questions grouped by plan, loaded once per file (the full set has 127k lines)."""
    with _QLOCK:
        if path not in _QCACHE:
            by = {}
            for line in open(path, encoding="utf-8"):
                q = json.loads(line)
                by.setdefault(os.path.basename(q["path"]).rsplit(".", 1)[0], []).append(q)
            _QCACHE[path] = by
    return _QCACHE[path].get(plan, [])

# ---------- environment ----------

class FloorPlanEnv:
    def __init__(self, plan, cfg):
        """cfg: dict(images_dir, gt_dir, questions_path, w_answers=0.8, w_items=0.2)"""
        self.plan, self.cfg = plan, cfg
        dirs = cfg["images_dir"] if isinstance(cfg["images_dir"], (list, tuple)) else str(cfg["images_dir"]).split(os.pathsep)
        path = next((os.path.join(d, plan + ext) for d in dirs for ext in (".png", ".jpg") if os.path.exists(os.path.join(d, plan + ext))),
                    os.path.join(dirs[0], plan + ".png"))
        self.img_path = path
        self.img = Image.open(path).convert("RGB")
        gt_path = os.path.join(cfg["gt_dir"], plan + ".json")
        self.gt = json.load(open(gt_path)) if os.path.exists(gt_path) else {}    # CubiCasa: answers only, no schematic
        self.questions = questions_for(cfg["questions_path"], plan)
        self.steps, self.done = 0, False

    def reset(self):
        self.steps, self.done = 0, False
        return {"image_size": list(self.img.size), "instructions": SCHEMA_HELP, "tools": TOOLS_SPEC}

    def step(self, action):
        assert not self.done, "episode finished"
        self.steps += 1
        if "submit" in action:
            self.done = True
            info = self.score(action["submit"])
            return {"result": "submitted"}, info["reward"], True, info
        name, args = action.get("tool"), action.get("args", {})
        if name == "crop":
            return tool_crop(self.img, **args)
        if name == "measure_area":
            return tool_measure_area(**args)
        if name == "segment_rooms" and self.cfg.get("cv_tool"):
            return tool_segment_rooms(self.img, args.get("seeds", []))
        return {"error": f"unknown tool {name}"}

    def snap_rooms(self, s):
        """Deterministic post-processing, identical for every model: move each room edge onto the
        nearest wall face. Uses the MODEL's own scale for the search radius (never the ground truth)."""
        from cv_rooms import snap_polygon, compute_masks
        ppm = (s.get("scale") or {}).get("px_per_m") or 0
        rad = int(0.45 * ppm) if 10 < ppm < 400 else 25
        s = json.loads(json.dumps(s))
        if not getattr(self, "_masks", None):
            self._masks = compute_masks(self.img, 1)      # per-env, so scoring is thread-safe
        for r in s.get("rooms", []):
            poly = r.get("polygon") or []
            try:
                if len(poly) >= 3:
                    r["polygon"] = snap_polygon(self.img, [p[:2] for p in poly], radius_px=rad, T=1, masks=self._masks)["polygon"]
            except Exception:
                pass
        return s

    def score(self, schematic):
        s = normalise(sanitize(schematic or {}), self.img.size)
        if self.cfg.get("snap"):
            s = self.snap_rooms(s)
        rows, scored = [], []
        for q in self.questions:
            kind, pred = answer(s, q["question"])
            ok = correct(pred, q["answer"], q["type"])
            rows.append(dict(question=q["question"], kind=kind, pred=pred, truth=q["answer"], ok=ok))
            # questions whose true answer is 0 are reported but not rewarded: an empty schematic
            # would otherwise get them right for free (do-nothing baseline must score ~0)
            if q["answer"] not in (0, 0.0):
                scored.append(ok)
        acc = sum(scored) / len(scored) if scored else 0.0
        tol = 0.5 * ((self.gt.get("scale") or {}).get("px_per_m") or 60)   # within half a metre of the true symbol (no scale: 30 px)
        P, R, F1 = item_f1(s.get("items", []), self.gt.get("items", []), tol)
        if self.gt.get("items"):
            reward = self.cfg.get("w_answers", 0.8) * acc + self.cfg.get("w_items", 0.2) * F1
        else:                                                       # no item positions known: reward = answers only
            reward = acc
        return dict(reward=round(reward, 4), answer_acc=round(acc, 4), items_p=round(P, 3), items_r=round(R, 3),
                    items_f1=round(F1, 3), n_questions=len(rows), rows=rows)
