"""Score the specialist models (seg_preds/) combined with a label source, on the eval plans, under Stoa's rule.
Room numbers / labels / apartments / scale come from a language model (or ground truth, as an upper bound);
each labelled room takes the outline of the segmented room it overlaps most.

    modal volume get --force jx seg_preds .
    IMAGES_DIR=... QUESTIONS=... GT_DIR=gt_full python3 score_seg.py --preds seg_preds
"""
import argparse, json, os, glob, collections, math, statistics
from shapely.geometry import Polygon
from env import FloorPlanEnv
from eval_policy import CFG, table
import tiles
from seg_post import split_merged, total_indoor

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="seg_preds"); ap.add_argument("--model", default="gemini-3.8-flash")
ap.add_argument("--plans-file", default="split_full.json"); ap.add_argument("--key", default="eval")
ap.add_argument("--cubi", action="store_true", help="score the CubiCasa test plans (split_cubi.json / gt_cubi)")
ap.add_argument("--gt-cubi", default="gt_cubi"); ap.add_argument("--qwen-dir", default="")
ap.add_argument("--confs", default="0.3", help="detector confidence cut-offs to report, comma list")
ap.add_argument("--variants", default="base", help="comma list of base,split,total,both: post-processing of the segments")
a = ap.parse_args()


def P(poly):
    try:
        p = Polygon(poly).buffer(0)
        return p if p.area > 0 else None
    except Exception:
        return None


def assign(rooms, segs, strict=False):
    """Each segment goes to the labelled room it overlaps most (by IoU); that room takes the segment's outline."""
    sp = [P(s["polygon"]) for s in segs]; best = {}
    for i, r in enumerate(rooms):
        rp = P(r["polygon"]) if len(r.get("polygon") or []) >= 3 else None
        if rp is None: continue
        for j, q in enumerate(sp):
            if q is None or not rp.intersects(q): continue
            iou = rp.intersection(q).area / rp.union(q).area
            if iou > 0.3 and iou > best.get(j, (0, None))[0]:
                best[j] = (iou, i)
    used = {}
    for j, (iou, i) in best.items():
        if iou > used.get(i, (0, None))[0]: used[i] = (iou, j)
    for i, (iou, j) in used.items():
        rooms[i]["polygon"] = segs[j]["polygon"]
    if strict:                       # upper-bound test: a room without a segment has no outline at all
        for i, r in enumerate(rooms):
            if i not in used: r["polygon"] = []
    return len(used)


def label_source(name, p, env):
    W, H = env.img.size
    if name == "truth" and env.gt.get("source") == "cubicasa5k":
        T2L = {"bedroom": "MH", "kitchen": "K", "bathroom": "KH", "wc": "WC", "sauna": "S", "balcony": "PARVEKE"}
        g = env.gt
        return {"coord_space": "pixel", "scale": {"px_per_m": None}, "apartments": [],
                "rooms": [dict(number=None, label=T2L.get(r.get("type")) or (r.get("class") or "room"), apartment=None, printed_area_m2=None,
                               polygon=r["polygon"]) for r in g["rooms"]]}
    if name == "truth":
        g = env.gt; s = {"coord_space": "pixel", "scale": dict(g["scale"]), "apartments": g.get("apartments", []),
                         "rooms": [dict(number=r.get("number"), label=r.get("label"), apartment=r.get("apartment"),
                                        printed_area_m2=None, polygon=r["polygon"]) for r in g["rooms"]]}   # no printed areas: tests the outlines
        return json.loads(json.dumps(s))
    qd = a.qwen_dir or ("results_modal/results/c_zeroshot" if a.cubi else "results_modal/results/g_sft")
    f = {"gemini": f"results/gemini_single_{a.model}/{p}.json", "qwen": f"{qd}/{p}_greedy.json"}[name]
    if not os.path.exists(f): return None
    s = json.loads(json.dumps(json.load(open(f))["norm_schematic"]))
    for r in s.get("rooms", []): r["polygon"] = [[x * W / 1000, y * H / 1000] for x, y in r.get("polygon", [])]
    s["coord_space"] = "pixel"; return s


def items_source(name, p, env, pred, lab):
    if name == "detector": return [{"type": i["type"], "point": i["point"]} for i in pred.get("items", []) if i.get("conf", 1) >= CONF]
    if name == "truth": return env.gt["items"]
    if name == "label model": return lab.get("items", []) if isinstance(lab.get("items"), list) else []
    if name == "gemini tiles":
        f = f"results/gemini_tiles_{a.model}_v2/{p}.json"
        if not os.path.exists(f): return None
        t = json.load(open(f)); ppm = (lab.get("scale") or {}).get("px_per_m") or 59.06
        return tiles.merge_owned(t["per_tile"], t["boxes"], ppm=ppm)


if a.cubi:
    a.key = "cubi_eval"
    if a.plans_file == "split_full.json": a.plans_file = "split_cubi.json"
plans = [p for p in json.load(open(a.plans_file))[a.key] if os.path.exists(os.path.join(a.preds, p + ".json"))]
print(len(plans), "plans with predictions")
cfg = dict(CFG, snap=False, gt_dir=a.gt_cubi if a.cubi else CFG["gt_dir"])
envs = {p: FloorPlanEnv(p, cfg) for p in plans}
preds = {p: json.load(open(os.path.join(a.preds, p + ".json"))) for p in plans}
setups = [("truth", "label model", "detector"), ("gemini", "label model", "label model"), ("gemini", "label model", "detector"),
          ("qwen", "label model", "detector")] if a.cubi else [("truth", "seg", "truth"), ("truth", "seg", "detector"),
          ("gemini", "label model", "label model"), ("gemini", "seg", "detector"), ("gemini", "seg", "gemini tiles"),
          ("qwen", "label model", "label model"), ("qwen", "seg", "detector")]
CONF = 0.3
if a.cubi and "," in a.confs:                      # report several detector cut-offs as separate setups
    pass
variants = a.variants.split(",")
if variants != ["base"] and not a.cubi:
    setups = [s for s in setups if s[1] == "seg" and s[2] in ("truth", "detector") and s != ("truth", "seg", "detector")]
    if not any(v in ("base", "both", "total") for v in variants): setups = [s for s in setups if s[0] != "truth"]
runs = [(v, float(cf)) for v in variants for cf in a.confs.split(",")]
for var, CONF in runs:
  if len(runs) > 1: print(f"---- variant: {var} | detector cut-off {CONF} ----")
  for lab_name, room_src, item_src in setups:
    res = []; matched = total = 0; nsplit = 0; ncal = 0; scale_err = []
    for p in plans:
        env = envs[p]; lab = label_source(lab_name, p, env)
        if lab is None: continue
        lab["rooms"] = lab.get("rooms", [])
        if room_src == "seg":
            segs = preds[p].get("rooms", [])
            ppm = (lab.get("scale") or {}).get("px_per_m")
            if var.startswith("split") or var == "both":
                segs, k = split_merged(lab["rooms"], segs, ppm); nsplit += k
            if var in ("total", "both"):
                lab["total_indoor_m2"] = total_indoor(lab["rooms"], segs, ppm)
            matched += assign(lab["rooms"], segs, strict=lab_name == "truth"); total += len(lab["rooms"])
            if var == "split_gtscale":                                  # diagnostic: is the language model's scale the problem?
                lab["scale"] = dict(env.gt.get("scale") or {})
            if var == "split_snapscale":                                # Swiss renders come at 59.06 or 29.53 px/m: snap a close estimate
                m = (lab.get("scale") or {}).get("px_per_m")
                if m:
                    k = min((59.0551, 29.5256), key=lambda c: abs(m / c - 1))
                    if abs(m / k - 1) < 0.06: lab["scale"] = {"found": True, "source": "snapped", "px_per_m": k}; ncal += 1
            if var == "split_bar":                                      # general: measure the drawing's scale bar in pixels
                import numpy as np
                from scale_bar import refine_scale
                r = refine_scale(np.asarray(env.img.convert("L")), (lab.get("scale") or {}).get("px_per_m"))
                if r: lab["scale"] = {"found": True, "source": "scale_bar_measured", "px_per_m": r}; ncal += 1
            if var in ("split_calib", "split_calib1"):                 # general: scale from segmented room sizes vs printed areas
                from answer import polygon_area_px
                rs = [math.sqrt(abs(polygon_area_px(r["polygon"])) / r["printed_area_m2"]) for r in lab["rooms"]
                      if r.get("printed_area_m2") and len(r.get("polygon") or []) >= 3 and r["printed_area_m2"] > 2]
                m0 = (lab.get("scale") or {}).get("px_per_m")
                if m0: rs = [x for x in rs if abs(x / m0 - 1) < 0.15]   # drop rooms matched to the wrong segment
                if len(rs) >= (3 if var == "split_calib" else 1):
                    lab["scale"] = {"found": True, "source": "calibrated", "px_per_m": statistics.median(rs)}; ncal += 1
            g_s = (env.gt.get("scale") or {}).get("px_per_m"); m_s = (lab.get("scale") or {}).get("px_per_m")
            if g_s and m_s: scale_err.append(abs(m_s / g_s - 1))
        its = items_source(item_src, p, env, preds[p], lab)
        if its is None: continue
        lab["items"] = its
        i = env.score(lab); res.append(dict(plan=p, reward=i["reward"], rows=i["rows"]))
    if res:
        t = table(res, f"labels: {lab_name:6s} | rooms: {room_src:11s} | items: {item_src:12s}")
        print(t.split("\n")[0] + (f" | rooms given a segment {matched}/{total}" if total else "") + (f" | segments split {nsplit}" if nsplit else "") + (f" | plans calibrated {ncal}" if ncal else ""))
        if scale_err:
            e = sorted(scale_err)
            print(f"    scale vs truth: within 0.5% {sum(x <= .005 for x in e)}/{len(e)}, within 1% {sum(x <= .01 for x in e)}/{len(e)}, median error {statistics.median(e):.2%}")
        for line in t.split("\n")[1:]:
            if any(k in line for k in ("area_", "count_item", "count_roomtype")): print(line)
