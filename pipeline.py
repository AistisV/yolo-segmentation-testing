"""Best system so far (Sat 12:00): Gemini reads the plan, our code does the precise part.

  rooms, apartments, scale : Gemini on the whole image (policy_prompt.md)
  doors, windows, fixtures : Gemini tile by tile (items_prompt.md), merged by code (ownership rule)
  room outlines            : snapped onto the walls by code
  no printed scale         : scale estimated from printed room areas; if impossible, areas are reported as unknown
  answers                  : answer.py (counts, room / label / apartment / total areas), with highlights

    export GEMINI_API_KEY=...
    # one image, your own questions
    python3 pipeline.py --image plan.png -q "How many doors are shown on this floor plan?" -q "What is the net floor area of room 1.02 ('Bad') in m²?" --overlay
    # a set of dataset plans, scored against Stoa's answers (uses IMAGES_DIR, QUESTIONS)
    python3 pipeline.py --plans-file split_full.json --key test --reuse
--reuse composes from earlier Gemini runs (results/gemini_single_*, results/gemini_tiles_*_v2) when they exist (no new cost).
"""
import argparse, collections, io, json, math, os, statistics, time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw
from answer import answer, correct, parse, room_type, OUTDOOR
from policy_io import parse_json, sanitize
from make_sft import prompt_for
from cv_rooms import snap_polygon, compute_masks
import tiles

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
_client = None


def client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def gemini(img, prompt, max_tokens=32000, high_res=False):
    from google.genai import types
    buf = io.BytesIO(); img.convert("RGB").save(buf, "PNG")
    cfg = dict(response_mime_type="application/json", max_output_tokens=max_tokens, temperature=0)
    if high_res: cfg["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_HIGH
    for attempt in range(4):
        try:
            r = client().models.generate_content(model=MODEL, contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"), prompt],
                                                 config=types.GenerateContentConfig(**cfg))
            u = r.usage_metadata
            return r.text or "", (getattr(u, "prompt_token_count", 0) or 0), (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
        except Exception:
            if attempt == 3: raise
            time.sleep(8 * (attempt + 1))


def read_rooms(img):
    W, H = img.size
    text, ti, to = gemini(img, prompt_for(W, H))
    s = sanitize(parse_json(text))
    for r in s["rooms"]:
        r["polygon"] = [[x * W / 1000, y * H / 1000] for x, y in r["polygon"]]
    s["items"] = []; s["coord_space"] = "pixel"
    return s, ti, to


def read_items(img, pool):
    W, H = img.size
    boxes = [b for b in tiles.tile_boxes(W, H) if tiles.has_ink(img, b)] or tiles.tile_boxes(W, H)
    outs = list(pool.map(lambda b: gemini(tiles.view(img, b), tiles.ITEMS_PROMPT, 8000, True), boxes))
    per = [tiles.tile_items_to_pixels(sanitize({"items": parse_json(t)})["items"], b) for b, (t, _, _) in zip(boxes, outs)]
    return per, boxes, sum(o[1] for o in outs), sum(o[2] for o in outs)


def finish_schematic(s, per_tile, boxes, img):
    """Scale fallback, item merge, snapping. Returns schematic + notes."""
    notes = []
    ppm = s["scale"].get("px_per_m")
    if not (ppm and 5 < ppm < 1000):
        est = []
        for r in s["rooms"]:
            if r.get("printed_area_m2") and len(r["polygon"]) >= 3:
                a = abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(r["polygon"], r["polygon"][1:] + r["polygon"][:1]))) / 2
                if a > 0: est.append(math.sqrt(a / r["printed_area_m2"]))
        if len(est) >= 2:
            ppm = statistics.median(est)
            s["scale"] = {"found": True, "source": "printed_area", "px_per_m": ppm}
            notes.append(f"no printed scale: estimated {ppm:.1f} px/m from {len(est)} printed room areas")
        else:
            ppm = None
            s["scale"] = {"found": False, "source": "none", "px_per_m": None}
            notes.append("no scale and no printed room areas: areas that are not printed cannot be measured")
    s["items"] = tiles.merge_owned(per_tile, boxes, ppm=ppm)
    masks = compute_masks(img, 1)
    rad = int(0.45 * ppm) if ppm else 25
    for r in s["rooms"]:
        if len(r["polygon"]) >= 3:
            try: r["polygon"] = snap_polygon(img, r["polygon"], radius_px=rad, T=1, masks=masks)["polygon"]
            except Exception: pass
    return s, notes


def evidence(s, question):
    """Which rooms / items an answer is based on (for the highlight)."""
    kind, arg = parse(question)
    norm = lambda v: (v or "").strip().lower()
    apt = lambda v: ((v or "").replace("App.", "").replace("Whg.", "").replace("Ap.", "").strip().split()[-1:] or [""])[0]
    R, I = s["rooms"], s["items"]
    if kind == "count_item": return [], [i for i in I if i["type"] == arg]
    if kind == "count_items": return [], [i for i in I if i["type"] in arg]
    if kind == "count_label": return [r for r in R if norm(r.get("label")) == norm(arg)], []
    if kind == "count_roomtype": return [r for r in R if arg in room_type(r.get("label"))], []
    if kind == "area_room": return [r for r in R if r.get("number") == arg][:1], []
    if kind == "area_label": return [r for r in R if norm(r.get("label")) == norm(arg)], []
    if kind == "area_apartment": return [r for r in R if apt(r.get("apartment")) == arg and r.get("number") and not OUTDOOR.search(r.get("label") or "")], []
    if kind == "area_total_indoor": return [r for r in R if r.get("number") and not OUTDOOR.search(r.get("label") or "")], []
    return R, []


def overlay(img, rooms, items, title, path):
    im = img.convert("RGB").copy(); d = ImageDraw.Draw(im, "RGBA")
    for r in rooms:
        if len(r["polygon"]) >= 3:
            d.polygon([tuple(p) for p in r["polygon"]], fill=(255, 140, 0, 70), outline=(230, 80, 0, 255))
            xs = [p[0] for p in r["polygon"]]; ys = [p[1] for p in r["polygon"]]
            d.text((sum(xs) / len(xs), sum(ys) / len(ys)), f"{r.get('number') or ''} {r.get('label') or ''}", fill=(120, 30, 0, 255))
    for i in items:
        x, y = i["point"]; d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(0, 90, 255, 255), width=3)
    d.rectangle([0, 0, im.width, 28], fill=(255, 255, 255, 220)); d.text((6, 8), title[:200], fill=(0, 0, 0, 255))
    im.save(path)


def run_image(img, pool, plan=None, reuse=False):
    t0 = time.time(); ti = to = 0
    W, H = img.size
    single = f"results/gemini_single_{MODEL}/{plan}.json" if plan else ""
    tiled = f"results/gemini_tiles_{MODEL}_v2/{plan}.json" if plan else ""
    if reuse and plan and os.path.exists(single):
        w = json.load(open(single)); s = sanitize(w["norm_schematic"])
        for r in s["rooms"]: r["polygon"] = [[x * W / 1000, y * H / 1000] for x, y in r["polygon"]]
        s["items"] = []; s["coord_space"] = "pixel"; ti += w.get("tokens", {}).get("prompt", 0); to += w.get("tokens", {}).get("completion", 0)
        fr = None
    else:
        fr = pool.submit(read_rooms, img)
    if reuse and plan and os.path.exists(tiled):
        t = json.load(open(tiled)); per, boxes = t["per_tile"], t["boxes"]
        ti += t.get("tokens", {}).get("prompt", 0); to += t.get("tokens", {}).get("completion", 0)   # includes its rooms call
    else:
        per, boxes, a, b = read_items(img, pool); ti += a; to += b
    if fr is not None:
        s, a, b = fr.result(); ti += a; to += b
    s, notes = finish_schematic(s, per, boxes, img)
    return s, notes, dict(prompt_tokens=ti, output_tokens=to, seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("-q", "--question", action="append", default=[])
    ap.add_argument("--plans-file"); ap.add_argument("--key", default="test"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse", action="store_true"); ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--out-dir", default="results/pipeline")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        if a.image:
            img = Image.open(a.image).convert("RGB")
            s, notes, cost = run_image(img, pool)
            name = os.path.splitext(os.path.basename(a.image))[0]
            out = dict(image=a.image, notes=notes, cost=cost, answers=[], schematic=s)
            for i, q in enumerate(a.question):
                kind, val = answer(s, q)
                rooms, items = evidence(s, q)
                out["answers"].append(dict(question=q, kind=kind, answer=val, rooms=[r.get("number") or r.get("label") for r in rooms], n_items=len(items)))
                print(f"Q: {q}\nA: {val}   ({kind}; based on {len(rooms)} rooms, {len(items)} items)")
                if a.overlay:
                    p = os.path.join(a.out_dir, f"{name}_q{i+1}.png"); overlay(img, rooms, items, f"{q} -> {val}", p); print("   highlight:", p)
            for n in notes: print("note:", n)
            print("cost:", cost)
            json.dump(out, open(os.path.join(a.out_dir, name + ".json"), "w"), ensure_ascii=False, indent=1)
        else:
            from env import FloorPlanEnv
            from eval_policy import CFG, table
            plans = json.load(open(a.plans_file))[a.key]
            if a.limit: plans = plans[:a.limit]
            test = set(json.load(open("split.json"))["test"]); res = []
            for k, p in enumerate(plans):
                env = FloorPlanEnv(p, CFG)
                try:
                    s, notes, cost = run_image(env.img, pool, plan=p, reuse=a.reuse)
                    info = env.score(dict(s, coord_space="pixel"))
                    r = dict(plan=p, sample="greedy", reward=info["reward"], rows=info["rows"], notes=notes, seconds=cost["seconds"],
                             tokens=dict(prompt=cost["prompt_tokens"], completion=cost["output_tokens"]))
                except Exception as e:
                    r = dict(plan=p, sample="greedy", reward=0.0, rows=[], error=repr(e)[:200])
                json.dump(r, open(os.path.join(a.out_dir, p + ".json"), "w"), ensure_ascii=False)
                res.append(r)
                print(f"[{k+1}/{len(plans)}] {p:32s} reward={r['reward']:.3f} {r.get('error') or ''} {'; '.join(r.get('notes', []))}", flush=True)
            print("\n=== pipeline (Gemini rooms + tiled items + snapping) ===")
            t = [r for r in res if r["plan"] in test]; h = [r for r in res if r["plan"] not in test]
            if t: print(table(t, "15 test plans"))
            if h: print(table(h, "other plans"))
            print(table(res, "all"))
