"""Areas: does zooming in per room give tighter outlines? For each room from the whole-image answer
(results/gemini_single_*), crop around it, enlarge, draw the rough outline in red, and ask for the precise
outline of that one room. Items come from the tiled run (v2) with the ownership merge. Snapping and scoring
unchanged. Rooms whose printed area was read are left alone.

    python3 gemini_room_zoom.py --key test
"""
import argparse, io, json, os, time
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageDraw, Image
from google import genai
from google.genai import types
from env import FloorPlanEnv
from eval_policy import CFG, table
from policy_io import parse_json
import tiles

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemini-3.8-flash")
ap.add_argument("--plans-file", default="split_full.json"); ap.add_argument("--key", default="test")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--margin-m", type=float, default=1.2)
ap.add_argument("--view", type=int, default=1280); ap.add_argument("--tag", default="")
a = ap.parse_args()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
PROMPT = ("This is an enlarged crop of an architectural floor plan. The RED line is a rough outline of room {num} ({label}).\n"
          "Return the PRECISE outline of that one room's floor area: follow the inner face of its walls exactly; where the room is open "
          "to another space with no wall, keep the boundary where the red line puts it.\n"
          'Output only JSON: {{"polygon": [[x, y], ...]}} with RELATIVE integer coordinates 0..1000 in THIS crop '
          "(0,0 = top-left, 1000,1000 = bottom-right). Use as few corners as the room really has.")


def ask(img, prompt):
    buf = io.BytesIO(); img.save(buf, "PNG")
    for attempt in range(4):
        try:
            r = client.models.generate_content(model=a.model, contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"), prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=4000, temperature=0,
                                                   media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH))
            u = r.usage_metadata
            return r.text or "", (getattr(u, "prompt_token_count", 0) or 0) + (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
        except Exception:
            if attempt == 3: raise
            time.sleep(8 * (attempt + 1))


def refine_room(img, room, ppm):
    xs = [p[0] for p in room["polygon"]]; ys = [p[1] for p in room["polygon"]]
    m = a.margin_m * ppm
    x0, y0, x1, y1 = max(0, min(xs) - m), max(0, min(ys) - m), min(img.width, max(xs) + m), min(img.height, max(ys) + m)
    crop = img.crop((int(x0), int(y0), int(x1), int(y1))).convert("RGB")
    up = min(4.0, a.view / max(crop.size))
    crop = crop.resize((max(1, round(crop.width * up)), max(1, round(crop.height * up))), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    d.line([((x - int(x0)) * up, (y - int(y0)) * up) for x, y in room["polygon"] + room["polygon"][:1]], fill=(255, 0, 0), width=3)
    text, tok = ask(crop, PROMPT.format(num=room.get("number") or "?", label=room.get("label") or "?"))
    poly = (parse_json(text) or {}).get("polygon") or []
    cw, ch = crop.size
    try:
        out = [[int(x0) + float(x) / 1000 * cw / up, int(y0) + float(y) / 1000 * ch / up] for x, y in poly]
    except Exception:
        out = []
    return (out if len(out) >= 3 else room["polygon"]), tok


plans = json.load(open(a.plans_file))[a.key]
if a.limit: plans = plans[:a.limit]
outdir = os.path.join("results", "gemini_roomzoom_" + a.model + (("_" + a.tag) if a.tag else "")); os.makedirs(outdir, exist_ok=True)
res, before = [], []
with ThreadPoolExecutor(16) as pool:
    for p in plans:
        f = os.path.join(outdir, p + ".json")
        env = FloorPlanEnv(p, CFG); W, H = env.img.size
        w = json.load(open(f"results/gemini_single_{a.model}/{p}.json")); t = json.load(open(f"results/gemini_tiles_{a.model}_v2/{p}.json"))
        s = json.loads(json.dumps(w["norm_schematic"]))
        for r in s.get("rooms", []): r["polygon"] = [[x * W / 1000, y * H / 1000] for x, y in r["polygon"]]
        s["coord_space"] = "pixel"
        ppm = (s.get("scale") or {}).get("px_per_m") or 59.06
        s["items"] = tiles.merge_owned(t["per_tile"], t["boxes"], ppm=ppm)
        b = env.score(s); before.append(dict(plan=p, reward=b["reward"], rows=b["rows"]))
        if os.path.exists(f):
            res.append(json.load(open(f))); continue
        t0 = time.time()
        todo = [r for r in s["rooms"] if len(r["polygon"]) >= 3 and r.get("printed_area_m2") is None]
        outs = list(pool.map(lambda r: refine_room(env.img, r, ppm), todo))
        tok = 0
        for r, (poly, k) in zip(todo, outs): r["polygon"] = poly; tok += k
        i = env.score(s)
        r = dict(plan=p, sample="greedy", reward=i["reward"], rows=i["rows"], schematic=s, n_refined=len(todo),
                 seconds=round(time.time() - t0, 1), tokens=dict(prompt=tok, completion=0))
        json.dump(r, open(f, "w"), ensure_ascii=False); res.append(r)
        print(f"{p:32s} before {b['reward']:.3f} -> after {i['reward']:.3f}  ({len(todo)} rooms refined, {r['seconds']:.0f}s)", flush=True)
print(table(before, "BEFORE (whole-image rooms + tile items)"))
print(table(res, "AFTER room zoom"))
