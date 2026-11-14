"""Run Gemini (native Google API) as an agent in FloorPlanEnv. Same env, tools and scoring as run_agent.py.

    pip install google-genai
    export GEMINI_API_KEY=...
    python run_gemini.py --model gemini-3.8-flash --split test --limit 1
"""
import argparse, base64, io, json, os, re, time
from google import genai
from google.genai import types
from env import FloorPlanEnv

CFG = dict(images_dir=os.environ.get("IMAGES_DIR", "data/swiss_dwellings"),
           gt_dir=os.environ.get("GT_DIR", "gt"),
           questions_path=os.environ.get("QUESTIONS", "data/questions.jsonl"))

BASE = [
    types.FunctionDeclaration(
        name="crop",
        description="Zoom into a region of the ORIGINAL image (pixel coordinates). Returns a zoomed image.",
        parameters={"type": "object", "properties": {k: {"type": "number"} for k in ("x0", "y0", "x1", "y1")},
                    "required": ["x0", "y0", "x1", "y1"]}),
    types.FunctionDeclaration(
        name="measure_area",
        description="Exact area in m² of a polygon given in original image pixels, using px_per_m.",
        parameters={"type": "object", "properties": {
            "polygon": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "px_per_m": {"type": "number"}}, "required": ["polygon", "px_per_m"]}),
]
CV_DECL = types.FunctionDeclaration(
        name="segment_rooms",
        description="Classic computer-vision room segmentation. Give one seed point [x, y] (original pixels) inside each "
                    "room, e.g. at its label. Returns each room's polygon and pixel area (split at walls and door "
                    "openings) plus an overlay image. Areas are in pixels: divide by px_per_m**2.",
        parameters={"type": "object", "properties": {
            "seeds": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}},
            "required": ["seeds"]})
SUBMIT = [
    types.FunctionDeclaration(
        name="submit_schematic",
        description="Submit the final schematic as a JSON string following the schema. Ends the episode.",
        parameters={"type": "object", "properties": {"schematic_json": {"type": "string"}},
                    "required": ["schematic_json"]}),
]

def png_bytes(img, max_side):
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)))
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()

def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}

def run_episode(client, model, plan, max_turns=15, max_side=2048, cv_tool=False, snap=False):
    env = FloorPlanEnv(plan, dict(CFG, cv_tool=cv_tool, snap=snap))
    TOOLS = types.Tool(function_declarations=BASE + ([CV_DECL] if cv_tool else []) + SUBMIT)
    obs = env.reset()
    W, H = obs["image_size"]
    note = ""
    if max(W, H) > max_side:
        s = max_side / max(W, H)
        note = (f" The overview image is downscaled by {s:.3f}; multiply its coordinates by {1/s:.3f} for original "
                f"pixels, or use crop for exact detail.")
    contents = [types.Content(role="user", parts=[
        types.Part.from_text(text=f"Floor plan image, original size {W}x{H} pixels.{note} "
                                  f"You have at most {max_turns} turns. Zoom only where symbols are unclear "
                                  f"(crop several regions per turn if needed), then call submit_schematic."),
        types.Part.from_bytes(data=png_bytes(env.img, max_side), mime_type="image/png")])]
    config = types.GenerateContentConfig(
        system_instruction=obs["instructions"], tools=[TOOLS],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    calls_log = []
    schematic, usage, turn = None, {"prompt_tokens": 0, "completion_tokens": 0}, 0
    force = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
        mode="ANY", allowed_function_names=["submit_schematic"]))
    for turn in range(max_turns):
        if turn >= max_turns - 1:   # last turn: the only allowed action is submitting
            config.tool_config = force
        resp = client.models.generate_content(model=model, contents=contents, config=config)
        um = resp.usage_metadata
        if um:
            usage["prompt_tokens"] += um.prompt_token_count or 0
            usage["completion_tokens"] += (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
        cand = resp.candidates[0] if resp.candidates else None
        if not cand or not cand.content or not cand.content.parts:
            break
        contents.append(cand.content)
        calls = [p.function_call for p in cand.content.parts if p.function_call]
        if not calls:
            schematic = parse_json("".join(p.text or "" for p in cand.content.parts))
            break
        reply_parts, images = [], []
        for fc in calls:
            args = dict(fc.args or {})
            if fc.name == "submit_schematic":
                raw = args.get("schematic_json", "")
                schematic = parse_json(raw) if isinstance(raw, str) else raw
                reply_parts.append(types.Part.from_function_response(name=fc.name, response={"result": "submitted"}))
                continue
            calls_log.append(fc.name)
            out = env.step({"tool": fc.name, "args": args})
            if "image_png_b64" in out:
                images.append(base64.b64decode(out["image_png_b64"]))
                out = {"note": out["note"] + " (image attached)"}
            reply_parts.append(types.Part.from_function_response(name=fc.name, response=out))
        for b in images:
            reply_parts.append(types.Part.from_bytes(data=b, mime_type="image/png"))
        left = max_turns - turn - 1
        reply_parts.append(types.Part.from_text(
            text=f"Turns left: {left}. " + ("Submit your schematic NOW with submit_schematic." if left <= 2 else "")))
        contents.append(types.Content(role="user", parts=reply_parts))
        if schematic is not None:
            break
    _, reward, _, info = env.step({"submit": schematic or {}})
    info.update(plan=plan, turns=turn + 1, usage=usage, schematic=schematic, tool_calls=calls_log)
    return info

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=1, help="episodes in parallel")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--cv-tool", action="store_true", help="also give the model the segment_rooms CV tool")
    ap.add_argument("--snap", action="store_true", help="snap submitted room outlines to walls before scoring")
    a = ap.parse_args()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    plans = json.load(open("split.json"))[a.split]
    if a.limit: plans = plans[:a.limit]
    outdir = os.path.join("results", a.model.replace("/", "_") + ("+cv" if a.cv_tool else "")); os.makedirs(outdir, exist_ok=True)
    summary = []
    from concurrent.futures import ThreadPoolExecutor
    def job(kp):
        k, plan = kp
        t0 = time.time()
        try:
            info = run_episode(client, a.model, plan, max_turns=a.max_turns, max_side=a.max_side, cv_tool=a.cv_tool, snap=a.snap)
        except Exception as ex:
            print("ERR", plan, repr(ex)[:500], flush=True); return None
        info["seconds"] = round(time.time() - t0, 1)
        json.dump(info, open(os.path.join(outdir, f"{plan}_run{k}.json"), "w"), ensure_ascii=False, indent=1)
        print(f"run{k} {plan:34} reward={info['reward']:.3f} acc={info['answer_acc']:.3f} "
              f"itemsF1={info['items_f1']:.3f} turns={info['turns']} tokens={info['usage']} {info['seconds']}s", flush=True)
        return info
    jobs = [(k, p) for k in range(a.runs) for p in plans]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        summary = [r for r in ex.map(job, jobs) if r]
    if summary:
        avg = lambda k: sum(s[k] for s in summary) / len(summary)
        print(f"\nMEAN over {len(summary)} episodes: reward={avg('reward'):.3f} acc={avg('answer_acc'):.3f} itemsF1={avg('items_f1'):.3f}")
