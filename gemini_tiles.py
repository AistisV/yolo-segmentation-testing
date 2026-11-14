"""Is item counting a METHOD problem? Gemini, unchanged, on the same tiles as eval_tiles.py
(rooms from the whole image; items tile by tile, merged by code). Compare with gemini_policy.py (whole image).

    export GEMINI_API_KEY=...  IMAGES_DIR=... QUESTIONS=... GT_DIR=gt_full
    python3 gemini_tiles.py --plans-file split_full.json --key test
"""
import argparse, io, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
import eval_tiles
from eval_tiles import run_plan
from eval_policy import table

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemini-3.8-flash")
ap.add_argument("--plans-file", default="split_full.json"); ap.add_argument("--key", default="test")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--tag", default="")
a = ap.parse_args()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def ask(_client, _model, img, prompt, fmt, max_tokens, guided=True):
    buf = io.BytesIO(); img.save(buf, "PNG")
    for attempt in range(4):
        try:
            r = client.models.generate_content(model=a.model, contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"), prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=max(max_tokens, 8000), temperature=0,
                                                   media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH))
            break
        except Exception:
            if attempt == 3: raise
            time.sleep(10 * (attempt + 1))
    u = r.usage_metadata
    fin = str(r.candidates[0].finish_reason) if r.candidates else ""
    return (r.text or ""), ("length" if "MAX_TOKENS" in fin else "stop"), (getattr(u, "prompt_token_count", 0) or 0,
                                                                          (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0))


eval_tiles.ask = ask
plans = json.load(open(a.plans_file))[a.key]
if a.limit: plans = plans[:a.limit]
outdir = os.path.join("results", "gemini_tiles_" + a.model + (("_" + a.tag) if a.tag else "")); os.makedirs(outdir, exist_ok=True)
res, todo = [], []
for p in plans:
    f = os.path.join(outdir, p + ".json")
    if os.path.exists(f): res.append(json.load(open(f)))
    else: todo.append(p)
print(f"{len(plans)} plans, {len(res)} done, {len(todo)} to run", flush=True)
with ThreadPoolExecutor(16) as pool, ThreadPoolExecutor(a.workers) as ex:
    futs = {ex.submit(run_plan, None, a.model, p, pool): p for p in todo}
    for i, f in enumerate(as_completed(futs)):
        p = futs[f]
        try:
            r = f.result(); json.dump(r, open(os.path.join(outdir, p + ".json"), "w"), ensure_ascii=False)
        except Exception as e:
            r = dict(plan=p, sample="greedy", reward=0.0, acc=0.0, items_f1=0.0, rows=[], error="PLAN " + repr(e)[:200])
        res.append(r)
        print(f"[{i+1}/{len(todo)}] {p:32s} reward={r['reward']:.3f} itemsF1={r.get('items_f1',0):.3f} tiles={r.get('n_tiles','')} {r.get('error') or ''}", flush=True)
print("\n=== gemini on tiles: " + a.model + " ===")
print(table(res, a.key))
