"""Gemini, single-turn: same prompt, same scoring, same plans as our trained model (fair comparison).

    export GEMINI_API_KEY=...
    modal volume get --force jx split_full.json split_full.json
    modal volume get --force jx gt_full gt_full
    export IMAGES_DIR=~/Downloads/junction-dataset/data/swiss_dwellings QUESTIONS=~/Downloads/junction-dataset/questions.jsonl GT_DIR=gt_full
    python3 gemini_policy.py --plans-file split_full.json --key eval --workers 8

Resumable (one file per plan in results/gemini_single_<model>/). Prints the same tables as eval_policy.py.
"""
import argparse, io, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from env import FloorPlanEnv
from eval_policy import CFG, score_text, table
from make_sft import prompt_for

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemini-3.8-flash")
ap.add_argument("--plans-file", default="split_full.json"); ap.add_argument("--key", default="eval")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=32000)
a = ap.parse_args()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
here = os.path.dirname(os.path.abspath(__file__))
test = set(json.load(open(os.path.join(here, "split.json")))["test"])
plans = json.load(open(a.plans_file))[a.key]
if a.limit: plans = plans[:a.limit]
outdir = os.path.join("results", "gemini_single_" + a.model); os.makedirs(outdir, exist_ok=True)


def one(plan):
    env = FloorPlanEnv(plan, CFG)
    W, H = env.img.size
    buf = io.BytesIO(); env.img.save(buf, "PNG")
    t0 = time.time()
    for attempt in range(4):
        try:
            r = client.models.generate_content(model=a.model, contents=[
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"), prompt_for(W, H)],
                config=types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=a.max_tokens, temperature=0))
            break
        except Exception as e:
            if attempt == 3: raise
            time.sleep(10 * (attempt + 1))
    text = r.text or ""
    info, s = score_text(env, text)
    u = r.usage_metadata
    fin = str(r.candidates[0].finish_reason) if r.candidates else ""
    return dict(plan=plan, sample="greedy", reward=info["reward"], acc=info["answer_acc"], items_f1=info["items_f1"], rows=info["rows"],
                error=info.get("error"), norm_schematic=s, finish="length" if "MAX_TOKENS" in fin else "stop",
                seconds=round(time.time() - t0, 1), image_size=[W, H],
                tokens=dict(prompt=getattr(u, "prompt_token_count", 0) or 0,
                            completion=(getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)))


res, todo = [], []
for p in plans:
    f = os.path.join(outdir, p + ".json")
    if os.path.exists(f): res.append(json.load(open(f)))
    else: todo.append(p)
print(f"{len(plans)} plans, {len(res)} done, {len(todo)} to run", flush=True)
with ThreadPoolExecutor(a.workers) as ex:
    futs = {ex.submit(one, p): p for p in todo}
    for i, f in enumerate(as_completed(futs)):
        p = futs[f]
        try:
            r = f.result(); json.dump(r, open(os.path.join(outdir, p + ".json"), "w"), ensure_ascii=False)
        except Exception as e:
            r = dict(plan=p, sample="greedy", reward=0.0, acc=0.0, items_f1=0.0, rows=[], error="REQUEST " + repr(e)[:200])
        res.append(r)
        print(f"[{i+1}/{len(todo)}] {p:32s} reward={r['reward']:.3f} acc={r.get('acc',0):.3f} {r.get('finish','')} {r.get('error') or ''}", flush=True)
print("\n=== gemini single-turn: " + a.model + " ===")
print(table([r for r in res if r["plan"] in test], "15 test plans"))
print(table([r for r in res if r["plan"] not in test], "holdout plans"))
print(table(res, "all"))
