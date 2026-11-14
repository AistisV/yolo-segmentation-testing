"""Two-part policy evaluation: rooms from the whole image, items tile by tile, merged by code, scored by the env.

    vllm serve <model> --served-model-name policy ...
    python eval_tiles.py --plans-file split.json --key eval --tag tiles --out-dir results/tiles

Resumable (one file per plan). --summary-only prints the tables from existing files.
"""
import argparse, base64, io, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from env import FloorPlanEnv
from eval_policy import CFG, table
from policy_io import parse_json, sanitize
from policy_schema import ROOMS_FORMAT, ITEMS_FORMAT
from tiles import ITEMS_PROMPT, ROOMS_PROMPT, tile_boxes, view, has_ink, tile_items_to_pixels, merge_tiles, merge_owned


def ask(client, model, img, prompt, fmt, max_tokens, guided=True):
    buf = io.BytesIO(); img.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    extra = {"response_format": fmt} if guided else {}
    r = client.chat.completions.create(model=model, temperature=0.0, max_tokens=max_tokens, **extra, messages=[
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}])
    u = r.usage
    return r.choices[0].message.content or "", r.choices[0].finish_reason, (u.prompt_tokens, u.completion_tokens) if u else (0, 0)


def run_plan(client, model, plan, pool, guided=True, rooms_tokens=12000, tile_tokens=1500):
    env = FloorPlanEnv(plan, CFG)
    W, H = env.img.size
    t0 = time.time()
    boxes = [b for b in tile_boxes(W, H) if has_ink(env.img, b)]
    fr = pool.submit(ask, client, model, env.img, ROOMS_PROMPT.replace("{W}", str(W)).replace("{H}", str(H)), ROOMS_FORMAT, rooms_tokens, guided)
    ft = [pool.submit(ask, client, model, view(env.img, b), ITEMS_PROMPT, ITEMS_FORMAT, tile_tokens, guided) for b in boxes]
    ptok = ctok = 0; cut = 0; errs = []
    try:
        text, fin, (p, c) = fr.result(); ptok += p; ctok += c; cut += fin == "length"
        s = sanitize(parse_json(text))
    except Exception as e:
        s = sanitize({}); errs.append("rooms: " + repr(e)[:150])
    for r in s["rooms"]:
        r["polygon"] = [[x * W / 1000, y * H / 1000] for x, y in r["polygon"]]
    per_tile = []
    for b, f in zip(boxes, ft):
        try:
            text, fin, (p, c) = f.result(); ptok += p; ctok += c; cut += fin == "length"
            per_tile.append(tile_items_to_pixels(sanitize({"items": parse_json(text)})["items"], b))
        except Exception as e:
            per_tile.append([]); errs.append("tile: " + repr(e)[:150])
    ppm = (s.get("scale") or {}).get("px_per_m")
    s["items"] = merge_owned(per_tile, boxes, ppm=ppm if ppm and 10 < ppm < 400 else None); s["coord_space"] = "pixel"
    try:
        info = env.score(s)
    except Exception as e:
        info = dict(reward=0.0, answer_acc=0.0, items_f1=0.0, rows=[]); errs.append("score: " + repr(e)[:150])
    return dict(plan=plan, sample="greedy", reward=info["reward"], acc=info["answer_acc"], items_f1=info["items_f1"], rows=info["rows"],
                error="; ".join(errs[:3]) or None, schematic=s, finish="length" if cut else "stop", n_tiles=len(boxes),
                boxes=boxes, per_tile=per_tile,
                seconds=round(time.time() - t0, 1), image_size=[W, H], tokens=dict(prompt=ptok, completion=ctok))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="policy"); ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--plans-file", default=""); ap.add_argument("--key", default="eval"); ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="tiles"); ap.add_argument("--out-dir", default="")
    ap.add_argument("--no-guided", action="store_true"); ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    test = set(json.load(open(os.path.join(here, "split.json")))["test"])
    plans = json.load(open(a.plans_file))[a.key] if a.plans_file else json.load(open(os.path.join(here, "split.json")))[a.split]
    if a.limit: plans = plans[:a.limit]
    plans = plans[a.shard::a.nshards]
    outdir = a.out_dir or os.path.join("results", a.tag); os.makedirs(outdir, exist_ok=True)
    res, todo = [], []
    for p in plans:
        f = os.path.join(outdir, p + ".json")
        if os.path.exists(f):
            try: res.append(json.load(open(f))); continue
            except Exception: pass
        todo.append(p)
    print(f"{len(plans)} plans, {len(res)} done, {len(todo)} to run", flush=True)
    if not a.summary_only and todo:
        from openai import OpenAI
        client = OpenAI(base_url=a.base_url, api_key="x", timeout=3600, max_retries=1)
        with ThreadPoolExecutor(48) as pool, ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(run_plan, client, a.model, p, pool, not a.no_guided): p for p in todo}
            for i, f in enumerate(as_completed(futs)):
                p = futs[f]
                try:
                    r = f.result(); json.dump(r, open(os.path.join(outdir, p + ".json"), "w"), ensure_ascii=False)
                except Exception as e:
                    r = dict(plan=p, sample="greedy", reward=0.0, acc=0.0, items_f1=0.0, rows=[], error="PLAN " + repr(e)[:200])
                res.append(r)
                print(f"[{i+1}/{len(todo)}] {p:32s} reward={r['reward']:.3f} acc={r.get('acc',0):.3f} itemsF1={r.get('items_f1',0):.3f} "
                      f"tiles={r.get('n_tiles','')} {r.get('error') or ''}", flush=True)
    print("\n=== " + a.tag + " ===")
    t = [r for r in res if r["plan"] in test]; h = [r for r in res if r["plan"] not in test]
    if t: print(table(t, "TILES, 15 test plans"))
    if h: print(table(h, "TILES, other plans"))
    print(table(res, "TILES, all"))
