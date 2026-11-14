"""Single-turn evaluation / sampling of a policy behind an OpenAI-compatible API (vLLM).
Same prompt as training (policy_prompt.md), same environment scoring for every model (snap on).

    vllm serve <model> --served-model-name policy ...
    python eval_policy.py --split test --greedy --samples 3 --tag qwen_sft

Resumable: every (plan, sample) result is a file in --out-dir; existing files are not recomputed.
--save-best FILE writes, per plan, the highest-reward sample as a training row (experience from the environment).
"""
import argparse, base64, collections, io, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from env import FloorPlanEnv
from policy_io import parse_json, sanitize, canonical_text
from make_sft import prompt_for, row

CFG = dict(images_dir=os.environ.get("IMAGES_DIR", "data/swiss_dwellings"), gt_dir=os.environ.get("GT_DIR", "gt"),
           questions_path=os.environ.get("QUESTIONS", "data/questions.jsonl"), snap=True)


def score_text(env, text):
    """Model reply -> (info, schematic in 0..1000 coords). Never raises."""
    try:
        s = sanitize(parse_json(text)); s["coord_space"] = "norm1000_xy"
        return env.score(s), s
    except Exception as e:
        return dict(reward=0.0, answer_acc=0.0, items_f1=0.0, rows=[], error=repr(e)[:300]), {}


def one(client, model, plan, k, temperature, max_tokens, guided=False):
    env = FloorPlanEnv(plan, CFG)
    W, H = env.img.size
    buf = io.BytesIO(); env.img.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    t0 = time.time()
    extra = {}
    if guided:
        from policy_schema import RESPONSE_FORMAT
        extra["response_format"] = RESPONSE_FORMAT
    r = client.chat.completions.create(model=model, temperature=temperature, max_tokens=max_tokens, **extra, messages=[
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                     {"type": "text", "text": prompt_for(W, H)}]}])
    text = r.choices[0].message.content or ""
    info, s = score_text(env, text)
    u = r.usage
    return dict(plan=plan, sample=k, temperature=temperature, reward=info["reward"], acc=info["answer_acc"],
                items_f1=info["items_f1"], rows=info["rows"], error=info.get("error"), norm_schematic=s,
                finish=r.choices[0].finish_reason, seconds=round(time.time() - t0, 1), image_size=[W, H], image_path=env.img_path,
                tokens=dict(prompt=u.prompt_tokens, completion=u.completion_tokens) if u else {})


def table(results, title):
    tot, ok = collections.Counter(), collections.Counter()
    for r in results:
        for x in r.get("rows", []):
            tot[x["kind"]] += 1; ok[x["kind"]] += bool(x["ok"])
    if not results:
        return f"{title}: no results"
    area = [k for k in tot if k.startswith("area")]; cnt = [k for k in tot if k.startswith("count")]
    pct = lambda ks: f"{sum(ok[k] for k in ks)}/{sum(tot[k] for k in ks)} = {sum(ok[k] for k in ks)/max(1,sum(tot[k] for k in ks)):.1%}"
    out = [f"{title}: {len(results)} plans | mean reward {sum(r['reward'] for r in results)/len(results):.3f} | "
           f"area questions {pct(area)} | count questions {pct(cnt)} | cut off by token limit {sum(r.get('finish') == 'length' for r in results)} | "
           f"mean prompt tokens {sum((r.get('tokens') or {}).get('prompt', 0) for r in results)/len(results):,.0f} | "
           f"mean completion tokens {sum((r.get('tokens') or {}).get('completion', 0) for r in results)/len(results):,.0f} | "
           f"mean seconds {sum(r.get('seconds', 0) for r in results)/len(results):.0f}"]
    out += [f"    {k:20} {ok[k]:5}/{tot[k]:5}  {ok[k]/tot[k]:.0%}" for k in sorted(tot)]
    return "\n".join(out)


if __name__ == "__main__":
    from openai import OpenAI
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="policy")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--plans-file", default=""); ap.add_argument("--key", default="eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--greedy", action="store_true", help="one temperature-0 run per plan (the headline number)")
    ap.add_argument("--samples", type=int, default=0, help="extra sampled runs per plan")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="policy")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--save-best", default=""); ap.add_argument("--min-reward", type=float, default=0.5)
    ap.add_argument("--guided", action="store_true", help="constrained decoding with policy_schema.py (caps runaway outlines)")
    ap.add_argument("--require-complete", action="store_true", help="--save-best: skip samples cut off by the token limit")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--summary-only", action="store_true", help="don't call the model; summarise (and --save-best) existing results")
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="x", timeout=3600, max_retries=1)
    here = os.path.dirname(os.path.abspath(__file__))
    test = json.load(open(os.path.join(here, "split.json")))["test"]
    plans = json.load(open(a.plans_file))[a.key] if a.plans_file else json.load(open(os.path.join(here, "split.json")))[a.split]
    if a.limit: plans = plans[:a.limit]
    plans = plans[a.shard::a.nshards]
    outdir = a.out_dir or os.path.join("results", "policy_" + a.tag); os.makedirs(outdir, exist_ok=True)
    jobs = ([(p, "greedy", 0.0) for p in plans] if a.greedy else []) + [(p, f"s{k}", a.temperature) for p in plans for k in range(a.samples)]
    res, todo = [], []
    for p, k, T in jobs:
        f = os.path.join(outdir, f"{p}_{k}.json")
        if os.path.exists(f):
            try: res.append(json.load(open(f))); continue
            except Exception: pass
        todo.append((p, k, T))
    print(f"{len(jobs)} jobs, {len(res)} already done, {len(todo)} to run", flush=True)
    if a.summary_only:
        todo = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, client, a.model, p, k, T, a.max_tokens, a.guided): (p, k) for p, k, T in todo}
        for i, f in enumerate(as_completed(futs)):
            p, k = futs[f]
            try:
                r = f.result()
                json.dump(r, open(os.path.join(outdir, f"{p}_{k}.json"), "w"), ensure_ascii=False)
            except Exception as e:      # request failed: not saved, so a rerun retries it
                r = dict(plan=p, sample=k, reward=0.0, acc=0.0, items_f1=0.0, rows=[], error="REQUEST " + repr(e)[:200])
            res.append(r)
            print(f"[{i+1}/{len(todo)}] {p:32s} {k:6s} reward={r['reward']:.3f} acc={r.get('acc',0):.3f} itemsF1={r.get('items_f1',0):.3f} "
                  f"{r.get('finish','')} {r.get('error') or ''}", flush=True)
    greedy = [r for r in res if r["sample"] == "greedy"]; sampled = [r for r in res if r["sample"] != "greedy"]
    print("\n=== " + a.tag + " ===")
    if greedy:
        print(table([r for r in greedy if r["plan"] in test], "GREEDY, 15 test plans"))
        hold = [r for r in greedy if r["plan"] not in test]
        if hold: print(table(hold, "GREEDY, holdout plans"))
        print(table(greedy, "GREEDY, all"))
    if sampled:
        by = collections.defaultdict(list)
        for r in sampled: by[r["plan"]].append(r)
        mean = sum(r["reward"] for r in sampled) / len(sampled)
        best = sum(max(x["reward"] for x in v) for v in by.values()) / len(by)
        print(f"SAMPLED (T={a.temperature}): {len(by)} plans x {a.samples} | mean reward {mean:.3f} | best-of-{a.samples} {best:.3f}")
        if a.save_best:
            n = 0
            with open(a.save_best, "w", encoding="utf-8") as f:
                for p, v in by.items():
                    v = [x for x in v if not (a.require_complete and x.get("finish") == "length")]
                    if not v: continue
                    b = max(v, key=lambda x: x["reward"])
                    if b["reward"] >= a.min_reward and b.get("norm_schematic"):
                        W, H = b["image_size"]
                        img = b.get("image_path") or os.path.join(str(CFG["images_dir"]).split(os.pathsep)[0], p + ".png")
                        f.write(json.dumps(dict(row(p, os.path.abspath(img), W, H,
                                                    canonical_text(b["norm_schematic"])), reward=b["reward"]), ensure_ascii=False) + "\n")
                        n += 1
            print(f"saved {n} best-of-{a.samples} samples with reward >= {a.min_reward} -> {a.save_best}")
