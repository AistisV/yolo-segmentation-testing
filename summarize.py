"""Aggregate results/<model>/*.json by question kind: where does a model fail?

    python summarize.py results/gemini-3.8-flash
"""
import collections, glob, json, os, sys
from env import FloorPlanEnv

cfg = dict(images_dir=os.environ.get("IMAGES_DIR", "data/swiss_dwellings"), gt_dir=os.environ.get("GT_DIR", "gt"),
           questions_path=os.environ.get("QUESTIONS", "data/questions.jsonl"),
           snap=os.environ.get("SNAP") == "1")
tot, ok, eps = collections.Counter(), collections.Counter(), []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    plan = d.get("plan") or os.path.basename(f).rsplit("_run", 1)[0]
    info = FloorPlanEnv(plan, cfg).score(d.get("schematic") or {})   # re-score with the current scorer
    eps.append((info["reward"], info["items_f1"], d.get("usage", {}), d.get("seconds", 0)))
    for r in info["rows"]:
        tot[r["kind"]] += 1; ok[r["kind"]] += r["ok"]
print(f"{len(eps)} episodes | mean reward {sum(e[0] for e in eps)/len(eps):.3f} | "
      f"mean item F1 {sum(e[1] for e in eps)/len(eps):.3f} | "
      f"mean prompt tokens {sum(e[2].get('prompt_tokens', 0) for e in eps)/len(eps):,.0f} | "
      f"mean seconds {sum(e[3] for e in eps)/len(eps):.0f}")
for k in sorted(tot, key=lambda k: ok[k] / tot[k]):
    print(f"  {k:20} {ok[k]:4}/{tot[k]:4}  {ok[k]/tot[k]:.0%}")
