"""Sanity check: answer Stoa's questions from the GROUND-TRUTH schematics. Should be ~100%."""
import json, sys, collections, os
from answer import answer, correct
qs = [json.loads(l) for l in open(sys.argv[1]) if "swiss_dwellings" in l]
gtdir = sys.argv[2]; prefer = sys.argv[3] != "geom" if len(sys.argv) > 3 else True
cache = {}; tot = collections.Counter(); ok = collections.Counter(); misses = []
for q in qs:
    name = os.path.basename(q["path"]).rsplit(".", 1)[0]
    if name not in cache:
        f = os.path.join(gtdir, name + ".json")
        cache[name] = json.load(open(f)) if os.path.exists(f) else None
    if cache[name] is None:
        tot["_no_gt"] += 1; continue
    kind, pred = answer(cache[name], q["question"], prefer)
    good = correct(pred, q["answer"], q["type"])
    tot[kind] += 1; ok[kind] += good
    if not good: misses.append((kind, name, q["question"][:90], pred, q["answer"]))
for k in tot: print(f"{k:20} {tot[k]} (skipped)") if k == "_no_gt" else print(f"{k:20} {ok[k]:4}/{tot[k]:4}  {ok[k]/tot[k]:.1%}")
n = sum(v for k, v in tot.items() if k != "_no_gt")
print("TOTAL", sum(ok.values()), "/", n, f"{sum(ok.values())/max(n,1):.1%}")
for m in misses[:25]: print("  MISS", m)
