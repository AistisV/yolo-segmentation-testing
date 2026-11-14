"""How often does the detector give Stoa's exact count on CubiCasa plans? Chooses the confidence cut-off per item type
on a validation set of plans, then reports on the test plans (no tuning on test).

    python3 cubi_count_eval.py --preds /vol/seg_preds10 --questions /vol/data/questions.jsonl --split /vol/split_r10.json
"""
import argparse, json, os, collections
from answer import parse

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True); ap.add_argument("--questions", required=True); ap.add_argument("--split", required=True)
a = ap.parse_args()
sp = json.load(open(a.split))
want = {"door": ("door",), "window": ("window",), "toilet": ("toilet",), "showers+bathtubs": ("shower", "bathtub"), "sinks": ("washbasin", "sink")}
truth = collections.defaultdict(dict)
for line in open(a.questions, encoding="utf-8"):
    if "cubicasa5k" not in line: continue
    q = json.loads(line)
    if not q["question"].startswith("How many"): continue        # English copy of each question
    p = os.path.basename(q["path"]).rsplit(".", 1)[0]; kind, arg = parse(q["question"])
    key = arg if kind == "count_item" else ("showers+bathtubs" if arg == ("shower", "bathtub") else "sinks" if kind == "count_items" else None)
    if key in want: truth[p][key] = q["answer"]


def count(pred, types, conf):
    return sum(1 for i in pred.get("items_all", pred.get("items", [])) if i["type"] in types and i.get("conf", 1) >= conf)


def acc(plans, key, conf):
    n = ok = 0
    for p in plans:
        f = os.path.join(a.preds, p + ".json")
        if key not in truth.get(p, {}) or not os.path.exists(f): continue
        n += 1; ok += count(json.load(open(f)), want[key], conf) == truth[p][key]
    return ok, n


CONFS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
print(f"{'item':18s} {'cut-off (chosen on val)':>24s} {'val exact':>10s} {'TEST exact':>11s}")
tot_ok = tot_n = 0
for key in want:
    best = max(CONFS, key=lambda c: acc(sp["cubi_val"], key, c)[0])
    vo, vn = acc(sp["cubi_val"], key, best); to, tn = acc(sp["cubi_eval"], key, best)
    tot_ok += to; tot_n += tn
    print(f"{key:18s} {best:>24} {vo:>5}/{vn:<4} {to:>6}/{tn:<4} ({to / max(tn, 1):.0%})")
print(f"ALL item counts on the test plans: {tot_ok}/{tot_n} = {tot_ok / max(tot_n, 1):.1%}   (Gemini alone: 52% windows/doors/toilets, 44% sinks/showers)")
