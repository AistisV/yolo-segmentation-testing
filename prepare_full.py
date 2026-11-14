"""Full dataset -> ground truth for every Swiss plan, checked against Stoa's own answers, then a fixed split.

    python prepare_full.py --data /vol/data --gt /vol/gt_full --out /vol/split_full.json

A plan is kept only if answering Stoa's questions from its ground truth is >= 95% correct (label quality gate).
Split (seed 13):  test = the 15 preview test plans (never trained on)   holdout = 100 more unseen plans
                  eval = test + holdout     exp = 200 plans used ONLY through the environment reward
                  sft = the rest (supervised)
"""
import argparse, glob, json, os, random, collections
from multiprocessing import Pool

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--gt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--procs", type=int, default=16); ap.add_argument("--holdout", type=int, default=100); ap.add_argument("--exp", type=int, default=200)
a = ap.parse_args()
SW = os.path.join(a.data, "data", "swiss_dwellings")
os.makedirs(a.gt, exist_ok=True)


def one(dxf):
    base = dxf[:-4]; out = os.path.join(a.gt, os.path.basename(base) + ".json")
    if os.path.exists(out):
        return "skip"
    try:
        from gt_swiss import build
        json.dump(build(dxf, base + ".pdf"), open(out, "w"), ensure_ascii=False)
        return "ok"
    except Exception as e:
        return "ERR " + os.path.basename(base) + " " + repr(e)[:100]


if __name__ == "__main__":
    fs = sorted(glob.glob(os.path.join(SW, "*.dxf")))
    with Pool(a.procs) as p:
        rs = p.map(one, fs, chunksize=4)
    errs = [r for r in rs if r.startswith("ERR")]
    print(f"{len(fs)} DXF | {rs.count('ok')} built | {rs.count('skip')} already there | {len(errs)} errors")
    for e in errs[:10]: print("  ", e)

    from answer import answer, correct
    by = collections.defaultdict(list)
    for line in open(os.path.join(a.data, "questions.jsonl"), encoding="utf-8"):
        if "swiss_dwellings" in line:
            q = json.loads(line); by[os.path.basename(q["path"]).rsplit(".", 1)[0]].append(q)
    good, bad, tot, ok = [], [], collections.Counter(), collections.Counter()
    for plan, qs in sorted(by.items()):
        f = os.path.join(a.gt, plan + ".json")
        if not (os.path.exists(f) and os.path.exists(os.path.join(SW, plan + ".png"))):
            bad.append(plan); continue
        g = json.load(open(f)); n = 0
        for q in qs:
            kind, pred = answer(g, q["question"]); c = correct(pred, q["answer"], q["type"])
            tot[kind] += 1; ok[kind] += c; n += c
        (good if n / len(qs) >= 0.95 else bad).append(plan)
    print(f"ground truth answers {sum(ok.values())}/{sum(tot.values())} = {sum(ok.values())/max(1,sum(tot.values())):.1%} of Swiss questions")
    for k in sorted(tot): print(f"    {k:20} {ok[k]:6}/{tot[k]:6}  {ok[k]/tot[k]:.1%}")
    print(f"plans with questions: {len(by)} | kept (>=95% of own questions) {len(good)} | dropped {len(bad)}")

    here = os.path.dirname(os.path.abspath(__file__))
    test = [p for p in json.load(open(os.path.join(here, "split.json")))["test"] if os.path.exists(os.path.join(a.gt, p + ".json"))]
    rest = [p for p in good if p not in set(test)]
    random.Random(13).shuffle(rest)
    holdout, exp, sft = rest[:a.holdout], rest[a.holdout:a.holdout + a.exp], rest[a.holdout + a.exp:]
    json.dump(dict(test=test, holdout=holdout, eval=test + holdout, exp=exp, sft=sft), open(a.out, "w"))
    print(f"split -> {a.out}: test {len(test)} | holdout {len(holdout)} | exp {len(exp)} | sft {len(sft)}")
