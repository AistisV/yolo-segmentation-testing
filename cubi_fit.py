"""Learn which CubiCasa SVG classes / room texts Stoa counted for each question type, from Stoa's own answers.

    python3 cubi_fit.py --zip cubicasa5k.zip --images <cubicasa images> --questions questions.jsonl --out cubi_map.json

Our first conversion took only the first word of each class ("Space Bath Sauna" -> Bath) and matched Stoa exactly
for doors, windows and bedrooms but not for kitchens, saunas, WCs, toilets or sinks. Here each room is described by
every word of its class plus the words of any text inside it, each fixture by its full class string, and for every
question type we greedily pick the set of words that makes our counts equal Stoa's on the most plans.
Fitted on the training half of the plans, reported on the other half.
"""
import argparse, collections, json, os, re, sys, zipfile
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer import parse

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True); ap.add_argument("--images", required=True)
ap.add_argument("--questions", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()

ours = {os.path.splitext(f)[0] for f in os.listdir(a.images) if f.endswith(".png")}
zf = zipfile.ZipFile(a.zip)
words = lambda s: {w for w in re.split(r"[^0-9A-Za-zÅÄÖåäö]+", s or "") if w}
rooms, fixtures = {}, {}
for k, n in enumerate(x for x in zf.namelist() if x.endswith("model.svg")):
    parts = n.strip("/").split("/"); name = f"{parts[-3]}-{parts[-2]}"
    if name not in ours: continue
    try:
        root = ET.fromstring(zf.read(n))
    except Exception:
        continue
    R, F = [], []
    for g in root.iter():
        if g.tag.split("}")[-1] != "g": continue
        cls = g.get("class") or ""
        if cls.startswith("Space "):
            txt = " ".join(t for e in g.iter() if e.tag.split("}")[-1] == "text" for t in e.itertext())
            R.append(frozenset({"C:" + w for w in words(cls[6:])} | {"T:" + w.lower() for w in words(txt)}))
        elif cls.startswith("FixedFurniture "):
            F.append(frozenset({"F:" + cls[15:].strip(), "W:" + cls[15:].split(" ")[0]}))
    rooms[name], fixtures[name] = R, F
    if k % 1000 == 0: print("svg", k, flush=True)
print(len(rooms), "plans parsed", flush=True)

truth = collections.defaultdict(dict)                     # question type -> plan -> answer
for line in open(a.questions, encoding="utf-8"):
    if "cubicasa5k" not in line: continue
    q = json.loads(line); p = os.path.basename(q["path"]).rsplit(".", 1)[0]
    if p not in rooms: continue
    kind, arg = parse(q["question"])
    if kind in ("count_roomtype", "count_item", "count_items"):
        truth[f"{kind}:{arg if isinstance(arg, str) else '+'.join(arg)}"][p] = q["answer"]

plans = sorted(rooms); fit_set = set(plans[::2])           # fit on half, report on the other half
result = {}
for qt, ans in sorted(truth.items()):
    src = rooms if qt.startswith("count_roomtype") else fixtures
    feat = collections.Counter(f for p in ans if p in fit_set for o in src[p] for f in o)
    cands = [f for f, c in feat.most_common(150) if c >= 5]
    def acc(S, subset):
        ps = [p for p in ans if (p in fit_set) == subset]
        return sum(sum(1 for o in src[p] if o & S) == ans[p] for p in ps) / max(len(ps), 1)
    S = frozenset(); best = acc(S, True)
    for _ in range(8):
        gain = max(((acc(S | {f}, True), f) for f in cands if f not in S), default=(0, None))
        if gain[0] <= best + 0.002: break
        best, S = gain[0], S | {gain[1]}
    result[qt] = sorted(S)
    print(f"{qt:32s} fit {best:6.1%}  held-out {acc(S, False):6.1%}  {sorted(S)}", flush=True)
json.dump(result, open(a.out, "w"), indent=1)
print("saved", a.out)
