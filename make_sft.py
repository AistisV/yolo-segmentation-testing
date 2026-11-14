"""Build supervised rows: plan image -> the compact schematic text the policy should write.

    python make_sft.py --split train --out sft_train.jsonl                       # preview: split.json
    python make_sft.py --plans-file /vol/split_full.json --key sft --out x.jsonl  # full dataset

Coordinates are relative [x, y] in 0..1000 (Qwen3-VL's native convention); items are grouped by type.
Test plans (split.json "test") are always excluded.
"""
import argparse, json, os
from shapely.geometry import Polygon
from policy_io import to_norm1000, canonical_text

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(HERE, "policy_prompt.md"), encoding="utf-8").read()


def prompt_for(W, H):
    return PROMPT.replace("{W}", str(W)).replace("{H}", str(H))


def clean_poly(poly, tol=1.5):
    try:
        p = Polygon(poly).buffer(0).simplify(tol)
        if p.geom_type != "Polygon" or p.area < 1:
            return None
        return [[x, y] for x, y in list(p.exterior.coords)[:-1]]
    except Exception:
        return None


def target_text(g):
    W, H = g["image_size"]
    rooms = []
    for r in g["rooms"]:
        poly = clean_poly(r["polygon"])
        if poly:
            rooms.append({"number": r.get("number"), "label": r.get("label"), "printed_area_m2": r.get("printed_area_m2"),
                          "apartment": r.get("apartment"), "polygon": poly})
    rooms.sort(key=lambda r: (min(p[1] for p in r["polygon"]) // 40, min(p[0] for p in r["polygon"])))
    items = sorted(({"type": it["type"], "point": it["point"]} for it in g["items"]),
                   key=lambda it: (it["point"][1] // 40, it["point"][0]))
    s = {"coord_space": "pixel", "scale": {"found": True, "source": "scale_text", "px_per_m": g["scale"]["px_per_m"]},
         "apartments": [{"label": a["label"]} for a in g.get("apartments", [])], "rooms": rooms, "items": items}
    return canonical_text(to_norm1000(s, W, H))


def row(plan, image_path, W, H, text):
    return {"plan": plan, "image": image_path,
            "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_for(W, H)}]},
                         {"role": "assistant", "content": [{"type": "text", "text": text}]}]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default=os.environ.get("GT_DIR", "gt"))
    ap.add_argument("--images-dir", default=os.environ.get("IMAGES_DIR", "data/swiss_dwellings"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--plans-file", default="")
    ap.add_argument("--key", default="sft")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=22000, help="skip plans whose target is longer (very large floors)")
    ap.add_argument("--longest-first", action="store_true", help="smoke test: put the longest targets first")
    ap.add_argument("--out", default="sft_train.jsonl")
    a = ap.parse_args()
    test = set(json.load(open(os.path.join(HERE, "split.json")))["test"])
    plans = json.load(open(a.plans_file))[a.key] if a.plans_file else json.load(open(os.path.join(HERE, "split.json")))[a.split]
    if a.split != "test" or a.plans_file:
        plans = [p for p in plans if p not in test]
    rows, skipped, lens = [], 0, []
    if True:
        for p in plans:
            img = os.path.abspath(os.path.join(a.images_dir, p + ".png"))
            gt = os.path.join(a.gt_dir, p + ".json")
            if not (os.path.exists(img) and os.path.exists(gt)):
                skipped += 1; continue
            g = json.load(open(gt))
            t = target_text(g)
            if len(t) > a.max_chars:
                skipped += 1; continue
            W, H = g["image_size"]
            rows.append(row(p, img, W, H, t))
    tlen = lambda r: len(r["messages"][1]["content"][0]["text"])
    if a.longest_first:
        rows.sort(key=tlen, reverse=True)
    if a.limit:
        rows = rows[:a.limit]
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n, lens = len(rows), [tlen(r) for r in rows]
    print(f"{n} examples -> {a.out} | skipped {skipped} | target chars avg {sum(lens)/max(n,1):,.0f}, max {max(lens or [0]):,}")
