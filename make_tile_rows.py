"""Training rows for the two-part policy:
  rooms  : whole image  -> scale, apartments, rooms          (rooms_prompt.md)
  items  : one tile     -> items inside that tile            (items_prompt.md), row["box"] = tile in original pixels

    python make_tile_rows.py --plans-file /vol/split_full.json --key sft --out rows.jsonl
"""
import argparse, json, os, random
from PIL import Image
from make_sft import target_text, HERE
from policy_io import canonical_text
from tiles import ITEMS_PROMPT, ROOMS_PROMPT, tile_boxes, tile_target, has_ink


def msg(prompt, text):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": text}]}]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default=os.environ.get("GT_DIR", "gt"))
    ap.add_argument("--images-dir", default=os.environ.get("IMAGES_DIR", "data/swiss_dwellings"))
    ap.add_argument("--plans-file", default=""); ap.add_argument("--key", default="sft"); ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tiles-per-plan", type=int, default=4)
    ap.add_argument("--rooms-every", type=int, default=3, help="one rooms row for every N-th plan")
    ap.add_argument("--out", default="tile_rows.jsonl")
    a = ap.parse_args()
    test = set(json.load(open(os.path.join(HERE, "split.json")))["test"])
    plans = json.load(open(a.plans_file))[a.key] if a.plans_file else json.load(open(os.path.join(HERE, "split.json")))[a.split]
    plans = [p for p in plans if p not in test]
    if a.limit: plans = plans[:a.limit]
    rng, rows, n_rooms, n_tiles, n_empty = random.Random(7), [], 0, 0, 0
    imgdir = a.images_dir.split(os.pathsep)[0]
    for i, p in enumerate(plans):
        img_path, gt = os.path.abspath(os.path.join(imgdir, p + ".png")), os.path.join(a.gt_dir, p + ".json")
        if not (os.path.exists(img_path) and os.path.exists(gt)): continue
        g = json.load(open(gt)); W, H = g["image_size"]
        if i % a.rooms_every == 0:
            t = canonical_text(json.loads(target_text(g)), with_items=False)
            if len(t) <= 16000:
                rows.append({"plan": p, "image": img_path, "task": "rooms",
                             "messages": msg(ROOMS_PROMPT.replace("{W}", str(W)).replace("{H}", str(H)), t)}); n_rooms += 1
        img = Image.open(img_path)
        boxes = tile_boxes(W, H)
        tg = [(b, tile_target(g["items"], b)) for b in boxes]
        full = [x for x in tg if x[1] != "{}"]; empty = [x for x in tg if x[1] == "{}" and has_ink(img, x[0])]
        rng.shuffle(full); rng.shuffle(empty)
        for b, t in full[:a.tiles_per_plan] + (empty[:1] if i % 2 == 0 else []):
            rows.append({"plan": p, "image": img_path, "task": "items", "box": b, "messages": msg(ITEMS_PROMPT, t)})
            n_tiles += 1; n_empty += t == "{}"
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {a.out}: {n_rooms} rooms rows, {n_tiles} tile rows ({n_empty} empty tiles)")
