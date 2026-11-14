"""Tiling for items: the model lists doors/windows/fixtures per overlapping tile (short lists, bigger symbols);
code merges the tiles and removes duplicates in the overlaps. Rooms are read from the whole image separately."""
import json, math, os
import numpy as np
from PIL import Image

T, MIN_OVERLAP, VIEW, MAX_UP, DEDUPE_PX = 640, 200, 1920, 3.0, 24
HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS_PROMPT = open(os.path.join(HERE, "items_prompt.md"), encoding="utf-8").read()
ROOMS_PROMPT = open(os.path.join(HERE, "rooms_prompt.md"), encoding="utf-8").read()


def _starts(L):
    if L <= T:
        return [0]
    n = math.ceil((L - T) / (T - MIN_OVERLAP)) + 1
    return [round(i * (L - T) / (n - 1)) for i in range(n)]


def tile_boxes(W, H):
    return [[x, y, min(x + T, W), min(y + T, H)] for y in _starts(H) for x in _starts(W)]


def view(img, box):
    """Crop a tile and enlarge it so its long side is VIEW pixels (at most MAX_UP times)."""
    c = img.crop(tuple(box))
    up = min(MAX_UP, VIEW / max(c.size))
    return c.resize((max(1, round(c.width * up)), max(1, round(c.height * up))), Image.LANCZOS) if up > 1.01 else c


def has_ink(img, box, min_frac=0.002):
    g = np.asarray(img.crop(tuple(box)).convert("L"))
    return float((g < 200).mean()) >= min_frac


def tile_target(items, box):
    """Ground-truth items whose centre is inside the tile -> compact text, coords 0..1000 relative to the tile."""
    x0, y0, x1, y1 = box
    inside = [it for it in items if x0 <= it["point"][0] < x1 and y0 <= it["point"][1] < y1]
    inside.sort(key=lambda it: ((it["point"][1] - y0) // 60, it["point"][0]))
    out = {}
    for it in inside:
        out.setdefault(it["type"], []).append([round((it["point"][0] - x0) * 1000 / (x1 - x0)), round((it["point"][1] - y0) * 1000 / (y1 - y0))])
    return json.dumps(out, separators=(",", ":"))


def tile_items_to_pixels(items, box):
    """Sanitised items (0..1000 in the tile) -> original-image pixels, dropping points outside the tile."""
    x0, y0, x1, y1 = box
    out = []
    for it in items:
        x, y = it["point"]
        if -5 <= x <= 1005 and -5 <= y <= 1005 and it["type"]:
            out.append({"type": it["type"], "point": [x0 + x * (x1 - x0) / 1000, y0 + y * (y1 - y0) / 1000]})
    return out


def merge_owned(per_tile, boxes, dedupe_m=0.6, ppm=None):
    """Each item counts only from the tile that OWNS its location (the tile whose centre is nearest), so an item
    seen twice in an overlap is kept once. A looser duplicate check (dedupe_m) catches what ownership misses."""
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    owned = []
    for ti, items in enumerate(per_tile):
        keep = []
        for it in items:
            x, y = it["point"]
            if min(range(len(centres)), key=lambda k: (centres[k][0] - x) ** 2 + (centres[k][1] - y) ** 2) == ti:
                keep.append(it)
        owned.append(keep)
    global DEDUPE_PX
    old = DEDUPE_PX
    if ppm: DEDUPE_PX = dedupe_m * ppm
    try:
        return merge_tiles(owned)
    finally:
        DEDUPE_PX = old


def merge_soft(per_tile, boxes, ppm=None, radius_m=1.0):
    """Ownership, but an item from a non-owning tile is kept when the owning tile reported nothing of that type
    nearby (the owner missed it). Items rescued this way are de-duplicated among themselves."""
    R = radius_m * (ppm or 59.06)
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    owner = lambda x, y: min(range(len(centres)), key=lambda k: (centres[k][0] - x) ** 2 + (centres[k][1] - y) ** 2)
    owned, rescued = [], []
    for ti, items in enumerate(per_tile):
        for it in items:
            o = owner(*it["point"])
            if o == ti:
                owned.append(it)
            elif not any(j["type"] == it["type"] and math.dist(j["point"], it["point"]) <= R for j in per_tile[o]):
                rescued.append(it)
    global DEDUPE_PX
    old = DEDUPE_PX; DEDUPE_PX = R
    try:
        rescued = merge_tiles([[r] for r in rescued])                     # one-to-one dedupe among rescued items
    finally:
        DEDUPE_PX = old
    rescued = [r for r in rescued if not any(o["type"] == r["type"] and math.dist(o["point"], r["point"]) <= R for o in owned)]
    return [{"type": i["type"], "point": i["point"]} for i in owned + rescued]


def merge_tiles(per_tile):
    """per_tile: list of item lists (pixels), one per tile. An item seen in two overlapping tiles is kept once:
    one-to-one nearest matching (same type, different tiles, within DEDUPE_PX)."""
    kept = []                                   # {"type","point","tile"}
    for ti, items in enumerate(per_tile):
        cands = []
        for a, it in enumerate(items):
            for b, k in enumerate(kept):
                if k["type"] == it["type"]:
                    d = math.dist(k["point"], it["point"])
                    if d <= DEDUPE_PX:
                        cands.append((d, a, b))
        used_a, used_b = set(), set()
        for d, a, b in sorted(cands):
            if a not in used_a and b not in used_b:
                used_a.add(a); used_b.add(b)
        kept += [{"type": it["type"], "point": it["point"], "tile": ti} for a, it in enumerate(items) if a not in used_a]
    return [{"type": k["type"], "point": k["point"]} for k in kept]
