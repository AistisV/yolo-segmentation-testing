"""Shared I/O for single-turn policies (Qwen3-VL, or any model given policy_prompt.md).

- sanitize(): tolerant, model-agnostic clean-up of a schematic (types, aliases, both item layouts)
- to_norm1000() / canonical_text(): ground truth or model output -> the compact training text
- parse_json(): JSON extraction that also rescues outputs cut off by the token limit
"""
import json, math, re

ITEM_TYPES = ("door", "window", "toilet", "washbasin", "bathtub", "shower", "stove")
ALIASES = {"wc": "toilet", "toilets": "toilet", "sink": "washbasin", "basin": "washbasin", "lavabo": "washbasin",
           "washbasins": "washbasin", "bath": "bathtub", "tub": "bathtub", "bathtubs": "bathtub", "doors": "door",
           "windows": "window", "showers": "shower", "cooker": "stove", "hob": "stove", "stoves": "stove"}


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:[.,]\d+)?", v)
        return float(m.group(0).replace(",", ".")) if m else None
    return None


def _pt(p):
    if isinstance(p, dict):
        p = [p.get("x"), p.get("y")]
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        x, y = _num(p[0]), _num(p[1])
        if x is not None and y is not None:
            return [x, y]
    return None


def _txt(v):
    if v is None or isinstance(v, (dict, list)):
        return None
    v = str(v).strip()
    return v or None


def sanitize(s):
    """Return a schematic with predictable types. Never raises. Idempotent."""
    if not isinstance(s, dict):
        return {}
    out = {"coord_space": s.get("coord_space") if isinstance(s.get("coord_space"), str) else "pixel"}
    sc = s.get("scale")
    if isinstance(sc, dict):
        out["scale"] = {"found": bool(sc.get("found", sc.get("px_per_m") is not None)),
                        "source": _txt(sc.get("source")), "px_per_m": _num(sc.get("px_per_m"))}
    else:
        out["scale"] = {"found": _num(sc) is not None, "source": None, "px_per_m": _num(sc)}
    if out["scale"]["px_per_m"] is not None and out["scale"]["px_per_m"] <= 0:
        out["scale"]["px_per_m"] = None
    apts = []
    for a in s.get("apartments") if isinstance(s.get("apartments"), list) else []:
        if isinstance(a, dict):
            a = dict(a); a["label"] = _txt(a.get("label")); apts.append(a)
        elif _txt(a):
            apts.append({"label": _txt(a)})
    out["apartments"] = apts
    if _num(s.get("total_indoor_m2")) is not None:
        out["total_indoor_m2"] = _num(s.get("total_indoor_m2"))
    rooms = []
    for r in s.get("rooms") if isinstance(s.get("rooms"), list) else []:
        if not isinstance(r, dict):
            continue
        q = dict(r)
        q["number"], q["label"], q["apartment"] = _txt(r.get("number")), _txt(r.get("label")), _txt(r.get("apartment"))
        q["printed_area_m2"] = _num(r.get("printed_area_m2"))
        if "area_m2" in r:
            q["area_m2"] = _num(r.get("area_m2"))
        poly = [p for p in (_pt(p) for p in (r.get("polygon") if isinstance(r.get("polygon"), list) else [])) if p]
        q["polygon"] = poly if len(poly) >= 3 else []
        rooms.append(q)
    out["rooms"] = rooms
    items, src = [], s.get("items")
    if isinstance(src, dict):                       # grouped layout: {"door": [[x,y],...], ...}
        for t, pts in src.items():
            for p in pts if isinstance(pts, list) else []:
                items.append({"type": t, "point": p})
        src = items; items = []
    for it in src if isinstance(src, list) else []:
        if not isinstance(it, dict):
            continue
        t = (_txt(it.get("type")) or "").lower()
        t = ALIASES.get(t, t)
        p = _pt(it.get("point", it.get("position")))
        items.append({"type": t, "point": p if p else [-1e9, -1e9]})
    out["items"] = items
    for k in ("walls", "source", "image_size"):
        if k in s:
            out[k] = s[k]
    return out


def to_norm1000(s, W, H):
    """Pixel schematic -> relative [x, y] in 0..1000 (ints). px_per_m stays in original pixels."""
    s = sanitize(s)
    for r in s["rooms"]:
        r["polygon"] = [[round(x * 1000 / W), round(y * 1000 / H)] for x, y in r["polygon"]]
    for it in s["items"]:
        it["point"] = [round(it["point"][0] * 1000 / W), round(it["point"][1] * 1000 / H)]
    s["coord_space"] = "norm1000_xy"
    return s


def canonical_text(s, with_items=True):
    """Compact JSON the policy is trained to write (coords as given; items grouped by type)."""
    s = sanitize(s)
    ppm = s["scale"]["px_per_m"]
    I = lambda v: int(round(v))
    rooms = []
    for r in s["rooms"]:
        if not r["polygon"]:
            continue
        rooms.append({"number": r["number"], "label": r["label"], "printed_area_m2": r["printed_area_m2"],
                      "apartment": r["apartment"], "polygon": [[I(x), I(y)] for x, y in r["polygon"]]})
    grouped = {}
    for it in s["items"]:
        if it["type"] and it["point"][0] > -1e8:
            grouped.setdefault(it["type"], []).append([I(it["point"][0]), I(it["point"][1])])
    out = {"coord_space": "norm1000_xy",
           "scale": {"found": ppm is not None, "source": s["scale"]["source"] or ("scale_text" if ppm else "none"),
                     "px_per_m": round(ppm, 2) if ppm else None},
           "apartments": [{"label": a["label"]} for a in s["apartments"] if a.get("label")],
           "rooms": rooms, "items": grouped}
    if not with_items:
        out.pop("items")
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def parse_json(text):
    """Extract the JSON object from a model reply. If the reply was cut off, keep everything up to the
    last complete element and close the open brackets."""
    t = re.sub(r"```(?:json)?", "", text or "")
    i = t.find("{")
    if i < 0:
        return {}
    t = t[i:]
    try:
        return json.loads(t)
    except Exception:
        pass
    stack, in_str, esc, cuts = [], False, False, []
    for k, ch in enumerate(t):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch in "{[": stack.append(ch)
        elif ch in "}]":
            if not stack: break
            stack.pop()
            cuts.append((k + 1, list(stack)))
            if not stack: break
    for pos, st in reversed(cuts[-50:]):
        try:
            v = json.loads(t[:pos] + "".join("}" if c == "{" else "]" for c in reversed(st)))
            return v if isinstance(v, dict) else {}
        except Exception:
            continue
    return {}
