"""Shared image grid for training and evaluation, so coordinates mean the same thing in both.
Mirrors qwen_vl_utils.smart_resize (factor 28)."""
import json, math

MAX_PIXELS = 2480 * 1754
MIN_PIXELS = 256 * 28 * 28


def smart_resize(height, width, factor=28, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = math.floor(height / beta / factor) * factor
        w = math.floor(width / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def grid_size(W, H):
    h, w = smart_resize(H, W)
    return w, h


def scale_schematic(s, fx, fy):
    """Multiply all coordinates (and px_per_m) of a schematic by fx, fy."""
    s = json.loads(json.dumps(s))
    for r in s.get("rooms", []) or []:
        try:
            r["polygon"] = [[x * fx, y * fy] for x, y, *_ in r.get("polygon", [])]
        except Exception:
            r["polygon"] = []
    for it in s.get("items", []) or []:
        try:
            it["point"] = [it["point"][0] * fx, it["point"][1] * fy]
        except Exception:
            pass
    sc = s.get("scale") or {}
    if isinstance(sc, dict) and isinstance(sc.get("px_per_m"), (int, float)):
        sc["px_per_m"] = sc["px_per_m"] * (fx + fy) / 2
    return s


# ---------- Qwen3-VL: relative [x, y] coordinates in 0..1000 (independent of any resizing) ----------

def to_norm1000(s, W, H):
    """Pixel schematic -> 0..1000 coordinates. px_per_m stays in ORIGINAL pixels (the prompt gives W x H)."""
    s = scale_schematic(s, 1000 / W, 1000 / H)
    sc = s.get("scale") or {}
    if isinstance(sc, dict) and isinstance(sc.get("px_per_m"), (int, float)):
        sc["px_per_m"] = sc["px_per_m"] / ((1000 / W + 1000 / H) / 2)      # undo the scaling of px_per_m
    for r in s.get("rooms", []) or []:
        r["polygon"] = [[round(x), round(y)] for x, y in r.get("polygon", [])]
    for it in s.get("items", []) or []:
        it["point"] = [round(v) for v in it.get("point", [])]
    s["coord_space"] = "norm1000_xy"
    return s


def from_norm1000(s, W, H):
    """0..1000 [x, y] schematic -> original pixels; px_per_m left as written (original pixels)."""
    if not isinstance(s, dict):
        return {}
    sc = dict(s.get("scale") or {}) if isinstance(s.get("scale"), dict) else {}
    out = scale_schematic(s, W / 1000, H / 1000)
    out["scale"] = sc
    out["coord_space"] = "pixel"
    return out
