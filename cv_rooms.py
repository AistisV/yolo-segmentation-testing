"""Classic-CV room segmentation tool, shared by every model.

segment_rooms(img, seeds) -> one polygon + pixel area per seed.

Method (no learning):
 1. ink mask: every non-white pixel except the red apartment-boundary lines;
 2. wall mask: morphological opening of the ink removes thin strokes (text, door arcs,
    furniture, window lines) and keeps thick wall bands; kernel size adapts to wall thickness;
 3. watershed on the distance transform of free space, with one marker per seed plus an
    "outside" marker at the image border. Boundaries settle at the narrowest passages,
    i.e. door openings and windows, so rooms are split at doors without any door detector;
 4. each region's holes (text, fixtures) are filled; polygon = simplified outer contour.

The model supplies the seeds (a point inside each room, e.g. at its label). The tool does
not know the scale: areas come back in pixels, the model still has to get px_per_m right.
"""
import os
import cv2
import numpy as np
from skimage.segmentation import watershed


def _masks(rgb, T=None):
    T = T or int(os.environ.get('CV_T', 3))
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    gray = (r + g + b) / 3
    red = (r - g > 60) & (r - b > 60)
    ink = (gray < int(os.environ.get('CV_INK', 235))) & ~red
    ink = ink.astype(np.uint8)
    # walls (and window lines) are long straight horizontal/vertical runs; text, door arcs and
    # most furniture strokes are short or curved. Keep runs of at least L pixels.
    L = max(11, int(round(min(rgb.shape[:2]) / 110)))
    C = int(os.environ.get('CV_CLOSE', 0))
    if C:  # fill outlined (white-filled) wall bands: two thin parallel lines become one solid band
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((C, C), np.uint8))
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((T, T), np.uint8))  # drop 1-px fixture/stair lines
    h = cv2.morphologyEx(thick, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    v = cv2.morphologyEx(thick, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    walls = cv2.morphologyEx(h | v, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    best = L
    return ink, walls, best


GROW = int(os.environ.get('CV_GROW', 1))  # anti-aliased wall edges make the mask ~1 px too thick per side


def segment_rooms(img, seeds, simplify_px=2.0, return_labels=False):
    """img: PIL image or HxWx3 array. seeds: list of [x, y] in original pixels.
    Returns list (same order as seeds) of {"polygon": [[x,y],...], "area_px": float, "ok": bool}."""
    rgb = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img)
    H, W = rgb.shape[:2]
    ink, walls, k = _masks(rgb)
    gray_u8 = rgb.astype(np.uint16).sum(2) // 3
    free = (walls == 0).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    markers = np.zeros((H, W), np.int32)
    # outside marker: border frame
    markers[:3, :] = markers[-3:, :] = 1
    markers[:, :3] = markers[:, -3:] = 1
    out = []
    for i, (x, y) in enumerate(seeds):
        x, y = int(round(x)), int(round(y))
        x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
        # move seed off walls / text to nearest free pixel with decent clearance
        if free[y, x] == 0 or dist[y, x] < 2:
            ys, xs = np.nonzero(dist[max(0, y - 40):y + 40, max(0, x - 40):x + 40] >= 2)
            if len(xs):
                j = np.argmin((xs + max(0, x - 40) - x) ** 2 + (ys + max(0, y - 40) - y) ** 2)
                x, y = xs[j] + max(0, x - 40), ys[j] + max(0, y - 40)
        cv2.circle(markers, (x, y), 2, i + 2, -1)
    # elevation = -distance to nearest wall: each room is one basin, and two basins meet at the
    # narrowest passage between them (a door opening). Unseeded pockets join a neighbour.
    lab = watershed(-dist, markers, mask=free.astype(bool))
    lab[walls > 0] = 0
    for i in range(len(seeds)):
        m = ((lab == i + 2) & (walls == 0)).astype(np.uint8)
        if m.sum() == 0:
            out.append({"polygon": [], "area_px": 0.0, "ok": False}); continue
        # keep the component containing most pixels, fill holes (text, fixtures)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
        m = (cc == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        filled = np.zeros_like(m); cv2.drawContours(filled, [c], -1, 1, -1)
        if GROW:  # take back anti-aliased wall-edge pixels that are lighter than mid-grey
            ring = cv2.dilate(filled, np.ones((2 * GROW + 1,) * 2, np.uint8)) & (1 - filled)
            filled = filled | (ring & (gray_u8 >= 128))
        area_px = float(filled.sum())
        touches_border = st[1:, cv2.CC_STAT_LEFT].min() <= 3
        c = cv2.approxPolyDP(c, simplify_px, True)
        poly = c.reshape(-1, 2).astype(float).tolist()
        out.append({"polygon": poly, "area_px": area_px, "ok": not touches_border})
    if return_labels:
        return out, lab, walls
    return out


# ---------- snap a rough room outline onto the real wall faces ----------

_MASK_CACHE, _MASK_LOCK = {}, __import__("threading").Lock()


def compute_masks(img, T=2):
    """(wall mask, grey image) for snap_polygon. Pure function: safe to call from several threads."""
    rgb = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img)
    return (_masks(rgb, T=T)[1], rgb.astype(np.float32).mean(2))


def wall_mask(img, T=2):
    """Small cache for repeated calls on the same image object (single-image scripts)."""
    key = (id(img), T)
    with _MASK_LOCK:
        val = _MASK_CACHE.get(key)
    if val is None:
        val = compute_masks(img, T)
        with _MASK_LOCK:
            if len(_MASK_CACHE) >= 2:
                _MASK_CACHE.clear()
            _MASK_CACHE[key] = val
    return val


def _subpixel_face(profile, c0, out):
    """profile[k] = mean grey at offset k*out from the first free row/col c0. Return the position
    where grey crosses mid-level (the drawn wall edge), in image coordinates."""
    for k in range(1, len(profile)):
        if profile[k] < 128 <= profile[k - 1]:
            t = (profile[k - 1] - 128) / max(profile[k - 1] - profile[k], 1e-6)
            return float(c0 + out * (k - 1 + t + 0.5))
    return float(c0 + 0.5 * out)


def _poly_area(p):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(p, p[1:] + p[:1]))) / 2


def _room_side_free(walls, c, out, idx, run, horiz):
    """True if the room side of a candidate face stays mostly free for `run` pixels, i.e. this is a
    room face and not the inside of a hollow (double-line) wall band."""
    H, W = walls.shape
    for j in (run // 2, run):
        cc = c - out * j
        if horiz:
            if not (0 <= cc < H) or np.mean(walls[cc, idx] == 0) < 0.5: return False
        else:
            if not (0 <= cc < W) or np.mean(walls[idx, cc] == 0) < 0.5: return False
    return True


def snap_polygon(img, polygon, radius_px=30, min_support=0.35, T=2, masks=None):
    """Move each axis-aligned edge of a rough polygon to the nearest wall face within radius_px.
    An edge with no wall behind it (open-plan boundary) stays where the model put it.
    Returns {"polygon", "area_px", "snapped": [bool per edge]}."""
    walls, grey = masks if masks is not None else wall_mask(img, T)
    H, W = walls.shape
    P = [list(map(float, p)) for p in polygon]
    n = len(P)
    if n < 3:
        return {"error": "need >= 3 points"}
    # orientation: positive area => clockwise in image coords (y down)
    s = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(P, P[1:] + P[:1]))
    cw = s > 0
    run = max(4, radius_px // 2)   # ~0.22 m when radius is 0.45 m
    lines, snapped = [], []
    for i in range(n):
        (x1, y1), (x2, y2) = P[i], P[(i + 1) % n]
        horiz, vert = abs(y2 - y1) < 0.2 * abs(x2 - x1), abs(x2 - x1) < 0.2 * abs(y2 - y1)
        if not (horiz or vert):
            lines.append(None); snapped.append(False); continue
        # outward normal: for clockwise (y-down) polygons, outward of edge (dx,dy) is (dy,-dx) normalised
        dx, dy = x2 - x1, y2 - y1
        ox, oy = (dy, -dx) if cw else (-dy, dx)
        if horiz:
            c0 = (y1 + y2) / 2; out = 1 if oy > 0 else -1
            a, b = sorted((x1, x2)); m = 0.15 * (b - a)
            idx = np.arange(int(a + m), int(b - m) + 1).clip(0, W - 1)
            best, bc = None, 0
            for d in range(-radius_px, radius_px + 1):
                c = int(round(c0)) + d
                if not (1 <= c < H - 1): continue
                sup = np.mean((walls[c, idx] == 0) & (walls[c + out, idx] > 0)) if len(idx) else 0
                if sup >= min_support and _room_side_free(walls, c, out, idx, run, True) and (best is None or abs(d) < abs(best - round(c0))):
                    best, bc = c, sup
            if best is None:
                lines.append(("h", c0)); snapped.append(False)
            else:
                prof = [grey[min(max(best + k * out, 0), H - 1), idx].mean() for k in range(6)]
                lines.append(("h", _subpixel_face(prof, best, out))); snapped.append(True)
        else:
            c0 = (x1 + x2) / 2; out = 1 if ox > 0 else -1
            a, b = sorted((y1, y2)); m = 0.15 * (b - a)
            idx = np.arange(int(a + m), int(b - m) + 1).clip(0, H - 1)
            best = None
            for d in range(-radius_px, radius_px + 1):
                c = int(round(c0)) + d
                if not (1 <= c < W - 1): continue
                sup = np.mean((walls[idx, c] == 0) & (walls[idx, c + out] > 0)) if len(idx) else 0
                if sup >= min_support and _room_side_free(walls, c, out, idx, run, False) and (best is None or abs(d) < abs(best - round(c0))):
                    best = c
            if best is None:
                lines.append(("v", c0)); snapped.append(False)
            else:
                prof = [grey[idx, min(max(best + k * out, 0), W - 1)].mean() for k in range(6)]
                lines.append(("v", _subpixel_face(prof, best, out))); snapped.append(True)
    # rebuild vertices: vertex i joins edge i-1 and edge i
    Q = []
    for i in range(n):
        prev, cur = lines[i - 1], lines[i]
        x, y = P[i]
        for L in (prev, cur):
            if L is None: continue
            if L[0] == "h": y = L[1]
            else: x = L[1]
        Q.append([round(x, 1), round(y, 1)])
    return {"polygon": Q, "area_px": round(_poly_area(Q), 1), "snapped": snapped}


# ---------- refine: model outline + walls -> room region ----------

def refine_polygon(img, polygon, ppm, masks_by_t=None, r_m=0.35, T=3, trim_m=None):
    """Room = free space (no wall) reachable from inside the model's outline, but never further than r_m beyond it.
    Walls bound the room exactly where they exist; where there is no wall (open plan) the model's outline,
    widened by r_m (or trim_m if given), bounds it. Returns (polygon, area_px)."""
    if masks_by_t is not None and T in masks_by_t:
        walls, grey = masks_by_t[T]
    else:
        walls, grey = compute_masks(img, T)
    H, W = walls.shape
    P = np.array(polygon, dtype=np.float64)
    if len(P) < 3:
        return polygon, 0.0
    r = max(2, int(round(r_m * ppm)))
    x0, y0 = np.floor(P.min(0)).astype(int) - r - 2
    x1, y1 = np.ceil(P.max(0)).astype(int) + r + 3
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return polygon, 0.0
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round(P - [x0, y0]).astype(np.int32)], 1)
    if m.sum() == 0:
        return polygon, 0.0
    k = 2 * r + 1
    near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    free = (walls[y0:y1, x0:x1] == 0).astype(np.uint8)
    cand = free & near
    n, cc, st, _ = cv2.connectedComponentsWithStats(cand, 4)
    if n <= 1:
        return polygon, 0.0
    # the component that overlaps the model outline most
    ov = np.bincount(cc[m > 0].ravel(), minlength=n); ov[0] = 0
    reg = (cc == int(np.argmax(ov))).astype(np.uint8)
    cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    filled = np.zeros_like(reg); cv2.drawContours(filled, [c], -1, 1, -1)       # fill text / fixture holes
    if trim_m is not None:                                                     # optional tighter bound on open sides
        kt = 2 * max(1, int(round(trim_m * ppm))) + 1
        filled &= cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kt, kt)))
    g = grey[y0:y1, x0:x1]                                                     # take back anti-aliased wall-edge pixels
    ring = cv2.dilate(filled, np.ones((3, 3), np.uint8)) & (1 - filled) & (1 - free)
    filled = filled | (ring & (g >= 128))
    cs, _ = cv2.findContours(filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) + [x0, y0]
    return c.astype(float).tolist(), float(filled.sum())
