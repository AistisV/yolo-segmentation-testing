"""Measure a drawing's scale bar in pixels (general CV, no dataset priors).

find_bars(img, box=None) -> candidate bars [(x0, x1, y, length_px)], best first.
A scale bar is a thin, long, mostly filled dark strip (often alternating black/white segments inside one outline).
If a rough box is given (e.g. from a language model, 0-1 normalised), only that area (+ margin) is searched.
px_per_m = length_px / bar_length_m, where bar_length_m is the last number printed on the bar (read by the language model).
"""
import numpy as np, cv2


def find_bars(gray, box=None):
    H, W = gray.shape
    x0 = y0 = 0; g = gray
    if box:
        bx0, by0, bx1, by1 = box; m = 0.05
        x0, y0 = int(max(0, bx0 - m) * W), int(max(0, by0 - m) * H)
        g = gray[y0:int(min(1, by1 + m) * H), x0:int(min(1, bx1 + m) * W)]
    dark = (g < 128).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    cands = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if w < 0.04 * W or h < 3 or h > 0.02 * H or w < 8 * h: continue
        sub = cc[y:y + h, x:x + w] == k
        rows = sub.mean(1)
        if rows.max() < 0.95: continue                       # outline rows must run the full length
        cols = sub.mean(0)
        if cols[: max(2, h // 2)].min() < 0.9 or cols[-max(2, h // 2):].min() < 0.9: continue   # closed ends
        cands.append((x + x0, x + w + x0, y + h // 2 + y0, int(w) - 1, a / (w * h)))   # -1: centre of the end strokes
    cands.sort(key=lambda c: (-c[3]))
    return cands


NICE = (0.5, 1, 2, 2.5, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 100)


def refine_scale(gray, rough_ppm, box=None, tol=0.12):
    """Precise px/m from the scale bar, given a rough estimate (e.g. a language model's reading).
    The bar's length in metres is the 'nice' number closest to bar_px / rough_ppm; accepted only if within tol."""
    if not rough_ppm: return None
    for x0, x1, y, L, fill in find_bars(gray, box):
        m = L / rough_ppm
        nice = min(NICE, key=lambda v: abs(m / v - 1))
        if abs(m / nice - 1) < tol:
            return L / nice
    return None
