"""Swiss Dwellings DXF -> ground-truth schematic JSON in PNG pixel coordinates.

Usage: python gt_swiss.py <swiss_dwellings_dir> <out_dir>
Each output <name>.json follows the same schema the models must produce (see schema.md).
"""
import json, math, os, re, sys, glob
import numpy as np, ezdxf, pymupdf
from shapely.geometry import Polygon, Point

DPI = 150
OUTDOOR = re.compile(r"balkon|balcon|balcone|loggia|terrass|terrazz|veranda|sitzplatz", re.I)

def fixture_type(o, inner, circles):
    """Classify a Swiss fixture symbol by its drawn geometry (rule validated against the answer key)."""
    if len(circles) >= 4 and all(abs(r - 0.08) < 0.02 for r in circles[:4]):
        return "stove"
    ell = [p for p in inner if len(p.exterior.coords) - 1 > 20]
    rect = [p for p in inner if len(p.exterior.coords) - 1 == 4]
    if ell and not circles:
        return "toilet"
    if ell and circles:
        n = len(ell[0].exterior.coords) - 1
        return "bathtub" if n > 55 else "washbasin"
    if rect and circles:
        return "shower"
    return None  # shafts, washing machines, cupboards etc.

def fit_transform(msp, pdf_path):
    """Affine DXF metres -> PDF points, fitted on room labels that appear in both."""
    dx = {}
    for t in msp.query('TEXT[layer=="ROOM_TEXT"]'):
        dx.setdefault(t.dxf.text.strip(), []).append((t.dxf.insert.x, t.dxf.insert.y))
    page = pymupdf.open(pdf_path)[0]
    pw = {}
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        pw.setdefault(w, []).append(((x0 + x1) / 2, (y0 + y1) / 2))
    A, B = [], []
    for s, pts in dx.items():
        if len(pts) == 1 and s in pw and len(pw[s]) == 1:
            A.append(pts[0]); B.append(pw[s][0])
    A, B = np.array(A), np.array(B)
    X = np.c_[A, np.ones(len(A))]
    M, *_ = np.linalg.lstsq(X, B, rcond=None)
    resid = float(np.abs(X @ M - B).max()) if len(A) else 1e9
    return M * (DPI / 72), len(A), resid * DPI / 72, page.rect  # now maps to PNG pixels

def build(dxf_path, pdf_path):
    doc = ezdxf.readfile(dxf_path); msp = doc.modelspace()
    M, nmatch, resid_px, rect = fit_transform(msp, pdf_path)
    to_px = lambda x, y: [round(float(v), 1) for v in (np.array([x, y, 1.0]) @ M)]
    px_per_m = float(np.linalg.norm(M[0]))

    # rooms from ROOM_OUTLINE + the label stack (name / number / area) inside each outline
    texts = [(t.dxf.text.strip(), Point(t.dxf.insert.x, t.dxf.insert.y)) for t in msp.query('TEXT[layer=="ROOM_TEXT"]')]
    rooms = []
    for e in msp.query('LWPOLYLINE[layer=="ROOM_OUTLINE"]'):
        pts = [(p[0], p[1]) for p in e.get_points()]
        if len(pts) < 3: continue
        pg = Polygon(pts)
        inside = [s for s, p in texts if pg.buffer(0.05).contains(p)]
        num = next((s for s in inside if re.fullmatch(r"[A-Z]?\d+\.\d+", s)), None)
        area_txt = next((s for s in inside if s.endswith("m²")), None)
        name = next((s for s in inside if s != num and s != area_txt and not re.match(r"(Whg|App|Ap)\.", s)), None)
        rooms.append(dict(number=num, label=name,
                          printed_area_m2=float(area_txt.split()[0]) if area_txt else None,
                          area_m2=round(pg.area, 2),
                          polygon=[to_px(x, y) for x, y in pts], _pg=pg))

    # apartments: unit outlines that carry a unit label (Whg. A / App. A / Ap. A)
    units = []
    for e in msp.query('LWPOLYLINE[layer=="UNIT_OUTLINE"]'):
        pg = Polygon([(p[0], p[1]) for p in e.get_points()])
        lab = next((s for s, p in texts if re.match(r"(Whg|App|Ap)\.\s*\w", s) and pg.buffer(0.05).contains(p)), None)
        if lab:
            units.append(dict(label=lab.split()[-1], polygon=[to_px(*c) for c in pg.exterior.coords[:-1]], _pg=pg))
    for r in rooms:
        c = r["_pg"].representative_point()
        r["apartment"] = next((u["label"] for u in units if u["_pg"].buffer(0.05).contains(c)), None)

    items = []
    for e in msp.query('LWPOLYLINE[layer=="WINDOW"]'):
        if e.closed and len(e) == 4:
            c = Polygon([(p[0], p[1]) for p in e.get_points()]).centroid
            items.append(dict(type="window", point=to_px(c.x, c.y)))
    for e in msp.query('LWPOLYLINE[layer=="DOOR"]'):
        if len(e) == 5:
            c = Polygon([(p[0], p[1]) for p in e.get_points()]).centroid if len(e) >= 3 else None
            items.append(dict(type="door", point=to_px(c.x, c.y)))
    P = [Polygon([(p[0], p[1]) for p in e.get_points()]) for e in msp.query('LWPOLYLINE[layer=="FIXTURE"]') if len(e) >= 3]
    C = [(Point(e.dxf.center.x, e.dxf.center.y), e.dxf.radius) for e in msp.query('CIRCLE[layer=="FIXTURE"]')]
    for o in P:
        if any(q is not o and q.area > o.area and q.buffer(0.005).contains(o) for q in P): continue
        inner = [p for p in P if p is not o and o.buffer(0.005).contains(p)]
        circ = sorted(r for c, r in C if o.buffer(0.005).contains(c))
        t = fixture_type(o, inner, circ)
        if t:
            items.append(dict(type=t, point=to_px(o.centroid.x, o.centroid.y)))

    walls = [[to_px(p[0], p[1]) for p in e.get_points()] for e in msp.query('LWPOLYLINE[layer=="WALL"]')]
    for r in rooms: r.pop("_pg")
    for u in units: u.pop("_pg")
    return dict(source=os.path.basename(dxf_path),
                image_size=[round(rect.width * DPI / 72), round(rect.height * DPI / 72)],
                scale=dict(px_per_m=round(px_per_m, 4), found=True, source="ground_truth"),
                rooms=rooms, apartments=units, items=items, walls=walls,
                _fit=dict(matches=nmatch, max_residual_px=round(resid_px, 2)))

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for dxf in sorted(glob.glob(os.path.join(src, "*.dxf"))):
        base = dxf[:-4]
        try:
            g = build(dxf, base + ".pdf")
            json.dump(g, open(os.path.join(out, os.path.basename(base) + ".json"), "w"), ensure_ascii=False)
            print(f"ok  {os.path.basename(base)}  rooms={len(g['rooms'])} items={len(g['items'])} fit={g['_fit']}")
        except Exception as ex:
            print(f"ERR {os.path.basename(base)}: {ex}")
