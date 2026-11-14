"""Deterministic question answering from a floor-plan schematic (JSON).

The same code answers questions for the ground-truth schematic and for any model's schematic,
so the only thing a model influences is the schematic it produces.
"""
import re

OUTDOOR = re.compile(r"balkon|balcon|balcone|loggia|terrass|terrazz|veranda|sitzplatz", re.I)
ITEM_WORDS = {
    "door": r"\bdoors?\b|türen|\bportes?\b",
    "window": r"windows?|fenster|fenêtres|finestre",
    "washbasin": r"washbasins?|lavabos?|lavabi",
    "bathtub": r"bathtubs?|badewannen|baignoires|vasche",
    "shower": r"showers?|duschen|douches|docce",
    "toilet": r"toilets?|\bwcs?\b",
}
APT = r"apartments?|wohnungen|appartamenti|appartements?|unità abitative"
QUOTED = r"[«“„'‘\"]\s*(.+?)\s*[»”“'’\"](?=\s*(?:\)|are|y a|ci sono|gibt|on|in|à|en|\?|$))"

# CubiCasa (Finnish plans): room-type and combined-fixture questions, English + Finnish
CUBI = [
    ("count_floors", None, r"floors \(storeys\)|kerrosta"),
    ("count_roomtype", "wc", r"separate wc rooms|erillistä wc-tila"),
    ("count_roomtype", "bathroom", r"bathrooms or washrooms|kylpy- tai pesuhuone"),
    ("count_roomtype", "bedroom", r"bedrooms|makuuhuone"),
    ("count_roomtype", "kitchen", r"kitchens|keittiö"),
    ("count_roomtype", "sauna", r"saunas|saunaa"),
    ("count_roomtype", "balcony", r"balconies|parveke"),
    ("count_items", ("washbasin", "sink"), r"sinks and wash basins|pesu- ja tiskiallas"),
    ("count_items", ("shower", "bathtub"), r"showers and bathtubs|suihkua ja kylpyamme"),
    ("count_item", "toilet", r"toilet bowls|wc-istuin"),
    ("count_item", "door", r"\bovea\b"),
    ("count_item", "window", r"ikkunaa"),
]

# label -> room type (Finnish abbreviations first; Swiss words for completeness)
ROOM_TYPES = [
    ("bathroom", r"^(kh|kph|ph|pesuh|pesuhuone|kylpyhuone|sh|suihku|bad|bain|bagno|dusche|douche)$"),
    ("wc", r"^(wc|toilette|toilet)$"),
    ("sauna", r"^(s|su|sauna|löylyhuone)$"),
    ("bedroom", r"^(mh|makuuhuone|makuuh)$"),
    ("kitchen", r"^(k|kt|kk|tk|ak|keittiö|keittokomero|tupakeittiö|avokeittiö|apukeittiö|keittiötila|küche|cuisine|cucina|kochen)$"),
    ("balcony", r"^(parv|parveke|lasitettu|lasit|balkon|balcon|balcone|loggia)$"),
]

def room_type(label):
    """Types of a room from its printed label ('KH/WC' -> {'bathroom'}: a WC inside a bathroom is not a separate WC room)."""
    toks = [t for t in re.split(r"[\s/+,.&()\-]+", (label or "").lower()) if t and not re.fullmatch(r"[\d.,m²]+", t)]
    found = set()
    for t in toks:
        for typ, pat in ROOM_TYPES:
            if re.match(pat, t):
                found.add(typ); break
    if "bathroom" in found:
        found.discard("wc")
    return found

def parse(q):
    ql = q.lower()
    if "m²" not in q:
        for kind, arg, pat in CUBI:
            if re.search(pat, ql):
                return (kind, arg)
    is_area = "m²" in q or "m2" in ql
    if is_area:
        if re.search(r"numbered|nummerierten|numérotées|numerati", ql):
            return ("area_total_indoor", None)
        m = re.search(r"(?i:apartment|wohnung|appartement|appartamento)\s+([A-Z])\b", q)
        if m:
            return ("area_apartment", m.group(1))
        m = re.search(r"(?:room|raum|pièce|locale)\s+(\S*?\d+\.\d+)", q, re.I)
        if m:
            return ("area_room", m.group(1))
        m = re.search(QUOTED, q)
        if m:
            return ("area_label", m.group(1))
        return ("unknown", None)
    m = re.search(r"(?:labelled|bezeichnung|désignation|denominazione)\s*" + QUOTED, q, re.I)
    if m:
        return ("count_label", m.group(1))
    if re.search(APT, ql):
        return ("count_apartments", None)
    for t, pat in ITEM_WORDS.items():
        if re.search(pat, ql):
            return ("count_item", t)
    return ("unknown", None)

def polygon_area_px(poly):
    a = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        a += x1 * y2 - x2 * y1
    return abs(a) / 2

def room_area(r, prefer_printed=True, px_per_m=None):
    """Printed area (read off the drawing) > stored area > polygon area via the scale."""
    if prefer_printed and r.get("printed_area_m2") is not None:
        return r["printed_area_m2"]
    if r.get("area_m2") is not None:
        return r["area_m2"]
    if px_per_m and len(r.get("polygon") or []) >= 3:
        return polygon_area_px(r["polygon"]) / px_per_m ** 2
    return None

def answer(schematic, question, prefer_printed=True):
    kind, arg = parse(question)
    rooms = schematic.get("rooms", [])
    items = schematic.get("items", [])
    norm = lambda s: (s or "").strip().lower()
    ppm = (schematic.get("scale") or {}).get("px_per_m")
    area = lambda r: room_area(r, prefer_printed, ppm)
    def total(rs):                                   # None if any room cannot be measured (e.g. no scale)
        v = [area(r) for r in rs]
        return None if any(x is None for x in v) else round(sum(v), 2)
    if kind == "count_item":
        return kind, sum(1 for i in items if i.get("type") == arg)
    if kind == "count_items":
        return kind, sum(1 for i in items if i.get("type") in arg)
    if kind == "count_roomtype":
        return kind, sum(1 for r in rooms if arg in room_type(r.get("label")))
    if kind == "count_floors":
        f = schematic.get("floors")
        if isinstance(f, (int, float)) and f >= 1:
            return kind, f
        return kind, (1 if rooms else None)      # no floors field: one floor, but only if something was read (do-nothing earns 0)
    if kind == "count_label":
        return kind, sum(1 for r in rooms if norm(r.get("label")) == norm(arg))
    if kind == "count_apartments":
        return kind, len(schematic.get("apartments", []))
    if kind == "area_room":
        hit = [r for r in rooms if r.get("number") == arg]
        v = area(hit[0]) if hit else None
        return kind, (round(v, 2) if v is not None else None)
    if kind == "area_label":
        return kind, total([r for r in rooms if norm(r.get("label")) == norm(arg)])
    apt = lambda v: (v or "").replace("App.", "").replace("Whg.", "").replace("Ap.", "").strip().split()[-1:] or [""]
    if kind == "area_apartment":
        return kind, total([r for r in rooms
                            if apt(r.get("apartment"))[0] == arg and r.get("number") and not OUTDOOR.search(r.get("label") or "")])
    if kind == "area_total_indoor":
        if isinstance(schematic.get("total_indoor_m2"), (int, float)):     # measured directly (e.g. from a room segmenter)
            return kind, schematic["total_indoor_m2"]
        return kind, total([r for r in rooms if r.get("number") and not OUTDOOR.search(r.get("label") or "")])
    return kind, None

def correct(pred, truth, qtype):
    """Official rule: counts exact; areas, lengths, volumes within 2%."""
    if pred is None:
        return False
    if qtype == "count":
        return round(pred) == truth
    try:
        return abs(pred - float(truth)) <= 0.02 * abs(float(truth))
    except (TypeError, ValueError):
        return False
