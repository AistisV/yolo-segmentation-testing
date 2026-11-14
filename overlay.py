"""Draw a Gemini-generated floor-plan schematic over the original image.

Usage:  python overlay.py plan.png schematic.json  [out.png]
Coordinates: pixel [x, y] (default, our schema) or [y, x] normalised 0-1000 if "coord_space": "norm1000".
"""
import json, re, sys
from PIL import Image, ImageDraw, ImageFont

COLORS = {"toilet": "#e6194b", "sink": "#3cb44b", "washbasin": "#3cb44b", "shower": "#4363d8", "bathtub": "#4363d8",
          "sauna_stove": "#f58231", "window": "#911eb4", "door": "#46f0f0", "stove": "#f032e6"}

def load_json(path):
    txt = open(path, encoding="utf-8").read()
    txt = re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.M)  # tolerate ```json fences
    return json.loads(txt)

def main(img_path, json_path, out_path="overlay.png"):
    img = Image.open(img_path).convert("RGBA")
    W, H = img.size
    data = load_json(json_path)
    if "schematic" in data and "rows" in data:      # a results file from run_agent / run_gemini
        data = data["schematic"] or {}
    if data.get("coord_space") == "norm1000":
        to_px = lambda p: (p[1] / 1000 * W, p[0] / 1000 * H)
    else:
        to_px = lambda p: (p[0], p[1])
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = ImageFont.load_default()

    for i, room in enumerate(data.get("rooms", [])):
        pts = [to_px(p) for p in room.get("polygon", [])]
        if len(pts) < 3:
            continue
        hue = [(255, 200, 0), (0, 180, 255), (120, 220, 120), (255, 120, 200)][i % 4]
        d.polygon(pts, fill=hue + (70,), outline=hue + (255,))
        cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
        d.text((cx, cy), f"{room.get('label') or ''} {room.get('number') or ''}", fill=(0, 0, 0, 255), font=font)

    r = max(4, W // 200)
    for it in data.get("items", []):
        x, y = to_px(it["point"])
        c = COLORS.get(it.get("type"), "#000000")
        d.ellipse((x - r, y - r, x + r, y + r), outline=c, width=3)
        d.text((x + r + 2, y - r), it.get("type", "?"), fill=c, font=font)

    Image.alpha_composite(img, layer).convert("RGB").save(out_path)
    counts = {}
    for it in data.get("items", []):
        counts[it.get("type")] = counts.get(it.get("type"), 0) + 1
    print("saved", out_path, "| rooms:", len(data.get("rooms", [])), "| items:", counts)

if __name__ == "__main__":
    main(*sys.argv[1:])
