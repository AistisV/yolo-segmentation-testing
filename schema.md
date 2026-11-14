You are digitising an architectural floor-plan image into a schematic. Return ONE JSON object.

Coordinates: original image PIXELS, [x, y], origin top-left. The image size is given to you.
Use the `crop` tool to zoom in; coordinates you report always refer to the ORIGINAL image.

{
  "coord_space": "pixel",
  "scale": {"found": true|false, "source": "scale_text|scale_bar|dimension_line|printed_area|estimated|none",
            "px_per_m": <pixels per metre or null>},
  "apartments": [{"label": "A"}],                    // one per apartment/unit label (App. A, Whg. B …)
  "rooms": [{
      "number": "3.09" or null,                      // room number exactly as printed
      "label": "Bain" or null,                       // room name exactly as printed, original language
      "printed_area_m2": 4.36 or null,               // area printed inside the room, if any
      "apartment": "A" or null,                      // which apartment the room belongs to
      "polygon": [[x,y], ...]                        // inner face of the walls (net area), >= 4 points
  }],
  "items": [{"type": "door|window|toilet|washbasin|bathtub|shower|stove", "point": [x,y]}]
}

Rules
- List EVERY instance separately: each door leaf, each window, each fixture.
- Door = a door leaf with its swing arc. A double door counts as two leaves.
- Washbasin and toilet are different symbols; a shower is a square tray, a bathtub a long tub.
- Copy labels and numbers exactly as printed (do not translate, do not invent numbers).
- If there is no scale on the drawing, set found=false and px_per_m=null rather than guessing.
