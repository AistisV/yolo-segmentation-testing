Digitise the ROOMS of this architectural floor plan into ONE compact JSON schematic (doors, windows and fixtures are handled separately). Output only the JSON, no prose.

Coordinates are RELATIVE [x, y] integers in 0..1000 (0,0 = top-left corner of the image, 1000,1000 = bottom-right).
The original image is {W}x{H} pixels: give px_per_m in those ORIGINAL pixels.
{"coord_space":"norm1000_xy",
 "scale":{"found":true|false,"source":"scale_text|scale_bar|printed_area|estimated|none","px_per_m":<number or null>},
 "apartments":[{"label":"A"}],
 "rooms":[{"number":"1.01" or null,"label":"Bad" or null,"printed_area_m2":4.36 or null,"apartment":"A" or null,
           "polygon":[[x,y],...]}]}

- One room per labelled space, including corridors, balconies, stairs; outline the inner face of its walls.
  Where two rooms share an open space with no wall, draw the boundary where the plan's room division is.
- Assign every room to its apartment (the unit label such as "Whg. A" / "App. B" and the apartment boundary line);
  use only the letter, e.g. "A".
- Copy room numbers and labels exactly as printed. Do not invent numbers.
- Scale: px_per_m from the printed scale (e.g. 1:100) or the scale bar. If there is none, found=false, px_per_m=null.
