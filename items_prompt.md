This is one enlarged tile of an architectural floor plan. List every door, window and fixture whose centre is inside this tile.
Output only JSON, no prose. Coordinates are RELATIVE [x, y] integers in 0..1000 within this tile (0,0 = top-left, 1000,1000 = bottom-right).
{"door":[[x,y],...],"window":[...],"toilet":[...],"washbasin":[...],"sink":[...],"bathtub":[...],"shower":[...],"stove":[...]}
- One point per door leaf, window and fixture, at its centre. "sink" is a kitchen sink; "washbasin" is a bathroom basin.
- Leave out types that do not appear. If the tile shows none, output {}.
