"""JSON schema for constrained decoding (vLLM response_format). Same key order as the training targets.
Caps stop the runaway outlines seen overnight (one room growing to 2,000+ points until the token limit)."""
PT = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer"}}
STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
NUM = {"anyOf": [{"type": "number"}, {"type": "null"}]}
SCHEMA = {
    "type": "object",
    "properties": {
        "coord_space": {"type": "string"},
        "scale": {"type": "object", "properties": {"found": {"type": "boolean"}, "source": STR, "px_per_m": NUM},
                  "required": ["found", "source", "px_per_m"], "additionalProperties": False},
        "apartments": {"type": "array", "maxItems": 30, "items": {"type": "object", "properties": {"label": {"type": "string"}},
                                                                  "required": ["label"], "additionalProperties": False}},
        "rooms": {"type": "array", "maxItems": 150, "items": {
            "type": "object",
            "properties": {"number": STR, "label": STR, "printed_area_m2": NUM, "apartment": STR,
                           "polygon": {"type": "array", "minItems": 3, "maxItems": 48, "items": PT}},
            "required": ["number", "label", "printed_area_m2", "apartment", "polygon"], "additionalProperties": False}},
        "items": {"type": "object", "maxProperties": 12, "additionalProperties": {"type": "array", "maxItems": 250, "items": PT}},
    },
    "required": ["coord_space", "scale", "apartments", "rooms", "items"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "floorplan_schematic", "schema": SCHEMA}}

ROOMS_SCHEMA = {"type": "object", "properties": {k: v for k, v in SCHEMA["properties"].items() if k != "items"},
                "required": ["coord_space", "scale", "apartments", "rooms"], "additionalProperties": False}
ITEMS_SCHEMA = {"type": "object", "maxProperties": 10, "additionalProperties": {"type": "array", "maxItems": 60, "items": PT}}
ROOMS_FORMAT = {"type": "json_schema", "json_schema": {"name": "floorplan_rooms", "schema": ROOMS_SCHEMA}}
ITEMS_FORMAT = {"type": "json_schema", "json_schema": {"name": "floorplan_tile_items", "schema": ITEMS_SCHEMA}}
