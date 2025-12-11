import os, json, csv, random
from pathlib import Path

# Config
ROOT_DIR = "../tools/toolenv_v2"
NEW_DIR = "../tools/toolenv_v3"
CHANGE_LOG = "../tools/api_version_changes_v3.csv"

random.seed(42)

if os.path.exists(CHANGE_LOG):
    os.remove(CHANGE_LOG)

# Helpers:
def restructure_url(url: str) -> str:
    """Increment API version"""
    url = url.replace("/v2/", "/v3/").replace("rapidapi.com", "api-service.io")
    if random.random() < 0.3:
        url = url.replace("/get", "/fetch").replace("/post", "/submit")
    return url

def limited_type_drift(p: dict) -> bool:
    """Convert int or boolean to string"""
    t = p.get("type", "")
    if t:
        if t.lower() in ("int", "integer", "boolean"):
            old_type = p["type"]
            p["type"] = "string"
            return f"{p['name']}({old_type} to string)"
    return None

def swap_required_optional(api: dict):
    """Randomly move params between required and optional"""
    req, opt = api.get("required_parameters", []), api.get("optional_parameters", [])
    swaps = []
    if req and random.random() < 0.2:
        moved = req.pop(random.randrange(len(req)))
        opt.append(moved)
        swaps.append(f"Required to Optional: {moved['name']}")
    if opt and random.random() < 0.2:
        moved = opt.pop(random.randrange(len(opt)))
        req.append(moved)
        swaps.append(f"Optional to Required: {moved['name']}")
    api["required_parameters"], api["optional_parameters"] = req, opt
    return swaps

def flip_default_behavior(p: dict) -> str or None:
    """Flip common default values like asc/desc or true/false."""
    if "default" not in p:
        return None
    old = str(p["default"]).lower()
    new = None
    if old == "asc":
        new = "desc"
    elif old == "desc":
        new = "asc"
    elif old == "true":
        new = "false" 
    elif old == "false":
        new = "true"
    if new:
        p["default"] = new
        return f"{p['name']}({old} to {new})"
    return None

def create_nested_object(api: dict, field_group: list) -> str or None:
    group_fields = [f for f in api.get("required_parameters", [])
                    if any(k in f["name"] for k in field_group)]
    if not group_fields:
        return None
    for f in group_fields:
        api["required_parameters"].remove(f)
    nested_obj = {
        "name": field_group[0],
        "type": "OBJECT",
        "fields": [f["name"] for f in group_fields],
        "description": "Nested compound object automatically created."
    }
    api["required_parameters"].append(nested_obj)
    field_names = ", ".join(f["name"] for f in group_fields)
    return f"{field_group[0]} {{{field_names}}}"

# Transformation:
def evolve_api(api: dict) -> dict:
    rec = {
        "File": "",
        "API": api.get("name", ""),
        "URLChanged": "",
        "TypeDrifted": "",
        "DefaultFlipped": "",
        "ReqOptSwaps": "",
        "NestedObjects": ""
    }

    typed, flipped, swaps, nested = [], [], [], []

    old_url = api.get("url", "")
    api["url"] = restructure_url(old_url)
    if api["url"] != old_url:
        rec["URLChanged"] = f"{old_url} to {api['url']}"

    for sec in ["required_parameters", "optional_parameters"]:
        for p in api.get(sec, []):
            type_change = limited_type_drift(p)
            if type_change:
                typed.append(type_change)

            default_change = flip_default_behavior(p)
            if default_change:
                flipped.append(default_change)

    if random.random() < 0.3:
        nested_obj = create_nested_object(api, ["address", "geo", "coord"])
        if nested_obj:
            nested.append(nested_obj)

    swap_changes = swap_required_optional(api)
    swaps.extend(swap_changes)

    rec["TypeDrifted"] = "; ".join(typed) if typed else ""
    rec["DefaultFlipped"] = "; ".join(flipped) if flipped else ""
    rec["ReqOptSwaps"] = "; ".join(swaps) if swaps else ""
    rec["NestedObjects"] = "; ".join(nested) if nested else ""

    return rec


def evolve_file(src: Path, dest: Path):
    with open(src) as f:
        data = json.load(f)

    changes = []
    for api in data.get("api_list", []):
        rec = evolve_api(api)
        rec["File"] = str(dest)
        changes.append(rec)

    with open(dest, "w") as f:
        json.dump(data, f, indent=4)

    return changes

# Main:
for dirpath, _, files in os.walk(ROOT_DIR):
    for fname in files:
        if not fname.endswith(".json"):
            continue

        rel = Path(dirpath).relative_to(ROOT_DIR)
        new_path = Path(NEW_DIR) / rel / fname
        new_path.parent.mkdir(parents=True, exist_ok=True)

        changes = evolve_file(Path(dirpath) / fname, new_path)

        with open(CHANGE_LOG, "a", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=changes[0].keys())
            if csvf.tell() == 0:
                writer.writeheader()
            writer.writerows(changes)

print("\nV3 Versioning complete")
print(f"New versioned APIs saved under: {NEW_DIR}")
print(f"Change log saved to: {CHANGE_LOG}\n")