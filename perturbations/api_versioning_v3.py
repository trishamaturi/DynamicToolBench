import os, json, csv, random
from pathlib import Path

# Config
SCRIPT_DIR = Path(__file__).parent.resolve()

ROOT_DIR = SCRIPT_DIR / "../tools/toolenv2404_filtered"
NEW_DIR = SCRIPT_DIR / "../tools/toolenv_v3_new"
CHANGE_LOG = SCRIPT_DIR / "../tools/api_version_changes_v3_new.csv"
RENAME_FILE = SCRIPT_DIR / "renames.json"

random.seed(42)

if os.path.exists(CHANGE_LOG):
    os.remove(CHANGE_LOG)

with open(RENAME_FILE) as f:
    RENAMES = json.load(f)

# Helpers:
def rename_param(name: str) -> str:
    """Rename parameters."""
    return RENAMES.get(name, name)

def restructure_url(url: str) -> str:
    """Increment API version"""
    url = url.replace("/v2/", "/v3/").replace("rapidapi.com", "api-service.io")
    if random.random() < 0.99:
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
    if req and random.random() < 0.99:
        moved = req.pop(random.randrange(len(req)))
        opt.append(moved)
        swaps.append(f"Required to Optional: {moved['name']}")
    if opt and random.random() < 0.99:
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
        "RenamedParams": "",
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

    renamed_list = []
    for sec in ["required_parameters", "optional_parameters"]:
        for p in api.get(sec, []):
            old_name = p["name"]
            new_name = rename_param(old_name)
            if old_name != new_name:
                renamed_list.append(f"{old_name} to {new_name}")
                p["name"] = new_name
            
            type_change = limited_type_drift(p)
            if type_change:
                typed.append(type_change)

            default_change = flip_default_behavior(p)
            if default_change:
                flipped.append(default_change)
    
    if random.random() < 0.99:
        nested_obj = create_nested_object(api, ["address", "geo", "coord"])
        if nested_obj:
            nested.append(nested_obj)

    swap_changes = swap_required_optional(api)
    swaps.extend(swap_changes)

    rec["RenamedParams"] = "; ".join(renamed_list) if renamed_list else "x"
    rec["TypeDrifted"] = "; ".join(typed) if typed else "x"
    rec["DefaultFlipped"] = "; ".join(flipped) if flipped else "x"
    rec["ReqOptSwaps"] = "; ".join(swaps) if swaps else "x"
    rec["NestedObjects"] = "; ".join(nested) if nested else "x"

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