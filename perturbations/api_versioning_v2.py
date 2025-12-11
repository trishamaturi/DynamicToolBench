import os, json, csv
from pathlib import Path

# Config
ROOT_DIR = "../tools/toolenv2404_filtered"
NEW_DIR = "../tools/toolenv_v2"
CHANGE_LOG = "../tools/api_version_changes_v2.csv"
RENAME_FILE = "renames.json"

if os.path.exists(CHANGE_LOG):
    os.remove(CHANGE_LOG)

with open(RENAME_FILE) as f:
    RENAMES = json.load(f)

# Helpers:
def rename_param(name: str) -> str:
    """Rename parameters."""
    return RENAMES.get(name, name)

def restructure_url(url: str) -> str:
    """Increment API version."""
    url = url.replace("/v1/", "/v2/").replace("rapidapi.com", "api-service.io")
    if "/get" in url:
        url = url.replace("/get", "/fetch")
    return url

# Transformation:
def evolve_api(api: dict) -> dict:
    rec = {
        "File": "",
        "API": api.get("name", ""),
        "RenamedParams": "",
        "URLChanged": ""
    }

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

    rec["RenamedParams"] = "; ".join(renamed_list) if renamed_list else ""

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

print("\nV2 Versioning complete")
print(f"New versioned APIs saved under: {NEW_DIR}")
print(f"Change log saved to: {CHANGE_LOG}\n")