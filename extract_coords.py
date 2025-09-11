import os
import sys
import json
import argparse

# Reuse the robust parser from tsp_solver.py
try:
    import tsp_solver  # in the same folder
except Exception as e:
    print(f"Error importing tsp_solver: {e}")
    sys.exit(1)

# Explicit overrides for special cases
# - Thorvald: authoritative coordinates provided by user
THORVALD_OVERRIDE = {
    "house": "thorvald",
    "map": "hagga",
    "x": 285446,
    "y": -125153,
    "file": "override",
}


def guess_map_from_filename(name: str) -> str:
    """Heuristically map a .raw filename to a map group."""
    lname = name.lower()
    if "deepdesert" in lname:
        return "deep"
    if "arrakeen" in lname:
        return "arrakeen"
    if "harkovillage" in lname:
        return "harko"
    return "hagga"


def main():
    parser = argparse.ArgumentParser(
        description="Extract house coordinates from .raw files in this folder."
    )
    parser.add_argument(
        "--mode",
        choices=["aggregated", "per-file"],
        default="per-file",
        help="Output aggregated unique houses or list per file (default)",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Update data/world_data.json with aggregated houses and exits (default if no CSV/JSON)",
    )
    parser.add_argument("--csv", dest="csv_path", help="Write results to CSV file")
    parser.add_argument("--json", dest="json_path", help="Write results to JSON file")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raws = [f for f in os.listdir(script_dir) if f.lower().endswith(".raw")]
    raws.sort()

    # Collect per-file results
    results = []
    for fname in raws:
        fpath = os.path.join(script_dir, fname)
        try:
            coords = tsp_solver.load_house_coords(fpath)
        except Exception:
            coords = {}
        m = guess_map_from_filename(fname)
        for house, (x, y) in coords.items():
            results.append(
                {
                    "house": house,
                    "x": int(x),
                    "y": int(y),
                    "map": m,
                    "file": fname,
                }
            )

    # Also collect exits from hub files
    exits = {"arrakeen": [], "harko": []}
    for fname in raws:
        lname = fname.lower()
        fpath = os.path.join(script_dir, fname)
        if "arrakeen" in lname:
            for xy in tsp_solver.parse_exit_coords(fpath):
                exits["arrakeen"].append([int(xy[0]), int(xy[1])])
        if "harkovillage" in lname:
            for xy in tsp_solver.parse_exit_coords(fpath):
                exits["harko"].append([int(xy[0]), int(xy[1])])

    if args.mode == "per-file":
        # Sort for readability: by house, then map, then file
        rows = sorted(results, key=lambda r: (r["house"], r["map"], r["file"]))
    else:
        # aggregated: prefer hubs over deep over hagga; tie -> later file wins
        priority = {"arrakeen": 3, "harko": 3, "deep": 2, "hagga": 1}
        best = {}
        for row in results:
            key = row["house"]
            cur = best.get(key)
            if cur is None:
                best[key] = row
            else:
                p_new = priority.get(row["map"], 0)
                p_old = priority.get(cur["map"], 0)
                if (p_new > p_old) or (p_new == p_old and row["file"] >= cur["file"]):
                    best[key] = row

        # Apply explicit overrides
        # Alexin -> Harko (keep discovered coords if available; refined on freeze)
        if "alexin" in best:
            b = best["alexin"].copy()
            b["map"] = "harko"
            best["alexin"] = b

        # Thorvald -> force Hagga with fixed authoritative coordinates
        best["thorvald"] = {
            "house": THORVALD_OVERRIDE["house"],
            "map": THORVALD_OVERRIDE["map"],
            "x": int(THORVALD_OVERRIDE["x"]),
            "y": int(THORVALD_OVERRIDE["y"]),
            "file": THORVALD_OVERRIDE["file"],
        }

        rows = sorted(best.values(), key=lambda r: r["house"])

        # If requested, freeze to JSON file
        if args.freeze or (not args.csv_path and not args.json_path and args.mode == "aggregated"):
            data_dir = os.path.join(script_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            world_path = os.path.join(data_dir, "world_data.json")
            payload = {
                "houses": {},
                "exits": exits,
            }
            for r in rows:
                h = r["house"]
                entry = payload["houses"].setdefault(h, {})
                entry["map"] = r["map"]
                entry["x"] = int(r["x"])
                entry["y"] = int(r["y"])

            # Ensure explicit overrides are present
            # Alexin -> Harko, take coordinates from Harko hit if available
            payload["houses"].setdefault("alexin", {}).update({"map": "harko"})
            alexin_harko = next(
                (r for r in results if r["house"] == "alexin" and r["map"] == "harko"), None
            )
            if alexin_harko:
                payload["houses"]["alexin"].update(
                    {"x": int(alexin_harko["x"]), "y": int(alexin_harko["y"])}
                )

            # Thorvald -> Hagga with fixed authoritative coordinates
            payload["houses"]["thorvald"] = {
                "map": THORVALD_OVERRIDE["map"],
                "x": int(THORVALD_OVERRIDE["x"]),
                "y": int(THORVALD_OVERRIDE["y"]),
            }

            with open(world_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Updated frozen data: {world_path}")

    # Emit
    if args.csv_path:
        import csv

        with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["house", "map", "x", "y", "file"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote CSV: {args.csv_path}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON: {args.json_path}")

    # Only print table when not writing files and not in aggregated mode
    if not args.csv_path and not args.json_path and args.mode != "aggregated":
        if not rows:
            print("No house coordinates found in .raw files.")
            return
        width_house = max(5, max(len(r["house"]) for r in rows))
        width_map = max(4, max(len(r["map"]) for r in rows))
        width_file = max(4, max(len(r["file"]) for r in rows))
        header = f"{'house'.ljust(width_house)}  {'map'.ljust(width_map)}  {'x':>10}  {'y':>10}  {'file'.ljust(width_file)}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['house'].ljust(width_house)}  {r['map'].ljust(width_map)}  {r['x']:>10}  {r['y']:>10}  {r['file'].ljust(width_file)}"
            )


if __name__ == "__main__":
    main()
