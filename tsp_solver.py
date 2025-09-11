import itertools
import os
import re
import sys
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress

# Overrides for special house slugs on dune.gaming.tools
SLUG_OVERRIDES = {
    "spinnette": "spinette",
    "mikarrol": "mikkarol",
}


def compute_distance(p1, p2):
    """Calculate Euclidean distance between two points (x, y).

    This helper returns the straight‑line distance between two
    coordinate pairs without any consideration of map boundaries or
    exits.  It is used for within‑map calculations where no map
    transitions are involved.

    Parameters
    ----------
    p1, p2 : tuple
        Two (x, y) coordinate tuples.

    Returns
    -------
    float
        The Euclidean distance between the two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def compute_cross_distance(p1, map1, p2, map2, exits):
    """Cross-map travel cost according to your rules."""
    if map1 == map2:
        return compute_distance(p1, p2)

    def min_dist_to_bounds(point, bounds):
        x, y = point
        left, right, top, bottom = bounds
        return min(abs(x - left), abs(right - x), abs(y - top), abs(bottom - y))

    bounds1 = exits[f"{map1}_bounds"]
    bounds2 = exits[f"{map2}_bounds"]
    world_cost = exits["world_cost"]

    # --- cost to EXIT current map ---
    if map1 == "deep":
        # Always go straight SOUTH to leave Deep Desert
        y1 = p1[1]
        _, _, _, bottom = bounds1
        cost_to_exit = abs(bottom - y1)
    elif map1 in ("arrakeen", "harko") and exits.get(f"{map1}_exit_coords"):
        # Hubs: go to their fixed exit point(s)
        cost_to_exit = min(compute_distance(p1, ex) for ex in exits[f"{map1}_exit_coords"])
    else:
        # Hagga (or fallback): nearest edge
        cost_to_exit = min_dist_to_bounds(p1, bounds1)

    # --- cost to ENTER destination map ---
    if map2 == "deep":
        # Enter via one of three south entries (closest horizontally)
        x2, y2 = p2
        left, right, _, bottom = bounds2
        spawn_xs = exits.get("deep_spawn_xs")
        vertical_cost = abs(bottom - y2)
        horiz_cost = (
            min(abs(x2 - sx) for sx in spawn_xs) if spawn_xs else abs(x2 - (left + right) / 2)
        )
        cost_to_enter = vertical_cost + horiz_cost
    elif map2 == "hagga":
        # Choose the closest border to enter Hagga as close as possible to target
        cost_to_enter = min_dist_to_bounds(p2, bounds2)
    else:
        # Arrakeen/Harko: single entry; intra-hub distance is negligible -> treat as 0
        # If you want to count hub-internal distance instead, you could do:
        #   ex = exits.get(f"{map2}_exit_coords"); cost_to_enter = compute_distance(ex[0], p2)
        cost_to_enter = 0.0

    return cost_to_exit + world_cost + cost_to_enter


def load_house_coords(path):
    """
    Load a raw map file from duneawakening.th.gl and extract house representative
    coordinates.  The file format is a simple binary/text mix where house
    markers appear either as ``house_<name>_representative`` followed by two
    integers or ``house_<name>@x:y``.  Lowercase names are used as keys in
    the returned dictionary.

    Parameters
    ----------
    path : str
        Location of the .raw file on disk.

    Returns
    -------
    dict
        Mapping of lowercase house names to (x, y) tuples.  If the file
        cannot be read or no markers are found, an empty dictionary is
        returned.
    """
    try:
        raw = open(path, "rb").read()
    except (IOError, OSError):
        return {}
    # convert to a printable string, replacing non‑printables with spaces
    text = "".join(chr(b) if 32 <= b <= 126 else " " for b in raw)
    coords = {}
    # Pattern 1: house_<name>_representative + numbers
    for match in re.finditer(r"house_([a-z]+)_representative", text):
        house = match.group(1)
        snippet = text[match.end() : match.end() + 80]
        nums = re.findall(r"-?\d+", snippet)
        if len(nums) >= 2:
            x, y = int(nums[0]), int(nums[1])
            coords[house] = (x, y)
    # Pattern 2: house_<name>@x:y
    for match in re.finditer(r"house_([a-z]+)@(-?\d+):(-?\d+)", text):
        house = match.group(1)
        x, y = int(match.group(2)), int(match.group(3))
        coords[house] = (x, y)
    return coords


def parse_exit_coords(path):
    """Extract exit marker coordinates from a .raw file.

    Certain social hub maps like Arrakeen and Harko Village define
    specific world‑map exit points within their raw files.  These
    entries are typically encoded as `<letter>exit@x:y` where
    `<letter>` may indicate the orientation.  This function returns
    all such exit coordinates as a list of (x, y) tuples.  If no
    exits are found or the file cannot be read, an empty list is
    returned.

    Parameters
    ----------
    path : str
        Path to the .raw file to parse.

    Returns
    -------
    list[tuple[int, int]]
        A list of (x, y) integer coordinate pairs for exit markers.
    """
    try:
        raw = open(path, "rb").read()
    except (IOError, OSError):
        return []
    text = "".join(chr(b) if 32 <= b <= 126 else " " for b in raw)
    exits = []
    for match in re.finditer(r"[a-z]exit@(-?\d+):(-?\d+)", text):
        x = int(match.group(1))
        y = int(match.group(2))
        exits.append((x, y))
    return exits


def get_user_input_coords(default_coords, skip_base=False, console=None, fancy=False):
    """
    Prompt the player for which houses to visit and return the list of
    coordinates along with the base location.  If a house exists in
    ``default_coords`` (case‑insensitive), its coordinates are used.
    Otherwise the user is prompted to enter them manually.  The user
    may use either a dot or a comma as the decimal separator for each
    coordinate component.  To avoid ambiguity between decimal commas
    and coordinate separators, this function prompts separately for
    each coordinate value.

    Parameters
    ----------
    default_coords : dict
        A mapping of lowercase house names to coordinate tuples.

    Returns
    -------
    tuple
        A tuple (house_names, coords, base) where ``house_names`` is
        the list of names in the order provided by the user
        (preserving original case), ``coords`` is a corresponding list
        of (x, y) floats/ints, and ``base`` is the (x, y) tuple for
        the player's base.
    """
    if default_coords:
        if console is not None:
            console.print("[bold]Known houses with available coordinates:[/bold]")
            names = [name.title() for name in sorted(default_coords.keys())]
            console.print(Columns(names, equal=True, expand=True))
            console.print("[dim]Reference: https://dune.gaming.tools/landsraad/houses[/dim]")
        else:
            print("Known houses with available coordinates:")
            print(", ".join(sorted(name.title() for name in default_coords.keys())))
    houses_raw = None
    if fancy and default_coords:
        try:
            import importlib

            questionary = importlib.import_module("questionary")
            choices = [h for h in sorted(default_coords.keys())]
            choices_display = [c.title() for c in choices]
            picked = questionary.checkbox(
                "Select houses (Space to toggle, Enter to confirm)",
                choices=[{"name": "All", "value": "__all__"}]
                + [{"name": n, "value": v} for n, v in zip(choices_display, choices)],
            ).ask()
            if not picked:
                houses_raw = []
            elif "__all__" in picked:
                houses_raw = list(default_coords.keys())
            else:
                # map back to raw keys
                inv = dict(zip(choices_display, choices))
                houses_raw = [inv[p] for p in picked]
        except Exception:
            houses_raw = None
    if houses_raw is None:
        selection = input(
            "Enter the houses you wish to visit (comma‑separated), or 'all' for all known: "
        ).strip()
        if selection.lower() == "all":
            houses_raw = list(default_coords.keys())
        else:
            houses_raw = [h.strip() for h in selection.split(",") if h.strip()]
    house_names = []
    coordinates = []

    def parse_number(prompt):
        """Parse a single floating‑point number allowing comma or dot decimals."""
        while True:
            s = input(prompt).strip()
            if not s:
                print("Input cannot be empty. Try again.")
                continue
            # replace comma with dot to support German decimal separator
            s_norm = s.replace(",", ".")
            try:
                return float(s_norm)
            except ValueError:
                print("Coordinate must be numeric. Try again.")

    for h in houses_raw:
        house_names.append(h)
        key = h.lower()
        if key in default_coords:
            # use known coordinates
            coordinates.append(default_coords[key])
        else:
            # ask for manual entry of x and y separately to avoid confusion
            x_val = parse_number(f"Enter X‑coordinate for {h}: ")
            y_val = parse_number(f"Enter Y‑coordinate for {h}: ")
            coordinates.append((x_val, y_val))
    # ask for base coordinates unless skipping
    base = None
    base_map = None
    if not skip_base:
        # First, ask for the base map to show relevant links (only Hagga or Deep Desert)
        if fancy:
            try:
                import importlib

                questionary = importlib.import_module("questionary")
                base_map_choice = questionary.select(
                    "Where is your base?",
                    choices=[
                        {"name": "Hagga Basin", "value": "hagga"},
                        {"name": "Deep Desert", "value": "deep"},
                    ],
                ).ask()
                base_map = base_map_choice
            except Exception:
                base_map = None
        if base_map is None:
            # enforce a valid answer
            while True:
                base_map_input = input("Where is your base? (hagga/deep): ").strip().lower()
                if base_map_input in ["hagga", "deep"]:
                    base_map = base_map_input
                    break
                print("Please enter 'hagga' or 'deep'.")

        # Show relevant links based on base map
        if console is not None:
            console.print(
                "[dim]Tip: You can find your base coordinates on the map of your region.[/dim]"
            )
            if base_map == "hagga":
                console.print(
                    "[dim]  - Hagga: https://duneawakening.th.gl/maps/Hagga%20Basin[/dim]"
                )
            elif base_map == "deep":
                console.print(
                    "[dim]  - Deep Desert: https://duneawakening.th.gl/maps/The%20Deep%20Desert[/dim]"
                )
            else:
                # fallback: show both links if something unexpected occurs
                console.print(
                    "[dim]  - Hagga: https://duneawakening.th.gl/maps/Hagga%20Basin[/dim]"
                )
                console.print(
                    "[dim]  - Deep Desert: https://duneawakening.th.gl/maps/The%20Deep%20Desert[/dim]"
                )
        else:
            print("Tip: You can find your base coordinates on the map of your region.")
            if base_map == "hagga":
                print("  - Hagga: https://duneawakening.th.gl/maps/Hagga%20Basin")
            elif base_map == "deep":
                print("  - Deep Desert: https://duneawakening.th.gl/maps/The%20Deep%20Desert")
            else:
                # fallback: show both links if something unexpected occurs
                print("  - Hagga: https://duneawakening.th.gl/maps/Hagga%20Basin")
                print("  - Deep Desert: https://duneawakening.th.gl/maps/The%20Deep%20Desert")

        base_x = parse_number("Enter your base X‑coordinate: ")
        base_y = parse_number("Enter your base Y‑coordinate: ")
        base = (base_x, base_y)
    return house_names, coordinates, base


def compute_route_cost(order, base, coords, maps, base_map, exits):
    """Compute the total travel cost of visiting points in a specific order.

    This helper evaluates a potential route starting and ending at the
    player's base while respecting map transitions.  It sums up the
    distances between successive points using the map‑aware
    ``compute_cross_distance`` function.  The cost for the return leg
    to the base is included.

    Parameters
    ----------
    order : tuple[int]
        A permutation of indices indicating the visit order of the points.
    base : tuple
        The (x, y) coordinates of the base.
    coords : list[tuple]
        List of coordinates for each house in the same order as
        ``maps``.
    maps : list[str]
        List of map identifiers ("hagga" or "deep") corresponding to
        each coordinate.
    base_map : str
        Normalised map identifier for the base.
    exits : dict
        Mapping containing map exit y‑coordinates and the transition cost.

    Returns
    -------
    float
        Total cost of the route defined by ``order``.
    """
    total_cost = 0.0
    current_pos = base
    current_map = base_map
    for idx in order:
        next_pos = coords[idx]
        next_map = maps[idx]
        total_cost += compute_cross_distance(current_pos, current_map, next_pos, next_map, exits)
        current_pos = next_pos
        current_map = next_map
    # return to base
    total_cost += compute_cross_distance(current_pos, current_map, base, base_map, exits)
    return total_cost


def find_shortest_route(base, base_map, points, exits):
    """
    Exhaustively compute the optimal TSP route for a small list of points.

    When the number of points to visit is small (<= 10), this function
    evaluates every permutation to find the route with the lowest total
    travel cost, taking map transitions into account.

    Parameters
    ----------
    base : tuple
        (x, y) coordinates of the base.
    base_map : str
        Normalised map identifier for the base ("hagga" or "deep").
    points : list[tuple[str, tuple, str]]
        A list of (house_name, (x, y) coord, map_name) for each house.
    exits : dict
        Mapping containing map exit y‑coordinates and the transition cost.

    Returns
    -------
    tuple
        (route_names, best_distance) where ``route_names`` is the
        sequence of house names in optimal order and
        ``best_distance`` is the corresponding total cost.
    """
    # points can include a display field; unpack four values if present
    names = [p[0] for p in points]
    coords = [p[1] for p in points]
    maps = [p[2] for p in points]
    best_route = None
    best_distance = float("inf")
    # iterate over all permutations of indices
    for perm in itertools.permutations(range(len(points))):
        distance = compute_route_cost(perm, base, coords, maps, base_map, exits)
        if distance < best_distance:
            best_distance = distance
            best_route = [names[i] for i in perm]
    return best_route, best_distance


def nearest_neighbour_route_map(base, base_map, points, exits):
    """Heuristic TSP solver using nearest neighbour with map transitions.

    For a larger set of points (>10), a brute‑force search becomes
    impractical.  This function greedily selects the next unvisited
    house that minimises the map‑aware distance from the current
    position.

    Parameters
    ----------
    base : tuple
        (x, y) coordinates of the starting and ending point.
    base_map : str
        Normalised map identifier for the base.
    points : list[tuple[str, tuple, str]]
        A list of (house_name, (x, y), map_name) entries.
    exits : dict
        Mapping containing map exit y‑coordinates and the transition cost.

    Returns
    -------
    tuple
        (route_names, total_distance) where ``route_names`` is the
        visiting order and ``total_distance`` is the approximate
        route cost.
    """
    n = len(points)
    visited = [False] * n
    # points may include display; unpack accordingly
    names = [p[0] for p in points]
    coords = [p[1] for p in points]
    maps = [p[2] for p in points]
    current_pos = base
    current_map = base_map
    route = []
    total_dist = 0.0
    for _ in range(n):
        nearest_idx = None
        nearest_dist = float("inf")
        for idx in range(n):
            if visited[idx]:
                continue
            d = compute_cross_distance(current_pos, current_map, coords[idx], maps[idx], exits)
            if d < nearest_dist:
                nearest_dist = d
                nearest_idx = idx
        if nearest_idx is None:
            break
        visited[nearest_idx] = True
        total_dist += nearest_dist
        current_pos = coords[nearest_idx]
        current_map = maps[nearest_idx]
        route.append(names[nearest_idx])
    # return to base
    total_dist += compute_cross_distance(current_pos, current_map, base, base_map, exits)
    return route, total_dist


def nearest_neighbour_route(base, points):
    """
    Heuristic route planner using the nearest‑neighbour approach.  This
    algorithm greedily visits the closest unvisited house from the current
    position.  It is substantially faster than the brute‑force solver
    but does not guarantee an optimal solution.  Useful when the list
    of houses is large (>10) and an approximate route is acceptable.

    Parameters
    ----------
    base : tuple
        The (x, y) coordinates of the starting/ending location.
    points : list of tuples
        A list of (house_name, (x, y)) entries representing the houses to
        visit.

    Returns
    -------
    list[str]
        The sequence of house names in the order they should be visited.
    float
        The approximate total route length.
    """
    names = [name for name, _ in points]
    coords = [coord for _, coord in points]
    n = len(points)
    visited = [False] * n
    route = []
    current_pos = base
    total_dist = 0.0
    for _ in range(n):
        nearest_idx = None
        nearest_dist = float("inf")
        for idx, coord in enumerate(coords):
            if not visited[idx]:
                d = compute_distance(current_pos, coord)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = idx
        if nearest_idx is None:
            break
        visited[nearest_idx] = True
        total_dist += nearest_dist
        current_pos = coords[nearest_idx]
        route.append(names[nearest_idx])
    # return to base
    total_dist += compute_distance(current_pos, base)
    return route, total_dist


def nearest_neighbour_no_return(start_point, points):
    """
    Greedy nearest‑neighbour path that starts at ``start_point`` and
    visits all given ``points`` exactly once without returning to the
    start.  Returns the visit order, the distance travelled and the
    coordinates of the final point.

    Parameters
    ----------
    start_point : tuple
        The (x, y) coordinates of the starting location.
    points : list of tuples
        A list of (house_name, (x, y)) entries.

    Returns
    -------
    route : list[str]
        The sequence of house names in the order they are visited.
    total_dist : float
        The total distance travelled from start to last point.
    last_coord : tuple
        The coordinates of the last visited point.
    """
    if not points:
        return [], 0.0, start_point
    names = [name for name, _ in points]
    coords = [coord for _, coord in points]
    n = len(points)
    visited = [False] * n
    route = []
    current_pos = start_point
    total_dist = 0.0
    last_coord = start_point
    for _ in range(n):
        nearest_idx = None
        nearest_dist = float("inf")
        for idx, coord in enumerate(coords):
            if not visited[idx]:
                d = compute_distance(current_pos, coord)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = idx
        if nearest_idx is None:
            break
        visited[nearest_idx] = True
        total_dist += nearest_dist
        current_pos = coords[nearest_idx]
        last_coord = current_pos
        route.append(names[nearest_idx])
    return route, total_dist, last_coord


def prompt_user_for_points():
    """
    Compatibility wrapper for older usage.  This function now loads
    default coordinates from the provided raw map file and delegates
    prompting to the improved logic in ``main``.  It is kept for
    backwards compatibility but advises the user to use the new
    interactive workflow instead.
    """
    print(
        "This version of the route planner can automatically prefill "
        "coordinates from the raw map file.  Please run the script "
        "normally to use the improved workflow."
    )
    main()


def solve_route_with_ortools(base, base_map, points, exits, time_limit_s=5):
    """Solve the route with OR-Tools (ATSP), using compute_cross_distance as cost."""
    try:
        import importlib

        cs = importlib.import_module("ortools.constraint_solver")
        pywrapcp = cs.pywrapcp
        routing_enums_pb2 = cs.routing_enums_pb2
    except Exception as e:
        raise ImportError("OR-Tools is not installed") from e

    names = ["Base"] + [p[0] for p in points]
    coords = [base] + [p[1] for p in points]
    maps = [base_map] + [p[2] for p in points]
    n = len(names)

    # Precomputed cost matrix (int64)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            else:
                c = compute_cross_distance(coords[i], maps[i], coords[j], maps[j], exits)
                dist[i][j] = int(round(c))

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1 vehicle, depot=0 (Base)
    routing = pywrapcp.RoutingModel(manager)

    def transit_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return dist[i][j]

    transit_index = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if not solution:
        raise RuntimeError("OR-Tools could not find a solution")

    order_indices = []
    index = routing.Start(0)
    total_cost = 0
    while not routing.IsEnd(index):
        next_index = solution.Value(routing.NextVar(index))
        node = manager.IndexToNode(index)
        if node != 0:
            order_indices.append(node)
        total_cost += routing.GetArcCostForVehicle(index, next_index, 0)
        index = next_index

    route_names = [names[i] for i in order_indices]
    return route_names, float(total_cost)


def main():
    """
    Entry point for the modern Landsraad route planner.  It attempts
    to prepopulate coordinates by loading a raw map file from the
    current directory (``survival_1.4ef33d6dfe9b3fb6d1c945bc620551bc.raw``).
    The user can then choose which houses to visit and will be
    prompted for any missing coordinates along with their base location.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Landsraad Route Planner")
    parser.add_argument(
        "--force-ortools", action="store_true", help="Use OR-Tools for all point counts (also <=10)"
    )
    parser.add_argument(
        "--progress", action="store_true", help="Show a progress bar while processing the route"
    )
    parser.add_argument(
        "--ascii-map", action="store_true", help="Show a rough ASCII map of the points"
    )
    parser.add_argument(
        "--fancy-prompt",
        action="store_true",
        help="Interactive selection with checkboxes (requires questionary)",
    )
    parser.add_argument(
        "--speed-kmh", type=float, default=170.0, help="Speed for ETA estimate (km/h, default 170)"
    )
    parser.add_argument("--minimal", action="store_true", help="Reduced colors and subtle output")
    parser.add_argument(
        "--pause", action="store_true", help="Keep window open at the end (for .exe/Explorer start)"
    )
    args, _ = parser.parse_known_args()

    console = Console()
    console.rule("Landsraad Representative Route Planner")
    console.print("[dim]Plan your optimal visit order across maps.[/dim]")
    # Load frozen data only (no dependency on .raw files)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_coords = {}
    frozen = None
    frozen_path = os.path.join(script_dir, "data", "world_data.json")
    if not os.path.exists(frozen_path):
        print(
            "Error: data/world_data.json not found. Generate it with 'extract_coords.py --mode aggregated --freeze'."
        )
        return
    try:
        import json

        with open(frozen_path, "r", encoding="utf-8") as f:
            frozen = json.load(f)
    except Exception as e:
        print(f"Error reading {frozen_path}: {e}")
        return
    # Build coordinates from frozen data
    houses_obj = frozen.get("houses", {})
    for h, info in houses_obj.items():
        if isinstance(info, dict) and ("x" in info and "y" in info):
            default_coords[h] = (float(info["x"]), float(info["y"]))
    if default_coords:
        console.print(
            f"[green]Loaded {len(default_coords)} house coordinates from frozen JSON.[/green]"
        )
    else:
        console.print(
            "[yellow]No coordinates found in frozen JSON. You may enter them manually.[/yellow]"
        )

    # Ask for base coordinates early if a saved configuration exists.
    # Load previously saved base coordinates and map from a user config file.
    # Prefer user config dir (APPDATA/XDG), fallback to legacy file next to the script.
    def _user_config_path():
        try:
            if os.name == "nt":
                base = os.getenv("APPDATA") or os.path.expanduser("~")
                cfgdir = os.path.join(base, "TSP-Dune")
            else:
                base = os.getenv("XDG_CONFIG_HOME") or os.path.join(
                    os.path.expanduser("~"), ".config"
                )
                cfgdir = os.path.join(base, "tsp-dune")
            os.makedirs(cfgdir, exist_ok=True)
            return os.path.join(cfgdir, ".tsp_config")
        except Exception:
            # Fallback: in the script directory (may be temporary for .exe)
            return os.path.join(script_dir, ".tsp_config")

    user_config_path = _user_config_path()
    legacy_config_path = os.path.join(script_dir, ".tsp_config")
    saved_base = None
    read_path = (
        user_config_path
        if os.path.exists(user_config_path)
        else (legacy_config_path if os.path.exists(legacy_config_path) else None)
    )
    if read_path:
        try:
            with open(read_path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
            parts = line.split(",")
            if len(parts) == 3:
                sx, sy, smap = parts
                saved_base = (float(sx), float(sy), smap)
                console.print(
                    f"[cyan]Saved base coordinates detected: {sx}, {sy} in {smap.title()}.[/cyan]"
                )
        except Exception:
            saved_base = None
    # Decide whether to use saved base
    use_saved_base = False
    base = None
    base_map = None
    if saved_base:
        resp = input("Use saved base coordinates? (Y/n): ").strip().lower()
        if resp == "" or resp.startswith("y"):
            use_saved_base = True
            base = (saved_base[0], saved_base[1])
            base_map = "deep" if saved_base[2].lower().startswith("d") else "hagga"
            console.print(
                f"[green]Using saved base: {base[0]}, {base[1]} in {saved_base[2].title()}.[/green]"
            )
    # gather user input (house names, their coordinates, base)
    # If we use the saved base, skip prompting for base coordinates in the input function
    house_names, coords, manual_base = get_user_input_coords(
        default_coords, skip_base=use_saved_base, console=console, fancy=args.fancy_prompt
    )
    # Use the manual base if not using saved, otherwise keep saved base
    if use_saved_base:
        pass  # base is already set above
    else:
        base = manual_base

    # Build mappings from JSON (map group + display)
    house_to_map_group = {}
    house_to_map_display = {}
    pretty = {
        "hagga": "Hagga Basin",
        "harko": "Harko Village",
        "arrakeen": "Arrakeen",
        "deep": "Deep Desert",
    }
    for h, info in houses_obj.items():
        m = info.get("map") if isinstance(info, dict) else None
        if m:
            house_to_map_group[h] = m
            house_to_map_display[h] = pretty.get(m, m.title())

    # Determine base map and optionally save new base coordinates.  The base_map
    # may already be set if a saved base was used earlier; in that case we
    # skip prompting for it here.
    if base_map is None:
        base_map_input = (
            input("Is your base in Hagga Basin (H) or Deep Desert (D)? Enter H or D: ")
            .strip()
            .lower()
        )
        base_map = "deep" if base_map_input.startswith("d") else "hagga"
    # Ask if the user wants to save new base settings only if they differ from
    # a previously saved configuration.  This prevents prompting when no
    # change has occurred.
    if not use_saved_base:
        # Only prompt to save if there is no saved base or the coordinates differ
        config_path = user_config_path
        old = None
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    line = f.readline().strip()
                    parts = line.split(",")
                    if len(parts) == 3:
                        old = (float(parts[0]), float(parts[1]), parts[2])
            except Exception:
                old = None
        # Determine whether coordinates or map changed compared with saved values
        need_save = False
        if old is None:
            need_save = True
        else:
            # Compare numeric coordinates and map
            same_coords = abs(old[0] - base[0]) < 1e-6 and abs(old[1] - base[1]) < 1e-6
            same_map = old[2].lower().startswith("d") == (base_map == "deep")
            need_save = not (same_coords and same_map)
        if need_save:
            save_choice = input("Save these base coordinates for next run? (y/N): ").strip().lower()
            if save_choice.startswith("y"):
                smap = "Deep" if base_map == "deep" else "Hagga"
                try:
                    with open(config_path, "w", encoding="utf-8") as f:
                        f.write(f"{base[0]},{base[1]},{smap}\n")
                    console.print("[green]Base coordinates saved.[/green]")
                except Exception:
                    console.print("[yellow]Warning: could not save base configuration.[/yellow]")

    # Normalise house maps and build a list of (name, coord, group, display)
    points = []
    for name, coord in zip(house_names, coords):
        key = name.lower()
        m = house_to_map_group.get(key)
        d = house_to_map_display.get(key)
        # Default to base map and a generic display if unknown
        if m is None:
            m = base_map
            d = "Hagga Basin" if base_map == "hagga" else "Deep Desert"
        points.append((name, coord, m, d))

    # Define map boundaries for Hagga and Deep Desert.  Boundaries are
    # (left, right, top, bottom) in map coordinates.  For Hagga Basin
    # these values are based on the user's measurements of the extreme
    # corners of the map.  For Deep Desert they come from the parsed
    # RAW file extremes.  These boundaries are used to compute the
    # minimal distance from a point to any edge.
    HAGGA_LEFT, HAGGA_RIGHT = -456536.8404817487, 354289.859516434
    HAGGA_TOP, HAGGA_BOTTOM = -454942.10120437166, 354289.859516434
    DEEP_LEFT, DEEP_RIGHT = -755998, 693699
    DEEP_TOP, DEEP_BOTTOM = 866212, 906992
    # Approximate cost of travelling through the world map between
    # Hagga and Deep Desert.  This constant need not be exact; it just
    # needs to be large relative to within‑map distances so that the
    # solver favours staying on one map until necessary.  The value
    # below (about 8e5) was derived from the difference between the
    # southern border of Hagga and the southern entry of Deep Desert.
    WORLD_COST = 1170402.3360795523 - 354289.859516434
    # Precompute spawn positions for deep desert entry: three evenly spaced
    # locations along the southern boundary.  Players spawn in the middle
    # of each third when entering from the world map.
    third_width = (DEEP_RIGHT - DEEP_LEFT) / 3.0
    deep_spawn_xs = [DEEP_LEFT + (i + 0.5) * third_width for i in range(3)]

    # Derive bounding boxes for Arrakeen and Harko from the selected points.
    # We expand the bounding boxes by a small margin to avoid zero‑area boxes.
    arr_min_x = arr_max_x = None
    arr_min_y = arr_max_y = None
    har_min_x = har_max_x = None
    har_min_y = har_max_y = None
    for name, coord, group, _disp in points:
        x, y = coord
        if group == "arrakeen":
            if arr_min_x is None:
                arr_min_x = arr_max_x = x
                arr_min_y = arr_max_y = y
            else:
                arr_min_x = min(arr_min_x, x)
                arr_max_x = max(arr_max_x, x)
                arr_min_y = min(arr_min_y, y)
                arr_max_y = max(arr_max_y, y)
        elif group == "harko":
            if har_min_x is None:
                har_min_x = har_max_x = x
                har_min_y = har_max_y = y
            else:
                har_min_x = min(har_min_x, x)
                har_max_x = max(har_max_x, x)
                har_min_y = min(har_min_y, y)
                har_max_y = max(har_max_y, y)
    # Apply a margin if houses exist in those maps
    margin = 10000.0
    if arr_min_x is not None:
        arr_left = arr_min_x - margin
        arr_right = arr_max_x + margin
        arr_top = arr_min_y - margin
        arr_bottom = arr_max_y + margin
    else:
        arr_left = arr_right = arr_top = arr_bottom = 0
    if har_min_x is not None:
        har_left = har_min_x - margin
        har_right = har_max_x + margin
        har_top = har_min_y - margin
        har_bottom = har_max_y + margin
    else:
        har_left = har_right = har_top = har_bottom = 0

    # Load exit coordinates from JSON
    arr_exit_coords = []
    har_exit_coords = []
    exits_obj = frozen.get("exits", {}) if isinstance(frozen, dict) else {}
    if exits_obj:
        arr_exit_coords = [tuple(xy) for xy in (exits_obj.get("arrakeen") or [])]
        har_exit_coords = [tuple(xy) for xy in (exits_obj.get("harko") or [])]
    # Build the exits mapping including boundary boxes and exit markers
    exits = {
        "hagga_bounds": (HAGGA_LEFT, HAGGA_RIGHT, HAGGA_TOP, HAGGA_BOTTOM),
        "deep_bounds": (DEEP_LEFT, DEEP_RIGHT, DEEP_TOP, DEEP_BOTTOM),
        "arrakeen_bounds": (arr_left, arr_right, arr_top, arr_bottom),
        "harko_bounds": (har_left, har_right, har_top, har_bottom),
        "world_cost": WORLD_COST,
        "deep_spawn_xs": deep_spawn_xs,
        "arrakeen_exit_coords": arr_exit_coords if arr_exit_coords else None,
        "harko_exit_coords": har_exit_coords if har_exit_coords else None,
    }
    # Overlay exit markers from frozen JSON if present
    if frozen and isinstance(frozen.get("exits"), dict):
        arr = frozen["exits"].get("arrakeen")
        har = frozen["exits"].get("harko")
        if arr:
            exits["arrakeen_exit_coords"] = [tuple(xy) for xy in arr]
        if har:
            exits["harko_exit_coords"] = [tuple(xy) for xy in har]

    # Overlay coordinates from frozen data last so they win over parsed values
    if frozen and isinstance(frozen.get("houses"), dict):
        for h, info in frozen["houses"].items():
            if "x" in info and "y" in info:
                default_coords[h] = (info["x"], info["y"])
    # Choose algorithm with optional override (with status spinner)
    if len(points) == 0:
        print("No houses selected. Route is trivial: stay at base.")
        return
    with console.status("Solving route...", spinner="dots"):
        if args.force_ortools:
            try:
                route, _route_cost = solve_route_with_ortools(
                    base, base_map, points, exits, time_limit_s=5
                )
                alg = "OR-Tools (GLS, forced)"
            except Exception:
                console.print(
                    "[yellow]OR-Tools unavailable or failed. Falling back to optimal solver.[/yellow]"
                )
                route, _route_cost = find_shortest_route(base, base_map, points, exits)
                alg = "optimal (Fallback)"
        elif len(points) <= 10:
            route, _route_cost = find_shortest_route(base, base_map, points, exits)
            alg = "optimal"
        else:
            try:
                route, _route_cost = solve_route_with_ortools(
                    base, base_map, points, exits, time_limit_s=5
                )
                alg = "OR-Tools (GLS)"
            except Exception:
                route, _route_cost = nearest_neighbour_route_map(base, base_map, points, exits)
                alg = "approximate"
    # Display summary of the route (Rich)
    console.rule(f"Computed route ({alg})")
    console.print(" -> ".join(["Base"] + route + ["Base"]), style="dim")

    # Build lookup tables for map group, display name and coordinates.  These
    # are used both for computing human‑readable distances and for
    # generating detailed instructions below.  The lookups map house
    # names to their routing group, display name and coordinate.
    map_group_lookup = {p[0]: p[2] for p in points}
    # map_display_lookup removed; visit lines no longer show map name
    coord_lookup = {p[0]: p[1] for p in points}

    # Control color usage (minimal mode reduces colors)
    use_colors = not args.minimal
    # Preferred color mapping:
    # - Hagga Basin: turquoise (water/sietch) -> turquoise2
    # - Deep Desert: sand/orange -> gold3 (bright)
    # - Arrakeen (Atreides): green -> green3
    # - Harko Village (Harkonnen): blood red -> dark_red
    map_colors = {"hagga": "turquoise2", "deep": "gold3", "harko": "dark_red", "arrakeen": "green3"}

    # Compute an approximate real‑world distance (in km) by applying
    # simple scaling factors per map.  Hagga, Arrakeen and Harko use
    # 1 unit ≈ 0.01 m; Deep Desert uses 1 unit ≈ 0.10 m.  World‑map
    # transitions are not counted separately.  This provides a
    # human‑readable estimate of the route length in kilometres.  The
    # distance is computed by summing the Euclidean distance between
    # consecutive points, multiplied by the scale of the *destination*
    # map (i.e., the map of the next point).
    scale_map = {"hagga": 0.01, "arrakeen": 0.01, "harko": 0.01, "deep": 0.10}
    real_dist_m = 0.0
    prev_pos = base
    # Sum the distance for each segment using the destination map's scale.
    for name in route:
        dest_pos = coord_lookup[name]
        dest_group = map_group_lookup[name]
        # Use the scale of the destination map; default to Hagga scale.
        scale = scale_map.get(dest_group, 0.01)
        real_dist_m += compute_distance(prev_pos, dest_pos) * scale
        prev_pos = dest_pos
    # Finally add the return leg back to the base using the base map's scale.
    scale = scale_map.get(base_map, 0.01)
    real_dist_m += compute_distance(prev_pos, base) * scale
    real_dist_km = real_dist_m / 1000.0
    print(f"Approximate real distance: {real_dist_km:.2f} km")

    # Helper to describe the direction to enter Hagga from.  The
    # perimeter is divided into twelve segments: four cardinal
    # directions (north, east, south, west) and two intermediate
    # segments on each side.  For a given target coordinate this
    # function identifies the closest side and subdivides that side
    # into three equal parts to choose a granular approach direction.
    # (legacy wrapper removed)

    def entry_side_generic(coord, bounds):
        """Determine the best map entry direction for a rectangular map.

        Given a target coordinate and the bounding box of a rectangular
        map (left, right, top, bottom), this function computes which
        side of the map (north, south, east or west) is closest to the
        point and then subdivides that side into three equal segments.
        The result is one of twelve compass directions such as
        'north‑west', 'south', or 'east‑north' (which will be normalised
        to 'north‑east', etc.).

        Parameters
        ----------
        coord : tuple
            The (x, y) coordinates of the target within the map.
        bounds : tuple
            A 4‑tuple (left, right, top, bottom) representing the map
            boundaries.

        Returns
        -------
        str
            A direction string indicating the optimal entry sector.
        """
        x, y = coord
        left, right, top, bottom = bounds
        # Distances to each side
        d_w = abs(x - left)
        d_e = abs(right - x)
        d_n = abs(y - top)
        d_s = abs(bottom - y)
        # Determine the nearest side
        min_dist = min(d_w, d_e, d_n, d_s)
        if min_dist == d_n:
            side = "north"
        elif min_dist == d_s:
            side = "south"
        elif min_dist == d_w:
            side = "west"
        else:
            side = "east"
        # Subdivide the chosen side into thirds to decide on
        # intermediate directions
        if side in ("north", "south"):
            # horizontal ratio across width
            ratio = (x - left) / (right - left) if right != left else 0.5
            if ratio < 1 / 3:
                direction = f"{side}-west"
            elif ratio < 2 / 3:
                direction = side
            else:
                direction = f"{side}-east"
        else:
            # east or west: vertical ratio across height
            ratio = (y - top) / (bottom - top) if bottom != top else 0.5
            if ratio < 1 / 3:
                direction = f"{side}-north"
            elif ratio < 2 / 3:
                direction = side
            else:
                direction = f"{side}-south"
        # Normalise compound names to conventional compass terms
        mapping = {
            "west-north": "north-west",
            "west-south": "south-west",
            "east-north": "north-east",
            "east-south": "south-east",
        }
        return mapping.get(direction, direction)

    def nearest_cardinal_side(coord, bounds):
        """Return the nearest cardinal side (north/south/east/west) to a point within bounds."""
        x, y = coord
        left, right, top, bottom = bounds
        d_w = abs(x - left)
        d_e = abs(right - x)
        d_n = abs(y - top)
        d_s = abs(bottom - y)
        min_dist = min(d_w, d_e, d_n, d_s)
        if min_dist == d_n:
            return "north"
        elif min_dist == d_s:
            return "south"
        elif min_dist == d_w:
            return "west"
        else:
            return "east"

    # Optional ASCII map
    if args.ascii_map:
        try:
            console.print(Panel.fit("ASCII Map (approx)", style="blue"))
            pts = [(name, coord_lookup[name]) for name in route]
            # include base
            pts = [("Base", base)] + pts
            # compute bounds
            xs = [p[1][0] for p in pts]
            ys = [p[1][1] for p in pts]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            width, height = 60, 16

            def project(pt):
                x = int((pt[0] - minx) / (maxx - minx + 1e-6) * (width - 1))
                y = int((pt[1] - miny) / (maxy - miny + 1e-6) * (height - 1))
                return x, y

            grid = [[" "] * width for _ in range(height)]
            for name, (px, py) in pts:
                gx, gy = project((px, py))
                gy = (height - 1) - gy
                ch = name[0].upper()
                grid[gy][gx] = ch
            art = "\n".join("".join(row) for row in grid)
            console.print(Panel(art, title="Map (initial letter marks)", style="blue"))
        except Exception:
            console.print("[yellow]ASCII map rendering failed.[/yellow]")

    # Prepare detailed route instructions.  We track the current map, the
    # current position and the display name of the current map.  For each
    # map transition we either advise which side of the current map to
    # exit (for departures from Hagga) or which sector to enter (for
    # arrivals in Deep Desert).  When leaving Arrakeen or Harko, no
    # direction is needed because those maps have a single exit point.
    console.print("\n[bold]Route instructions:[/bold]")
    if use_colors:
        console.print(
            "[dim]Legend: [turquoise2]Hagga Basin[/], [gold3]Deep Desert[/], [green3]Arrakeen[/], [dark_red]Harko Village[/][/dim]"
        )
    # Clock-face mapping in English
    dir_to_clock = {
        "north": "12 o'clock",
        "north-east": "2 o'clock",
        "east": "3 o'clock",
        "south-east": "5 o'clock",
        "south": "6 o'clock",
        "south-west": "7 o'clock",
        "west": "9 o'clock",
        "north-west": "10 o'clock",
    }

    current_group = base_map
    current_pos = base  # Current position

    # Optional progress display
    prog = Progress() if args.progress else None
    if prog:
        prog.start()
        task = prog.add_task("Route", total=len(route))

    for name in route:
        dest_group = map_group_lookup[name]
        dest_pos = coord_lookup[name]

        if dest_group != current_group:
            # Combine leaving + entering into a compact instruction
            leave_msg = ""
            if current_group == "hagga":
                main_dir = nearest_cardinal_side(current_pos, exits["hagga_bounds"])
                clock = dir_to_clock.get(main_dir, "")
                leave_msg = (
                    f"Leave Hagga Basin via its {main_dir} border ({clock})"
                    if clock
                    else f"Leave Hagga Basin via its {main_dir} border"
                )
            elif current_group == "deep":
                leave_msg = "Leave Deep Desert heading south"
            elif current_group in ("arrakeen", "harko"):
                hub = "Arrakeen" if current_group == "arrakeen" else "Harko Village"
                leave_msg = f"Leave {hub}"

            enter_msg = ""
            if dest_group == "deep":
                left, right, _, _ = exits["deep_bounds"]
                ratio = (dest_pos[0] - left) / (right - left) if right != left else 0.5
                sector = (
                    "south-west" if ratio < 1 / 3 else ("south" if ratio < 2 / 3 else "south-east")
                )
                clock = dir_to_clock.get(sector, "")
                # west -> left, middle -> middle, east -> right
                if sector == "south-west":
                    which = "left"
                elif sector == "south":
                    which = "middle"
                else:
                    which = "right"
                enter_msg = (
                    f"enter Deep Desert via the {which} entrance ({clock})"
                    if clock
                    else f"enter Deep Desert via the {which} entrance"
                )
            elif dest_group == "hagga":
                side12 = entry_side_generic(dest_pos, exits["hagga_bounds"])
                clock = dir_to_clock.get(side12, "")
                enter_msg = (
                    f"re-enter Hagga Basin from the {side12} ({clock})"
                    if clock
                    else f"re-enter Hagga Basin from the {side12}"
                )
            else:
                hub = "Arrakeen" if dest_group == "arrakeen" else "Harko Village"
                enter_msg = f"travel to {hub}"

            combined = (
                f"{leave_msg} and {enter_msg}."
                if leave_msg and enter_msg
                else (f"{leave_msg}." if leave_msg else f"{enter_msg}.")
            )
            console.print(Panel.fit(combined, style="orange1"))

            current_group = dest_group

        # Visit destination
        x, y = dest_pos
        # Capitalize house name and generate link (incl. slug overrides)
        slug_name = SLUG_OVERRIDES.get(name.lower(), name.lower())
        house_url = f"https://dune.gaming.tools/landsraad/house{slug_name}"
        house_label = name.capitalize()
        if use_colors:
            color = map_colors.get(dest_group, "white")
            console.print(
                f"Visit [link={house_url}][{color}]{escape(house_label)}[/][/link] [dim]({x:.0f}, {y:.0f})[/]."
            )
        else:
            console.print(
                f"Visit [link={house_url}]{escape(house_label)}[/link] ({x:.0f}, {y:.0f})."
            )

        current_pos = dest_pos
        if prog:
            prog.update(task, advance=1)

    if prog:
        prog.stop()

    console.print(Panel.fit("Return to base to complete the route.", style="orange1"))

    # Summary
    hours = real_dist_km / max(args.speed_kmh, 1e-6)
    hh = int(hours)
    mm = int((hours - hh) * 60)
    summary = Table.grid(padding=(0, 1))
    summary.add_row("[bold]Visited[/bold]", f"{len(route)} stations")
    summary.add_row("[bold]Distance[/bold]", f"[cyan]{real_dist_km:.2f} km[/cyan]")
    summary.add_row(
        "[bold]ETA[/bold]", f"[magenta]{hh}h {mm}m[/magenta] @ {args.speed_kmh:.0f} km/h"
    )
    summary.add_row("[bold]Solver[/bold]", alg)
    console.print(Panel(summary, title="Summary", border_style="green", expand=False))

    # On packaged .exe (PyInstaller) started via Explorer (no extra args),
    # keep the window open unless user disables it. Also allow explicit --pause.
    try:
        if args.pause or (getattr(sys, "frozen", False) and len(sys.argv) <= 1):
            input("Press Enter to exit…")
    except Exception:
        pass


if __name__ == "__main__":
    main()
