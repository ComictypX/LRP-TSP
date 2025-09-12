# Landsraad Route Planner (LRP-TSP)

A fast route planner (TSP - Traveling Salesman Problem) for **Dune: Awakening**. It computes efficient visit orders for Landsraad houses across maps (Hagga, Deep Desert, Arrakeen, Harko) and prints concise, color-coded instructions. Runtime uses a frozen data file (`data/world_data.json`) with known houses and exits.

## Features

- Fast routing: greedy heuristic, optional OR-Tools solver for higher quality.
- Semantic coloring: map-based colors (Hagga=turquoise, Deep Desert=sand-gold, Arrakeen=green, Harko=red).
- Compact instructions: visit lines with clickable links; transitions in one line.
- Interactive prompts: select base and targets (Rich UI, optional).
- Optional extras: ASCII map, progress bar, ETA estimate (default 170 km/h).
- Windows EXE build (PyInstaller + UPX compression).
- Configurable flags (minimal mode, force solver, etc.).

## Installation

### Requirements
- Python 3.10+ (Windows EXE requires no Python install).
- Optional: OR-Tools for improved routing (`pip install ortools`).

### Python install
```powershell
# Optional virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Or via batch script:
```powershell
./setup.bat
```

### Standalone downloads (Multi-OS)
- Grab the latest prerelease from [GitHub Releases](https://github.com/ComictypX/LRP-TSP/releases):
- Windows: `LRP-TSP.exe`
- Linux: `LRP-TSP-linux.tar.gz`
- macOS: `LRP-TSP-macos.tar.gz`

## Usage

### Basic
```powershell
# Run with Python
python tsp_solver.py

# With options
python tsp_solver.py --ascii-map --progress --speed-kmh 200 --minimal
```

Or with the EXE (Windows):
```powershell
LRP-TSP.exe --ascii-map --progress
```

### Flags
- `--minimal`: reduce colors/decorations.
- `--force-ortools`: force OR-Tools if installed; otherwise greedy is used.
- `--ascii-map`: draw an ASCII map of points.
- `--progress`: show a progress bar during processing.
- `--speed-kmh <km/h>`: ETA estimate (default: 170 km/h).
- `--pause`: keep console open after finishing (handy for EXE).
- `--help`: show all options.

### Example output
```
LRP-TSP v0.2.x - Landsraad Route Planner for Dune: Awakening

Visit Harkonnen (12, 34) [🔗 dune.gaming.tools/harkonnen]
Leave Harko and travel to Arrakeen (right entrance).
Visit Atreides (56, 78) [🔗 dune.gaming.tools/atreides]

Distance: 123.4 km | ETA: 0.7 h @ 170 km/h
Solver: Greedy
```

## Data source and customization

- Runtime data: `data/world_data.json` (houses, coords, exits) only.
- No `.raw` files in this repo. If you have your own `.raw` files you can regenerate JSON:

```powershell
python extract_coords.py --mode aggregated --freeze
```

This updates `data/world_data.json` (with overrides like Thorvald in Hagga).

## Contributing

Contributions are welcome:
- Fork the repo and create a feature branch.
- Add tests (see `tests/`).
- Follow code style (Black/Flake8).
- Open a Pull Request with a description.

Issues/Feature requests: [GitHub Issues](https://github.com/ComictypX/TSP-Dune/issues).

## FAQ

**Where can I find my base coordinates?**  
Use the map of your region in-game.  
- Hagga: https://duneawakening.th.gl/maps/Hagga%20Basin  
- Deep Desert: https://duneawakening.th.gl/maps/The%20Deep%20Desert

**Why is the EXE large?**  
PyInstaller bundles Python + deps. UPX compression lowers size to ~20–30 MB.

**Does it work without OR-Tools?**  
Yes. The greedy heuristic is the fallback.

**How do I update house data?**  
Edit `data/world_data.json` or run `extract_coords.py` with your own `.raw` files.

**License?**  
MIT (see LICENSE). Data from Dune: Awakening — no guarantee for completeness.

## Credits

- Dune: Awakening — Funcom.
- OR-Tools — Google.
- Rich — Console UI.
- PyInstaller — EXE build.

## Changelog

See [CHANGELOG.md](CHANGELOG.md)

---

Made with ❤️ for Dune fans. Feedback? [GitHub Discussions](https://github.com/ComictypX/TSP-Dune/discussions).
