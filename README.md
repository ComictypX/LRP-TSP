# TSP-Dune

Ein schneller Routenplaner (TSP) für Dune: Awakening–Häuser. Lädt eine gefrorene Datenbasis (`data/world_data.json`) mit bekannten Häusern und berechnet eine sinnvolle Besuchsreihenfolge inkl. knapper Weganweisungen.

## Features
- Semantic Coloring (Hagga=türkis, Deep Desert=sand‑gold, Arrakeen=grün, Harko=rot)
- Kompakte Übergangspanels (Verlassen + Betreten in einer Zeile)
- Besuchszeilen: klickbare Hauslinks, kurze Koordinatenanzeige
- Optional: ASCII‑Karte, Fortschrittsbalken, OR‑Tools Solver

## Installation
Voraussetzungen: Python 3.10+

```powershell
# optional: virtuelles Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Oder mit Batch:

```powershell
./setup.bat
```

## Nutzung
```powershell
# Start (alle Flags sind optional)
python tsp_solver.py --ascii-map --progress --speed-kmh 25

# oder via Batch
./start.bat --ascii-map --progress
```

Nützliche Flags:
- `--minimal` reduziert Farbe/Schmuck
- `--force-ortools` erzwingt OR‑Tools, falls installiert
- `--ascii-map` ASCII‑Karte anzeigen
- `--progress` Fortschrittsbalken
- `--speed-kmh 25` ETA‑Schätzung

## Datenquelle
- Laufzeit: ausschließlich `data/world_data.json` (Häuser + Exits)
- Keine .raw Dateien im Repo. Falls du eigene .raw Dateien besitzt, kannst du die JSON selbst erzeugen:

```powershell
python extract_coords.py --mode aggregated --freeze
```

Das schreibt/aktualisiert `data/world_data.json` (mit Overrides, z. B. Thorvald).

## Release‑Hinweise
- .raw Dateien werden nicht versioniert/veröffentlicht
- Temporäre/Benutzerspezifische Dateien (z. B. `.tsp_config`, `.venv/`) sind in `.gitignore`

## Lizenz
Bitte bei Bedarf eine Lizenzdatei (z. B. MIT) ergänzen.
