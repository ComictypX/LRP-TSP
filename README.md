# TSP-Dune

Ein schneller Routenplaner (TSP - Traveling Salesman Problem) für das Spiel **Dune: Awakening**. Berechnet optimale Besuchsreihenfolgen für Häuser auf den Karten (Hagga, Deep Desert, Arrakeen, Harko) mit knappen, farbkodierten Anweisungen. Verwendet eine gefrorene Datenbasis (`data/world_data.json`) für bekannte Häuser und Exits.

## Features

- **Schnelle Routenberechnung**: Greedy-Algorithmus oder optional OR-Tools für exakte Lösungen.
- **Semantic Coloring**: Farbkodierung nach Karte (Hagga=türkis, Deep Desert=sand-gold, Arrakeen=grün, Harko=rot).
- **Kompakte Anweisungen**: Besuchszeilen mit klickbaren Links, kurze Koordinaten; Übergänge in einer Zeile.
- **Interaktive Prompts**: Frage nach Basis-Haus und Zielen (mit Rich-UI).
- **Optionale Extras**: ASCII-Karte, Fortschrittsbalken, ETA-Schätzung (standard 170 km/h).
- **EXE-Build**: Standalone-EXE für Windows (via PyInstaller + UPX-Kompression).
- **Konfigurierbar**: Flags für Minimal-Modus, Solver-Zwang, etc.

## Installation

### Voraussetzungen
- Python 3.10+ (für Windows-EXE: keine Python-Installation nötig).
- Optional: OR-Tools für bessere Routen (`pip install ortools`).

### Python-Installation
```powershell
# Optional: Virtuelles Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Abhängigkeiten installieren
pip install -r requirements.txt
```

Oder mit Batch-Skript:
```powershell
./setup.bat
```

### Standalone-EXE
Lade die neueste EXE von [GitHub Releases](https://github.com/ComictypX/TSP-Dune/releases) herunter (z. B. `TSP-Dune.exe`). Keine Installation nötig – einfach ausführen.

Hinweis: Es gibt jetzt Multi‑OS‑Builds in den Releases. Suche im Release-Asset nach:
- `TSP-Dune-windows-latest` (Windows .exe)
- `TSP-Dune-linux-latest` (Linux tar.gz)
- `TSP-Dune-macos-latest` (macOS tar.gz)

## Nutzung

### Grundlegende Nutzung
```powershell
# Python-Version starten
python tsp_solver.py

# Mit Optionen
python tsp_solver.py --ascii-map --progress --speed-kmh 200 --minimal
```

Oder mit EXE:
```powershell
TSP-Dune.exe --ascii-map --progress
```

### Flags und Optionen
- `--minimal`: Reduziert Farben und Schmuck für einfachere Ausgabe.
- `--force-ortools`: Erzwingt OR-Tools-Solver (falls installiert), sonst Greedy-Fallback.
- `--ascii-map`: Zeigt eine ASCII-Karte der Route.
- `--progress`: Fortschrittsbalken während Berechnung.
- `--speed-kmh <km/h>`: ETA-Schätzung (Standard: 170 km/h).
- `--pause`: Hält Konsole offen nach Ausführung (nützlich für EXE).
- `--help`: Zeigt alle Optionen.

### Beispiel-Ausgabe
```
TSP-Dune v0.1.2 - Routenplaner für Dune: Awakening

Besuche diese Häuser in dieser Reihenfolge:
Visit Harkonnen (12, 34) [🔗 dune.gaming.tools/harkonnen]
Leave Harko and travel to Arrakeen (right entrance).
Visit Atreides (56, 78) [🔗 dune.gaming.tools/atreides]

Distance: 123.4 km | ETA: 0.7 h @ 170 km/h
Solver: Greedy
```

## Datenquelle und Anpassung

- **Laufzeit-Daten**: Ausschließlich `data/world_data.json` (enthält Häuser, Koordinaten, Exits).
- **Keine .raw-Dateien im Repo**: Diese werden nicht versioniert/veröffentlicht.
- **Eigene Daten erzeugen**: Wenn du .raw-Dateien hast (z. B. von Dune-Spiel), kannst du die JSON selbst erstellen:

```powershell
python extract_coords.py --mode aggregated --freeze
```

Das aktualisiert `data/world_data.json` mit Overrides (z. B. Thorvald in Hagga).

## Contributing

Beiträge sind willkommen! Bitte:
- Forke das Repo und erstelle einen Feature-Branch.
- Füge Tests hinzu (siehe `tests/`).
- Halte dich an den Code-Style (Black/Flake8).
- Öffne einen Pull Request mit Beschreibung.

Für Bugs/Features: [GitHub Issues](https://github.com/ComictypX/TSP-Dune/issues).

## FAQ

**Q: Wo finde ich die Koordinaten meiner Basis?**  
A: Öffne die Karte deines Gebiets im Spiel und suche deine Basis. Die Koordinaten (x, y) werden angezeigt.  
- [Hagga Basin](https://duneawakening.th.gl/maps/Hagga%20Basin)  
- [The Deep Desert](https://duneawakening.th.gl/maps/The%20Deep%20Desert)

**Q: Warum ist die EXE so groß?**  
A: PyInstaller bündelt Python + Abhängigkeiten. UPX-Kompression reduziert es auf ~20-30 MB.

**Q: Funktioniert es ohne OR-Tools?**  
A: Ja, Greedy-Algorithmus als Fallback.

**Q: Wie aktualisiere ich die Haus-Daten?**  
A: Bearbeite `data/world_data.json` oder verwende `extract_coords.py` mit eigenen .raw-Dateien.

**Q: Lizenz?**  
A: MIT (siehe LICENSE). Daten aus Dune: Awakening – keine Garantie für Aktualität.

## Credits

- **Dune: Awakening**: Spiel von Funcom.
- **OR-Tools**: Von Google für Routing.
- **Rich**: Für schöne Konsolen-UI.
- **PyInstaller**: Für EXE-Build.

## Changelog

- **v0.1.2**: UX-Verbesserungen, Pause-Modus, Config-Persistenz.
- **v0.1.1**: Prerelease-Setup, UPX-Kompression.
- **v0.1.0**: Initial Release mit Basis-Features.

Vollständiges Changelog: [CHANGELOG.md](CHANGELOG.md)

---

Entwickelt mit ❤️ für Dune-Fans. Feedback? [GitHub Discussions](https://github.com/ComictypX/TSP-Dune/discussions).
