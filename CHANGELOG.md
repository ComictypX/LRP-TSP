# Changelog

Alle signifikanten Änderungen am Projekt werden hier dokumentiert.

Die Konvention folgt weitgehend Keep a Changelog: https://keepachangelog.com/de/0.3.0/de/

## [Unreleased]
- Vorbereitung auf Multi-OS-Builds (CI) und automatische Prerelease-Erstellung bei Tags.
- README: Hinweise zu Multi-OS-Downloads ergänzt.

## [v0.1.3] - 2025-09-10
### Geändert
- CI: Multi‑OS Matrix-Build für Windows, Linux und macOS hinzugefügt.
- CI: Automatische Erstellung eines Prerelease bei Tag-Push (Release sammelt Artefakte aller Plattformen).
- CI: UPX-Download/Integration für Windows-Builds zur Reduzierung der EXE-Größe.
- README: Hinweise zu Multi‑OS-Assets und Release‑Assets ergänzt.
- UX: Basis-Abfrage verbessert (Frage nach Karte vor Koordinateneingabe). 

## [v0.1.2]
### Geändert
- UX: Pause-Modus (`--pause`) hinzugefügt, damit EXE-Fenster offen bleibt.
- Persistenz: Basis-Konfiguration wird nun in Benutzerkonfigurationsverzeichnis gespeichert.
- Standard-ETA auf 170 km/h gesetzt.
- CI: Release-Prerelease-Flag standardisiert.

## [v0.1.1]
### Geändert
- CI: Workflow bereinigt, Release-Berechtigungen (`contents: write`) gesetzt, Initiales Prerelease-Verhalten eingeführt.
- Build: UPX-Kompression als optionale Optimierung (später in v0.1.3 vollständig integriert).

## [v0.1.0] - Initial Release
- Erstveröffentlichung mit Basissolver, Rich-UI, `data/world_data.json` als Laufzeit-Datenquelle.

---

Hinweis: Wenn du möchtest, kann ich das Changelog automatisch aus Commit-Messages oder Tags generieren (z. B. mit `github-changelog-generator` oder einem GitHub Action Schritt). Sag mir, ob ich das ebenfalls einrichte.
