# Changelog

All notable changes to this project will be documented in this file.

The format roughly follows Keep a Changelog: https://keepachangelog.com/en/1.1.0/

## [Unreleased]
- CI: Prepare multi-OS builds and automatic prerelease creation on tags.
- Docs: Mention multi-OS downloads in README.

## [v0.1.3] - 2025-09-10
### Changed
- CI: Added multi-OS matrix build for Windows, Linux, and macOS.
- CI: Automatic prerelease creation on tag push (collect all platform artifacts).
- CI: UPX download/integration for Windows builds to reduce EXE size.
- Docs: Added notes about multi-OS assets and release artifacts in README.
- UX: Improved base prompt (ask for map before coordinates).

## [v0.1.2]
### Changed
- UX: Added `--pause` to keep the EXE window open after finishing.
- Persistence: Base configuration is now stored in the user's config directory.
- Defaults: ETA speed set to 170 km/h.
- CI: Standardized prerelease flag on releases.

## [v0.1.1]
### Changed
- CI: Cleaned up workflow, set release permissions (`contents: write`), introduced initial prerelease behavior.
- Build: Optional UPX compression (later fully integrated in v0.1.3).

## [v0.1.0] - Initial Release
- First release with base solver, Rich UI, and `data/world_data.json` as runtime data source.

---

Note: We can generate release notes automatically from commits/tags (e.g., via a GitHub Action). Let me know if you want that added.
