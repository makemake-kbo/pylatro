#!/usr/bin/env python3
"""Install only the repository-owned Pylatro bridge into a Proton prefix."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

APP_ID = "2379780"
BRIDGE_NAME = "pylatro_bridge"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "mods" / BRIDGE_NAME


def prefix_candidates() -> list[Path]:
    home = Path.home()
    candidates = [
        home / ".local/share/Steam/steamapps/compatdata" / APP_ID / "pfx",
        home / ".steam/steam/steamapps/compatdata" / APP_ID / "pfx",
        home / "snap/steam/common/.local/share/Steam/steamapps/compatdata" / APP_ID / "pfx",
        home
        / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/compatdata"
        / APP_ID
        / "pfx",
    ]
    compat = os.environ.get("STEAM_COMPAT_DATA_PATH")
    if compat:
        candidates.insert(0, Path(compat) / "pfx")
    return list(dict.fromkeys(path.resolve() for path in candidates))


def balatro_dirs(prefix: Path) -> tuple[Path, Path]:
    roaming = prefix / "drive_c/users/steamuser/AppData/Roaming/Balatro"
    return roaming, roaming / "Mods"


def find_prefix(explicit: Path | None) -> Path:
    candidates = [explicit.resolve()] if explicit else prefix_candidates()
    for prefix in candidates:
        roaming, _ = balatro_dirs(prefix)
        if roaming.is_dir():
            return prefix
    searched = "\n  ".join(str(path) for path in candidates)
    raise SystemExit(f"Could not locate Balatro's Proton prefix. Checked:\n  {searched}")


def steam_roots() -> list[Path]:
    home = Path.home()
    return [
        home / ".local/share/Steam",
        home / ".steam/steam",
        home / "snap/steam/common/.local/share/Steam",
        home / ".var/app/com.valvesoftware.Steam/.local/share/Steam",
    ]


def verify_lovely(prefix: Path) -> Path:
    candidates = []
    if len(prefix.parents) >= 3:
        candidates.append(prefix.parents[2] / "common/Balatro/version.dll")
    candidates.extend(
        root / "steamapps/common/Balatro/version.dll" for root in steam_roots()
    )
    for dll in dict.fromkeys(candidates):
        if dll.is_file():
            return dll
    raise SystemExit(
        "Lovely was not found: version.dll must be next to Balatro.exe. "
        "See docs/live_bridge.md before installing the bridge."
    )


def verify_steamodded(mods_dir: Path) -> Path:
    if not mods_dir.is_dir():
        raise SystemExit(f"Steamodded Mods directory does not exist: {mods_dir}")
    for child in mods_dir.iterdir():
        if (
            child.is_dir()
            and child.name.lower() in {"smods", "steamodded"}
            and ((child / "src").is_dir() or (child / "main.lua").is_file())
        ):
            return child
    raise SystemExit(
        f"Steamodded was not found under {mods_dir}. Install it before the bridge."
    )


def install(destination: Path, *, copy: bool) -> None:
    if destination.is_symlink() and destination.resolve() == SOURCE.resolve() and not copy:
        print(f"Bridge already linked: {destination}")
        return
    if destination.exists() or destination.is_symlink():
        raise SystemExit(
            f"Refusing to overwrite existing path: {destination}\n"
            "Move it aside or run this helper with --uninstall first."
        )
    if copy:
        shutil.copytree(SOURCE, destination)
        print(f"Copied bridge to {destination}")
    else:
        destination.symlink_to(SOURCE, target_is_directory=True)
        print(f"Linked bridge to {destination} -> {SOURCE}")


def uninstall(destination: Path) -> None:
    if destination.is_symlink():
        destination.unlink()
        print(f"Removed bridge link: {destination}")
        return
    if destination.is_dir():
        manifest = destination / "pylatro_bridge.json"
        if not manifest.is_file():
            raise SystemExit(f"Refusing to remove unrecognized directory: {destination}")
        shutil.rmtree(destination)
        print(f"Removed copied bridge directory: {destination}")
        return
    print(f"Bridge is not installed at {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, help="Explicit Proton pfx directory")
    parser.add_argument("--copy", action="store_true", help="Copy instead of creating a symlink")
    parser.add_argument("--uninstall", action="store_true", help="Remove only the Pylatro bridge")
    args = parser.parse_args()

    prefix = find_prefix(args.prefix)
    _, mods_dir = balatro_dirs(prefix)
    destination = mods_dir / BRIDGE_NAME
    if args.uninstall:
        uninstall(destination)
        return
    lovely = verify_lovely(prefix)
    steamodded = verify_steamodded(mods_dir)
    print(f"Lovely: {lovely}")
    print(f"Steamodded: {steamodded}")
    install(destination, copy=args.copy)


if __name__ == "__main__":
    main()
