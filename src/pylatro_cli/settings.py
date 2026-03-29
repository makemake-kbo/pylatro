"""User settings with persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "pylatro"
SETTINGS_FILE = CONFIG_DIR / "settings.json"


class KeyMode(StrEnum):
    VIM = "vim"
    ARROWS = "arrows"


@dataclass
class UserSettings:
    key_mode: KeyMode = KeyMode.VIM

    @classmethod
    def load(cls) -> UserSettings:
        if SETTINGS_FILE.exists():
            try:
                raw = json.loads(SETTINGS_FILE.read_text())
                return cls(key_mode=KeyMode(raw.get("key_mode", "vim")))
            except (json.JSONDecodeError, ValueError):
                pass
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2))
