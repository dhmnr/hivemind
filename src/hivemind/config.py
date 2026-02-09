"""Config loading from env and YAML."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ProjectPreset:
    path: str
    system_prompt: str = ""
    allowed_tools: list[str] = field(default_factory=list)


@dataclass
class Config:
    bot_token: str
    guild_id: int | None = None
    project_presets: dict[str, ProjectPreset] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> Config:
        path = Path(path)
        if not path.exists():
            return cls(bot_token=os.environ.get("DISCORD_BOT_TOKEN", ""))

        raw = path.read_text()
        raw = re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            raw,
        )
        data = yaml.safe_load(raw)

        presets: dict[str, ProjectPreset] = {}
        for name, d in data.get("projects", {}).items():
            presets[name] = ProjectPreset(
                path=os.path.expanduser(d.get("path", ".")),
                system_prompt=d.get("system_prompt", ""),
                allowed_tools=d.get("allowed_tools", []),
            )

        guild_id_raw = data.get("discord", {}).get("guild_id", "")
        guild_id = int(guild_id_raw) if guild_id_raw and str(guild_id_raw).isdigit() else None

        return cls(
            bot_token=data.get("discord", {}).get("bot_token", ""),
            guild_id=guild_id,
            project_presets=presets,
        )
