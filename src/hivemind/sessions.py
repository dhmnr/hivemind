"""Scan Claude Code session files for a project directory."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


@dataclass
class SessionInfo:
    session_id: str
    timestamp: str
    task: str  # first user message (truncated)


def _project_dir_name(project_path: str) -> str:
    """Convert an absolute project path to the Claude Code directory name.

    /home/dm/hivemind → -home-dm-hivemind
    """
    resolved = os.path.realpath(os.path.expanduser(project_path))
    return resolved.replace("/", "-")


def list_sessions(project_path: str, limit: int = 20) -> list[SessionInfo]:
    """List recent Claude Code sessions for a project directory."""
    dir_name = _project_dir_name(project_path)
    session_dir = CLAUDE_PROJECTS_DIR / dir_name

    if not session_dir.is_dir():
        return []

    jsonl_files = sorted(
        session_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    sessions: list[SessionInfo] = []
    for path in jsonl_files[:limit]:
        info = _parse_session_file(path)
        if info:
            sessions.append(info)

    return sessions


def _parse_session_file(path: Path) -> SessionInfo | None:
    """Extract session info from the first user message in a JSONL file."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj.get("type") != "user":
                    continue

                session_id = obj.get("sessionId", "")
                timestamp = obj.get("timestamp", "")
                message = obj.get("message", {})
                content = message.get("content", "")

                if isinstance(content, str):
                    task = content
                elif isinstance(content, list):
                    # Tool result arrays — skip, look for the next user message
                    # that has a plain string content
                    continue
                else:
                    task = ""

                task = task.strip().replace("\n", " ")
                if len(task) > 100:
                    task = task[:100] + "\u2026"

                if session_id:
                    return SessionInfo(
                        session_id=session_id,
                        timestamp=timestamp,
                        task=task or "(no prompt)",
                    )
    except (OSError, json.JSONDecodeError):
        pass

    return None
