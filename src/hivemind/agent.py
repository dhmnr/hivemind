"""Agent wrapper around ClaudeSDKClient.

Each Agent owns a persistent ClaudeSDKClient session and an asyncio.Queue
of AgentEvent objects that the Discord event consumer reads from.

When a session ends (max_turns or error), the agent can auto-resume using
the session_id from the previous ResultMessage.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from .tools import build_human_server

log = logging.getLogger(__name__)

CLAUDE_JSON = Path.home() / ".claude.json"


def _ensure_project_trusted(project_path: str) -> None:
    """Pre-accept the workspace trust dialog for a project directory.

    Claude Code checks ~/.claude.json → projects → <path> → hasTrustDialogAccepted.
    If missing or false, the CLI shows an interactive prompt that blocks the SDK.
    """
    resolved = os.path.realpath(os.path.expanduser(project_path))

    try:
        data = json.loads(CLAUDE_JSON.read_text()) if CLAUDE_JSON.exists() else {}
    except (json.JSONDecodeError, OSError):
        data = {}

    projects = data.setdefault("projects", {})
    entry = projects.setdefault(resolved, {})

    if entry.get("hasTrustDialogAccepted"):
        return

    entry["hasTrustDialogAccepted"] = True
    CLAUDE_JSON.write_text(json.dumps(data, indent=2) + "\n")
    log.info("Auto-trusted project directory: %s", resolved)


class Status(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentEvent:
    """An event pushed onto the agent's queue for the Discord consumer."""

    kind: str  # start, progress, tool_use, complete, error, resumed, compact
    text: str = ""
    tool_name: str = ""
    tool_input: str = ""
    cost: float | None = None
    session_id: str = ""


class Agent:
    """Wraps a ClaudeSDKClient persistent session."""

    def __init__(
        self,
        name: str,
        project_name: str,
        project_path: str,
        channel_id: int,
        system_prompt: str = "",
        allowed_tools: list[str] | None = None,
    ) -> None:
        self.name = name
        self.project_name = project_name
        self.project_path = project_path
        self.channel_id = channel_id
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools or []

        self.status = Status.IDLE
        self.event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._client: ClaudeSDKClient | None = None
        self._task: asyncio.Task | None = None
        self._session_id: str = ""
        self._total_cost: float = 0.0

    @property
    def full_name(self) -> str:
        return f"{self.project_name}/{self.name}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        resume_session: str | None = None,
        continue_conversation: bool = False,
    ) -> None:
        """Initialize the ClaudeSDKClient and connect.

        Args:
            resume_session: Reconnect to a specific session_id.
            continue_conversation: Pick up the most recent session in this
                project directory (works across bot restarts and CLI sessions).
        """
        _ensure_project_trusted(self.project_path)
        human_server = build_human_server(self.full_name)

        async def on_pre_compact(input_data, tool_use_id, context):
            log.info("Agent %s: context compaction triggered", self.full_name)
            await self.event_queue.put(AgentEvent(kind="compact"))
            return {}

        opts = ClaudeCodeOptions(
            system_prompt=self.system_prompt or None,
            cwd=self.project_path,
            allowed_tools=self.allowed_tools,
            permission_mode="bypassPermissions",
            mcp_servers={"human": human_server},
            resume=resume_session,
            continue_conversation=continue_conversation,
            hooks={
                "PreCompact": [HookMatcher(hooks=[on_pre_compact])],
            },
        )
        self._client = ClaudeSDKClient(opts)
        await self._client.connect()
        self.status = Status.IDLE
        if resume_session:
            self._session_id = resume_session
            log.info("Agent %s resumed session %s", self.full_name, resume_session)
        elif continue_conversation:
            log.info("Agent %s continuing last session (cwd=%s)", self.full_name, self.project_path)
        else:
            log.info("Agent %s connected (cwd=%s)", self.full_name, self.project_path)

    async def run_task(self, task: str) -> None:
        """Send a task and stream response events onto the queue."""
        if self._client is None:
            raise RuntimeError(f"Agent {self.full_name} not started")

        self.status = Status.RUNNING
        await self.event_queue.put(AgentEvent(kind="start", text=task))

        try:
            await self._client.query(task)
            async for msg in self._client.receive_response():
                await self._process_message(msg)
        except Exception as exc:
            self.status = Status.ERROR
            await self.event_queue.put(AgentEvent(kind="error", text=str(exc)))
            log.exception("Agent %s error during task", self.full_name)

    async def send_input(self, text: str) -> None:
        """Send follow-up human input into the active session."""
        if self._client is None:
            raise RuntimeError(f"Agent {self.full_name} not started")

        self.status = Status.RUNNING
        try:
            await self._client.query(text)
            async for msg in self._client.receive_response():
                await self._process_message(msg)
        except Exception as exc:
            self.status = Status.ERROR
            await self.event_queue.put(AgentEvent(kind="error", text=str(exc)))
            log.exception("Agent %s error during send_input", self.full_name)

    async def run_task_background(self, task: str) -> None:
        self._task = asyncio.create_task(
            self.run_task(task), name=f"agent-{self.full_name}"
        )

    async def send_input_background(self, text: str) -> None:
        self._task = asyncio.create_task(
            self.send_input(text), name=f"agent-input-{self.full_name}"
        )

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        if self._client:
            await self._client.disconnect()
            self._client = None
        self.status = Status.IDLE
        log.info("Agent %s stopped", self.full_name)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _process_message(self, msg) -> None:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text = block.text.strip()
                    if not text:
                        continue
                    await self.event_queue.put(
                        AgentEvent(kind="progress", text=text)
                    )

                elif isinstance(block, ToolUseBlock):
                    input_str = str(block.input)
                    if len(input_str) > 300:
                        input_str = input_str[:300] + "…"
                    await self.event_queue.put(
                        AgentEvent(
                            kind="tool_use",
                            tool_name=block.name,
                            tool_input=input_str,
                        )
                    )

                elif isinstance(block, ToolResultBlock):
                    pass

        elif isinstance(msg, ResultMessage):
            self._session_id = msg.session_id
            cost = msg.total_cost_usd or 0.0
            self._total_cost += cost
            if msg.is_error:
                self.status = Status.ERROR
                await self.event_queue.put(
                    AgentEvent(
                        kind="error",
                        text=msg.result or "Unknown error",
                        cost=cost,
                        session_id=msg.session_id,
                    )
                )
            else:
                self.status = Status.DONE
                await self.event_queue.put(
                    AgentEvent(
                        kind="complete",
                        text=msg.result or "",
                        cost=cost,
                        session_id=msg.session_id,
                    )
                )

        elif isinstance(msg, SystemMessage):
            pass
