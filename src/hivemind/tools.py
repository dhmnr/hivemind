"""Custom MCP tools for human-in-the-loop interaction.

Provides the `ask_human` tool that Claude agents can call to request
human input via Discord, and the ApprovalBridge that coordinates
between the MCP tool handler and Discord UI callbacks.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from claude_code_sdk import create_sdk_mcp_server, tool

log = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """A pending human-input request."""

    request_id: str
    agent_name: str
    question: str
    options: list[str]
    event: asyncio.Event = field(default_factory=asyncio.Event)
    response: str | None = None


class ApprovalBridge:
    """Bridges MCP tool calls with Discord UI.

    The agent calls `ask_human` → bridge creates a PendingRequest and fires
    a callback so the Discord side can post buttons → human clicks a button →
    Discord calls `resolve()` → the awaiting tool handler receives the answer.
    """

    def __init__(self) -> None:
        self._pending: dict[str, PendingRequest] = {}
        # Callback: (request: PendingRequest) -> None
        # Set by the bot to post the question to Discord.
        self.on_request: asyncio.Future | None = None
        self._request_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()

    async def request(
        self,
        agent_name: str,
        question: str,
        options: list[str] | None = None,
    ) -> str:
        """Called from the MCP tool handler.  Blocks until the human responds."""
        req = PendingRequest(
            request_id=str(uuid.uuid4()),
            agent_name=agent_name,
            question=question,
            options=options or [],
        )
        self._pending[req.request_id] = req
        log.info("ask_human request %s from agent %s: %s", req.request_id, agent_name, question)
        await self._request_queue.put(req)
        await req.event.wait()
        self._pending.pop(req.request_id, None)
        return req.response or ""

    def resolve(self, request_id: str, response: str) -> bool:
        """Called from Discord button/modal callbacks."""
        req = self._pending.get(request_id)
        if req is None:
            log.warning("resolve called for unknown request %s", request_id)
            return False
        req.response = response
        req.event.set()
        log.info("Resolved request %s with: %s", request_id, response)
        return True

    async def wait_for_request(self) -> PendingRequest:
        """Wait for the next pending request (used by event consumer)."""
        return await self._request_queue.get()


# ---------------------------------------------------------------------------
# Global bridge instance – shared between MCP tool handlers and Discord bot
# ---------------------------------------------------------------------------
approval_bridge = ApprovalBridge()


def build_human_server(agent_name: str):
    """Build an MCP server config with the ask_human tool bound to a specific agent."""

    @tool(
        "ask_human",
        "Ask the human operator a question and wait for their response. "
        "Use this when you need approval, clarification, or a decision.",
        {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the human.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of suggested answers.",
                },
            },
            "required": ["question"],
        },
    )
    async def ask_human(args: dict) -> dict:
        question = args["question"]
        options = args.get("options", [])
        response = await approval_bridge.request(
            agent_name=agent_name,
            question=question,
            options=options,
        )
        return {"content": [{"type": "text", "text": f"Human responded: {response}"}]}

    return create_sdk_mcp_server("human", tools=[ask_human])
