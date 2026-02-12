"""Custom MCP tools for human-in-the-loop and agent collaboration.

Provides the `ask_human` tool that Claude agents can call to request
human input via Discord, and the ApprovalBridge that coordinates
between the MCP tool handler and Discord UI callbacks.

Also provides collaboration tools (`post_to_main`, `list_agents`) that
let agents communicate through the project's #main channel.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
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


# ---------------------------------------------------------------------------
# Collaboration bridge – agent-to-#main channel messaging
# ---------------------------------------------------------------------------


@dataclass
class CollabMessage:
    """An agent wants to post a message to #main."""

    agent_name: str  # full_name like "project/agent"
    message: str
    mentioned_agents: list[str]


class CollabBridge:
    """Fire-and-forget queue for agent messages destined for #main."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[CollabMessage] = asyncio.Queue()

    async def post(
        self, agent_name: str, message: str, mentioned_agents: list[str]
    ) -> None:
        await self._queue.put(
            CollabMessage(
                agent_name=agent_name,
                message=message,
                mentioned_agents=mentioned_agents,
            )
        )

    async def wait_for_message(self) -> CollabMessage:
        return await self._queue.get()


collab_bridge = CollabBridge()


def build_collab_server(
    agent_name: str,
    get_peers: Callable[[], list[dict[str, str]]],
):
    """Build an MCP server with collaboration tools bound to a specific agent."""

    @tool(
        "post_to_main",
        "Post a message to the project's #main channel. All team members "
        "(human and agent) can see it. Use @agent_name to address a specific "
        "peer (e.g. '@tester the API is ready for testing').",
        {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to post. Use @agent_name to mention peers.",
                },
            },
            "required": ["message"],
        },
    )
    async def post_to_main(args: dict) -> dict:
        message = args["message"]
        peers = get_peers()
        peer_names = {p["name"] for p in peers}
        mentioned: list[str] = []
        for word in message.split():
            if word.startswith("@"):
                name = word[1:].strip(".,!?;:")
                if name in peer_names:
                    mentioned.append(name)
        await collab_bridge.post(agent_name, message, mentioned)
        return {"content": [{"type": "text", "text": "Message posted to #main."}]}

    @tool(
        "list_agents",
        "List all agents in your project with their current status, role, "
        "and what they're working on.",
        {
            "type": "object",
            "properties": {},
        },
    )
    async def list_agents(args: dict) -> dict:
        peers = get_peers()
        if not peers:
            return {
                "content": [
                    {"type": "text", "text": "No other agents in this project."}
                ]
            }
        lines: list[str] = []
        for p in peers:
            line = f"- {p['name']} [{p['status']}]"
            if p.get("persona"):
                line += f" ({p['persona']})"
            elif p.get("role"):
                line += f" ({p['role']})"
            if p.get("current_task"):
                line += f": {p['current_task']}"
            lines.append(line)
        text = "Agents in this project:\n" + "\n".join(lines)
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server("collab", tools=[post_to_main, list_agents])
