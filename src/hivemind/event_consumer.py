"""Event consumer — reads AgentEvents from agent queues and posts to Discord.

One consumer task runs per agent. Uses a single editable "status line" message
that shows the current tool call (like Claude Code's terminal spinner), plus
reactions on the triggering message for start/complete/error state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import discord

from .agent import AgentEvent, Status
from .tools import approval_bridge, collab_bridge
from .views import ApprovalView, event_embed

if TYPE_CHECKING:
    from .agent import Agent

log = logging.getLogger(__name__)

BATCH_INTERVAL = 2.0
MAX_MSG_LEN = 1900

# Status line frames for animation
SPINNER = ["\u280b", "\u2819", "\u2838", "\u2830", "\u2824", "\u2807"]

TOOL_EMOJIS = {
    "Read": "\U0001f4d6",
    "Write": "\u270d\ufe0f",
    "Edit": "\u270f\ufe0f",
    "Bash": "\U0001f4bb",
    "Glob": "\U0001f50d",
    "Grep": "\U0001f50e",
    "ask_human": "\U0001f9d1",
    "post_to_main": "\U0001f4e2",
    "list_agents": "\U0001f465",
    "Task": "\U0001f4cb",
    "WebFetch": "\U0001f310",
    "WebSearch": "\U0001f310",
    "NotebookEdit": "\U0001f4d3",
}


def _tool_emoji(name: str) -> str:
    return TOOL_EMOJIS.get(name, "\U0001f527")


def _tool_label(name: str, input_str: str) -> str:
    """Build a compact one-line label for a tool call."""
    emoji = _tool_emoji(name)
    # Extract the most useful bit from the input
    short = input_str.strip().replace("\n", " ")
    if len(short) > 120:
        short = short[:120] + "\u2026"
    return f"{emoji} **{name}** {short}" if short else f"{emoji} **{name}**"


async def consume_events(
    agent: Agent,
    channel: discord.TextChannel,
    approvals_channel: discord.TextChannel | None = None,
) -> None:
    """Long-running task: drain agent.event_queue -> Discord messages."""
    progress_buffer: list[str] = []
    status_msg: discord.Message | None = None  # the editable status line
    tool_history: list[str] = []  # recent tool labels for the status message
    task_start_time: float = 0.0
    trigger_msg: discord.Message | None = None  # last message in channel before start

    async def flush_progress() -> None:
        nonlocal progress_buffer
        if not progress_buffer:
            return
        text = "\n".join(progress_buffer)
        progress_buffer = []
        for chunk in _split_text(text, MAX_MSG_LEN):
            await channel.send(chunk)

    async def update_status(current_tool: str | None = None) -> None:
        """Edit the status line message to show recent tool activity."""
        nonlocal status_msg, tool_history

        if current_tool:
            tool_history.append(current_tool)
            # Keep last 8 tool calls
            if len(tool_history) > 8:
                tool_history = tool_history[-8:]

        elapsed = time.time() - task_start_time if task_start_time else 0
        elapsed_str = _format_elapsed(elapsed)

        # Build status content: recent tools + elapsed
        frame = SPINNER[int(elapsed) % len(SPINNER)]
        lines = [f"{frame} **Working** \u2014 {elapsed_str}"]
        if tool_history:
            # Show last few tools, most recent last
            display = tool_history[-5:]
            # Dim older ones, bold the current one
            for i, t in enumerate(display):
                if i < len(display) - 1:
                    lines.append(f"\u2003\u2502 {t}")
                else:
                    lines.append(f"\u2003\u251c {t}")

        content = "\n".join(lines)

        try:
            if status_msg is None:
                status_msg = await channel.send(content)
            else:
                await status_msg.edit(content=content)
        except discord.HTTPException:
            # Message may have been deleted, recreate
            status_msg = await channel.send(content)

    async def finish_status(final_line: str) -> None:
        """Replace the status message with a final summary."""
        nonlocal status_msg, tool_history
        tool_history = []
        try:
            if status_msg:
                await status_msg.edit(content=final_line)
                status_msg = None
            else:
                await channel.send(final_line)
        except discord.HTTPException:
            await channel.send(final_line)

    while True:
        try:
            try:
                event: AgentEvent = await asyncio.wait_for(
                    agent.event_queue.get(), timeout=BATCH_INTERVAL
                )
            except asyncio.TimeoutError:
                await flush_progress()
                # Refresh the status spinner if agent is still running
                if status_msg and tool_history:
                    await update_status()
                continue

            if event.kind == "progress":
                progress_buffer.append(event.text)
                total_len = sum(len(t) for t in progress_buffer)
                if total_len >= MAX_MSG_LEN:
                    await flush_progress()
                continue

            # Non-progress events: flush any buffered progress first
            await flush_progress()

            if event.kind == "start":
                task_start_time = time.time()
                tool_history = []
                status_msg = None
                # Try to find the triggering message to react on
                try:
                    async for msg in channel.history(limit=3):
                        if not msg.author.bot:
                            trigger_msg = msg
                            await msg.add_reaction("\u23f3")  # hourglass
                            break
                except discord.HTTPException:
                    pass

            elif event.kind == "tool_use":
                label = _tool_label(event.tool_name, event.tool_input)
                await update_status(label)

            elif event.kind == "complete":
                elapsed = time.time() - task_start_time if task_start_time else 0
                elapsed_str = _format_elapsed(elapsed)
                cost_str = f" \u2014 ${event.cost:.4f}" if event.cost else ""
                await finish_status(f"\u2705 **Done** in {elapsed_str}{cost_str}")

                # Update reaction on trigger message
                if trigger_msg:
                    try:
                        await trigger_msg.remove_reaction("\u23f3", channel.guild.me)
                        await trigger_msg.add_reaction("\u2705")
                    except discord.HTTPException:
                        pass
                    trigger_msg = None
                task_start_time = 0.0

            elif event.kind == "error":
                elapsed = time.time() - task_start_time if task_start_time else 0
                elapsed_str = _format_elapsed(elapsed)
                error_text = event.text[:200] if event.text else "Unknown error"
                await finish_status(f"\u274c **Error** after {elapsed_str}\n```\n{error_text}\n```")

                if trigger_msg:
                    try:
                        await trigger_msg.remove_reaction("\u23f3", channel.guild.me)
                        await trigger_msg.add_reaction("\u274c")
                    except discord.HTTPException:
                        pass
                    trigger_msg = None
                task_start_time = 0.0

            elif event.kind == "compact":
                label = "\U0001f5dc\ufe0f **Compacting** \u2014 compressing conversation history"
                await update_status(label)

            elif event.kind == "resumed":
                await channel.send(f"\U0001f504 {event.text}")

        except asyncio.CancelledError:
            log.info("Event consumer for %s cancelled", agent.name)
            return
        except Exception:
            log.exception("Event consumer error for %s", agent.name)
            await asyncio.sleep(1)


async def consume_approval_requests(
    bot: discord.Client,
    hivemind_bot,
) -> None:
    """Listens for ask_human MCP tool requests and posts ApprovalView to Discord."""
    while True:
        try:
            req = await approval_bridge.wait_for_request()
            agent = hivemind_bot._all_agents().get(req.agent_name)
            if agent is None:
                log.warning("Approval request for unknown agent %s", req.agent_name)
                approval_bridge.resolve(req.request_id, "Agent not found")
                continue

            channel = bot.get_channel(agent.channel_id)
            if channel is None or not isinstance(channel, discord.TextChannel):
                log.warning("Channel %d not found for agent %s", agent.channel_id, req.agent_name)
                approval_bridge.resolve(req.request_id, "Channel not found")
                continue

            embed = event_embed("question", req.question, req.agent_name)
            view = ApprovalView(
                request_id=req.request_id,
                options=req.options,
                bridge=approval_bridge,
                agent=agent,
            )
            await channel.send(
                "\U0001f514 **Agent needs input:**",
                embed=embed,
                view=view,
            )
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Error in approval request consumer")
            await asyncio.sleep(1)


async def consume_collab_messages(
    bot: discord.Client,
    hivemind_bot,
) -> None:
    """Listens for agent post_to_main calls and routes to #main + all peers."""
    webhooks: dict[int, discord.Webhook] = {}

    async def get_webhook(channel: discord.TextChannel) -> discord.Webhook:
        """Get or create a 'hivemind' webhook for the channel."""
        if channel.id in webhooks:
            return webhooks[channel.id]
        # Check for existing webhook
        for wh in await channel.webhooks():
            if wh.name == "hivemind":
                webhooks[channel.id] = wh
                return wh
        # Create new one
        wh = await channel.create_webhook(name="hivemind")
        webhooks[channel.id] = wh
        return wh

    while True:
        try:
            msg = await collab_bridge.wait_for_message()

            agent = hivemind_bot._all_agents().get(msg.agent_name)
            if agent is None:
                log.warning("Collab message from unknown agent %s", msg.agent_name)
                continue

            proj = hivemind_bot.projects.get(agent.project_name)
            if proj is None:
                log.warning("Project %s not found for collab message", agent.project_name)
                continue

            main_channel = bot.get_channel(proj.main_channel_id)
            if main_channel is None or not isinstance(main_channel, discord.TextChannel):
                log.warning("Main channel %d not found", proj.main_channel_id)
                continue

            # Post to #main via webhook (appears as the agent's name)
            webhook = await get_webhook(main_channel)
            content = msg.message
            if len(content) > 2000:
                content = content[:1997] + "..."
            await webhook.send(content, username=agent.name)

            # Deliver to ALL other agents in the project
            for peer_name, peer in proj.agents.items():
                if peer.full_name == msg.agent_name:
                    continue  # skip self
                text = f"[#main from {agent.name}] {msg.message}"
                if peer.status in (Status.IDLE, Status.DONE, Status.ERROR):
                    await peer.run_task_background(text)
                else:
                    await peer.send_input_background(text)

        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Error in collab message consumer")
            await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_text(text: str, limit: int) -> list[str]:
    """Split text into chunks at newline boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.split("\n"):
        if current_len + len(line) + 1 > limit and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"
