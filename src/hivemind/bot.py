"""Discord bot — slash commands, event handlers, and lifecycle management.

Organized around Projects (Discord category + directory) containing
independent Agents (each with its own channel and ClaudeSDKClient session).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import discord
from discord import app_commands

from .agent import Agent, Status
from .config import Config
from .event_consumer import consume_approval_requests, consume_collab_messages, consume_events
from .personas import get_persona
from .sessions import list_sessions
from .views import status_embed

log = logging.getLogger(__name__)

STATE_FILE = Path("state.json")

DEFAULT_TOOLS = [
    "Read", "Write", "Edit", "Bash", "Glob", "Grep",
    "mcp__human__ask_human",
    "mcp__collab__post_to_main",
    "mcp__collab__list_agents",
]


@dataclass
class Project:
    """A project groups agents that share a working directory."""

    name: str
    path: str
    category_id: int
    main_channel_id: int
    system_prompt: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    agents: dict[str, Agent] = field(default_factory=dict)


class HivemindBot(discord.Client):
    """The main Discord bot that orchestrates Claude Code agents."""

    def __init__(self, config: Config) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.config = config
        self.tree = app_commands.CommandTree(self)
        self.projects: dict[str, Project] = {}
        self._consumer_tasks: dict[str, asyncio.Task] = {}
        self._approval_task: asyncio.Task | None = None
        self._collab_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Serialize project metadata (including agents) to STATE_FILE."""
        data = {}
        for name, proj in self.projects.items():
            data[name] = {
                "name": proj.name,
                "path": proj.path,
                "category_id": proj.category_id,
                "main_channel_id": proj.main_channel_id,
                "system_prompt": proj.system_prompt,
                "allowed_tools": proj.allowed_tools,
                "agents": [
                    {
                        "name": agent.name,
                        "channel_id": agent.channel_id,
                        "session_id": agent._session_id,
                        "role": agent.role,
                        "persona": agent.persona,
                    }
                    for agent in proj.agents.values()
                ],
            }
        STATE_FILE.write_text(json.dumps(data, indent=2))
        log.info("State saved (%d projects)", len(data))

    def _load_state(self) -> None:
        """Load project metadata from STATE_FILE if it exists."""
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load state file: %s", exc)
            return
        for name, info in data.items():
            proj = Project(
                name=info["name"],
                path=info["path"],
                category_id=info["category_id"],
                main_channel_id=info["main_channel_id"],
                system_prompt=info.get("system_prompt", ""),
                allowed_tools=info.get("allowed_tools", []),
            )
            for agent_info in info.get("agents", []):
                agent = Agent(
                    name=agent_info["name"],
                    project_name=proj.name,
                    project_path=proj.path,
                    channel_id=agent_info["channel_id"],
                    system_prompt=proj.system_prompt,
                    allowed_tools=proj.allowed_tools,
                    role=agent_info.get("role", ""),
                    persona=agent_info.get("persona", ""),
                )
                agent._session_id = agent_info.get("session_id", "")
                proj.agents[agent.name] = agent
            self.projects[name] = proj
        log.info("State loaded (%d projects)", len(self.projects))

    async def _resume_agents(self) -> None:
        """Resume all persisted agents after a bot restart."""
        for proj in list(self.projects.values()):
            failed: list[str] = []
            for name, agent in list(proj.agents.items()):
                channel = self.get_channel(agent.channel_id)
                if channel is None:
                    log.warning(
                        "Channel %d for agent %s no longer exists, removing",
                        agent.channel_id, agent.full_name,
                    )
                    failed.append(name)
                    continue

                try:
                    agent.system_prompt = self._build_agent_system_prompt(proj, agent)
                    peers_fn = lambda pn=proj.name: self._get_peers_for_project(pn)
                    if agent._session_id:
                        await agent.start(resume_session=agent._session_id, get_peers=peers_fn)
                    else:
                        await agent.start(continue_conversation=True, get_peers=peers_fn)
                except Exception as exc:
                    log.warning(
                        "Failed to resume agent %s: %s", agent.full_name, exc,
                    )
                    failed.append(name)
                    continue

                self._consumer_tasks[agent.full_name] = asyncio.create_task(
                    consume_events(agent, channel),
                    name=f"consumer-{agent.full_name}",
                )
                sid = agent._session_id
                if sid:
                    label = f"resuming session `{sid[:12]}…`"
                else:
                    label = "continuing last session"
                log.info("Auto-resumed agent %s (%s)", agent.full_name, label)
                await channel.send(
                    f"**{agent.name}** auto-resumed after bot restart ({label})."
                )

            for name in failed:
                del proj.agents[name]

            if failed:
                self._save_state()

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully stop all agents and save state before disconnecting."""
        log.info("Graceful shutdown initiated")

        # Cancel watchdog
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()

        # Cancel all consumer tasks
        for name, task in self._consumer_tasks.items():
            if not task.done():
                task.cancel()
                log.info("Cancelled consumer task: %s", name)
        self._consumer_tasks.clear()

        # Cancel approval and collab consumers
        if self._approval_task and not self._approval_task.done():
            self._approval_task.cancel()
        if self._collab_task and not self._collab_task.done():
            self._collab_task.cancel()

        # Stop all agents
        for proj in self.projects.values():
            for agent in proj.agents.values():
                try:
                    await agent.stop()
                except Exception as exc:
                    log.warning("Error stopping agent %s: %s", agent.full_name, exc)

        # Persist latest session_ids
        self._save_state()
        log.info("State saved during shutdown")

        await super().close()

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    async def _watchdog(self) -> None:
        """Background task that monitors agent health every 30 seconds."""
        await asyncio.sleep(10)  # initial delay to let things settle
        while True:
            try:
                await self._watchdog_tick()
            except Exception:
                log.exception("Watchdog tick error")
            await asyncio.sleep(30)

    async def _watchdog_tick(self) -> None:
        # Dead consumer recovery
        for full_name, task in list(self._consumer_tasks.items()):
            if task.done():
                log.warning("Consumer task for %s is dead, restarting", full_name)
                # Find the agent and its channel
                agents = self._all_agents()
                agent = agents.get(full_name)
                if agent is None:
                    del self._consumer_tasks[full_name]
                    continue
                channel = self.get_channel(agent.channel_id)
                if channel is None:
                    log.warning("Channel for %s no longer exists", full_name)
                    del self._consumer_tasks[full_name]
                    continue
                self._consumer_tasks[full_name] = asyncio.create_task(
                    consume_events(agent, channel),
                    name=f"consumer-{full_name}",
                )
                log.info("Restarted consumer task for %s", full_name)

        # Agent auto-restart and stuck detection
        for proj in self.projects.values():
            for agent in proj.agents.values():
                # Auto-restart agents with repeated errors
                if agent._consecutive_errors >= 3 and agent._client is not None:
                    log.warning(
                        "Agent %s has %d consecutive errors, attempting restart",
                        agent.full_name, agent._consecutive_errors,
                    )
                    try:
                        await agent.stop()
                        peers_fn = lambda pn=agent.project_name: self._get_peers_for_project(pn)
                        if agent._session_id:
                            await agent.start(resume_session=agent._session_id, get_peers=peers_fn)
                        else:
                            await agent.start(continue_conversation=True, get_peers=peers_fn)
                        agent._consecutive_errors = 0
                        log.info("Watchdog restarted agent %s", agent.full_name)
                        # Notify channel
                        channel = self.get_channel(agent.channel_id)
                        if channel and hasattr(channel, "send"):
                            await channel.send(
                                f"**{agent.name}** auto-restarted by watchdog "
                                f"after repeated errors."
                            )
                    except Exception as exc:
                        log.warning(
                            "Watchdog failed to restart agent %s: %s",
                            agent.full_name, exc,
                        )

                # Stuck agent detection (warning only)
                if (
                    agent.status == Status.RUNNING
                    and agent._last_activity > 0
                    and time.time() - agent._last_activity > 300
                ):
                    log.warning(
                        "Agent %s appears stuck (no activity for %.0fs)",
                        agent.full_name,
                        time.time() - agent._last_activity,
                    )

        # Periodic state save
        self._save_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_agents(self) -> dict[str, Agent]:
        out: dict[str, Agent] = {}
        for proj in self.projects.values():
            for agent in proj.agents.values():
                out[agent.full_name] = agent
        return out

    def _get_peers_for_project(self, project_name: str) -> list[dict[str, str]]:
        """Return current info for all agents in a project."""
        proj = self.projects.get(project_name)
        if proj is None:
            return []
        return [
            {
                "name": agent.name,
                "status": agent.status.value,
                "role": agent.role,
                "persona": agent.persona,
                "current_task": agent._current_task,
            }
            for agent in proj.agents.values()
        ]

    def _build_agent_system_prompt(self, proj: Project, agent: Agent) -> str:
        """Build a system prompt combining persona/role, project context, and collab info."""
        parts: list[str] = []

        # Layer 1: Identity — persona prompt, role, or both
        persona_obj = get_persona(agent.persona) if agent.persona else None
        if persona_obj:
            parts.append(persona_obj.prompt)
            if agent.role:
                parts.append(f"Additional instructions: {agent.role}")
        elif agent.role:
            parts.append(f"Your role: {agent.role}")

        if proj.system_prompt:
            parts.append(proj.system_prompt)

        peer_names = []
        for a in proj.agents.values():
            if a.name == agent.name:
                continue
            label = a.name
            if a.persona:
                label += f" ({a.persona})"
            elif a.role:
                label += f" ({a.role})"
            peer_names.append(label)
        collab = (
            "\n## Collaboration\n"
            "You are part of a team of agents working on this project. "
            "The #main channel is a shared space where all messages are broadcast "
            "to every agent.\n\n"
            "Available tools:\n"
            "- `mcp__collab__post_to_main`: Post a message to #main. "
            "Use @agent_name to address a specific peer.\n"
            "- `mcp__collab__list_agents`: See all agents, their status, role, "
            "and current tasks.\n\n"
            "Guidelines:\n"
            "- Messages from #main arrive prefixed with [#main from ...]. "
            "ALWAYS reply to #main messages using `mcp__collab__post_to_main`. "
            "Text you output normally goes to your private channel, NOT #main. "
            "The ONLY way to talk in #main is via the `post_to_main` tool.\n"
            "- If you are @mentioned in a #main message, you MUST respond via `post_to_main`.\n"
            "- If you are NOT @mentioned, only respond if you have something "
            "specifically valuable to add. Avoid noise.\n"
            "- Post to #main for milestones, blockers, questions, or handoffs.\n"
        )
        if peer_names:
            collab += f"- Current peers: {', '.join(peer_names)}\n"

        parts.append(collab)
        return "\n\n".join(parts)

    async def _handle_main_message(self, proj: Project, message: discord.Message) -> None:
        """Broadcast a human's #main message to all agents in the project."""
        if not proj.agents:
            return

        text = f"[#main from {message.author.display_name}] {message.content}"
        await message.add_reaction("\U0001f4e8")

        for agent in proj.agents.values():
            if agent.status in (Status.IDLE, Status.DONE, Status.ERROR):
                await agent.run_task_background(text)
            else:
                await agent.send_input_background(text)

    def _agent_for_channel(self, channel_id: int) -> Agent | None:
        for proj in self.projects.values():
            for agent in proj.agents.values():
                if agent.channel_id == channel_id:
                    return agent
        return None

    def _project_for_channel(self, channel_id: int) -> Project | None:
        """Find the project whose category contains this channel."""
        for proj in self.projects.values():
            if channel_id == proj.main_channel_id:
                return proj
            ch = self.get_channel(channel_id)
            if ch and hasattr(ch, "category_id") and ch.category_id == proj.category_id:
                return proj
        return None

    def _resolve_project(
        self, interaction: discord.Interaction, project_name: str | None
    ) -> Project | None:
        """Resolve project by name or by the channel the command was run in."""
        if project_name:
            return self.projects.get(project_name)
        return self._project_for_channel(interaction.channel_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup_hook(self) -> None:
        _register_commands(self)
        if self.config.guild_id:
            guild = discord.Object(id=self.config.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            log.info("Slash commands synced to guild %s", self.config.guild_id)
        else:
            await self.tree.sync()
            log.info("Slash commands synced globally (may take up to 1h)")

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id)
        self._load_state()
        await self._resume_agents()
        self._approval_task = asyncio.create_task(
            consume_approval_requests(self, self),
            name="approval-consumer",
        )
        self._collab_task = asyncio.create_task(
            consume_collab_messages(self, self),
            name="collab-consumer",
        )
        self._watchdog_task = asyncio.create_task(
            self._watchdog(), name="watchdog",
        )
        log.info("Watchdog started")

    # ------------------------------------------------------------------
    # Channel messages → agent input
    # ------------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        # Check if message is in a project's #main channel
        proj = self._project_for_channel(message.channel.id)
        if proj is not None and message.channel.id == proj.main_channel_id:
            await self._handle_main_message(proj, message)
            return

        # Otherwise, route to the agent that owns this channel
        agent = self._agent_for_channel(message.channel.id)
        if agent is None:
            return

        if agent.status in (Status.IDLE, Status.DONE, Status.ERROR):
            await message.add_reaction("\U0001f4e8")
            await agent.run_task_background(message.content)
        elif agent.status in (Status.RUNNING, Status.WAITING):
            await message.add_reaction("\U0001f4e8")
            await agent.send_input_background(message.content)


# ======================================================================
# Slash commands
# ======================================================================

HELP_TEXT = """\
**Hivemind** — Claude Code agent orchestrator

**Projects**
`/project create name:<str> [path:<str>]` — Create a project (category + `#main` channel). Use a preset name or provide a path.
`/project list` — List all projects
`/project delete name:<str>` — Delete project, kill its agents, remove channels

**Agents** (run these from a project's `#main` channel, or pass `project:` explicitly)
`/spawn name:<str> [persona:<choice>] [extra_instr:<str>] [task:<str>] [session_id:<str>] [cont:True]` — Spawn a new agent. Pick a `persona` from the dropdown, or use `extra_instr` for a custom role. Pass `session_id` to resume a specific session, or `cont:True` to continue the most recent one.
`/task name:<str> task:<str>` — Assign a task to an existing agent
`/kill name:<str>` — Stop an agent
`/broadcast message:<str>` — Send a message to all active agents

**Info**
`/status` — Dashboard showing all projects and agents
`/sessions [project:<str>]` — List Claude Code sessions for a project (use in `#main`)
`/help` — This message

**In `#main` (collaboration)**
Type a message — it's broadcast to all agents. Use `@agent_name` to direct specific agents; @mentioned agents must respond, others may chime in if relevant.

**In agent channels**
Just type a message — it gets forwarded to the agent as input. If the agent is idle/done, it starts a new task. If it's running, it's sent as follow-up input.
"""


def _register_commands(bot: HivemindBot) -> None:
    tree = bot.tree

    # --- /help ---
    @tree.command(name="help", description="Show Hivemind command reference")
    async def help_cmd(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(HELP_TEXT, ephemeral=True)

    # --- /project ---
    project_group = app_commands.Group(name="project", description="Manage projects")

    @project_group.command(name="create", description="Create a new project")
    @app_commands.describe(
        name="Project name (or preset name from config.yaml)",
        path="Directory path (optional if using a preset)",
        system_prompt="System prompt for agents in this project",
    )
    async def project_create(
        interaction: discord.Interaction,
        name: str,
        path: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        await interaction.response.defer()
        guild = interaction.guild
        if guild is None:
            await interaction.followup.send("Must be used in a server.")
            return

        if name in bot.projects:
            await interaction.followup.send(f"Project **{name}** already exists.")
            return

        preset = bot.config.project_presets.get(name)
        if preset and path is None:
            proj_path = preset.path
            prompt = system_prompt or preset.system_prompt
            tools = preset.allowed_tools
        elif path is not None:
            proj_path = os.path.expanduser(path)
            prompt = system_prompt or ""
            tools = DEFAULT_TOOLS[:]
        else:
            await interaction.followup.send(
                f"Unknown preset **{name}**. Provide a `path` or use a preset name."
            )
            return

        os.makedirs(proj_path, exist_ok=True)

        category = await guild.create_category(name.upper())
        main_channel = await guild.create_text_channel("main", category=category)

        project = Project(
            name=name,
            path=proj_path,
            category_id=category.id,
            main_channel_id=main_channel.id,
            system_prompt=prompt,
            allowed_tools=tools,
        )
        bot.projects[name] = project
        bot._save_state()

        await interaction.followup.send(
            f"Project **{name}** created → `{proj_path}`\n"
            f"Use `/spawn name:<agent>` in {main_channel.mention} to add agents."
        )

    @project_group.command(name="list", description="List all projects")
    async def project_list(interaction: discord.Interaction) -> None:
        if not bot.projects:
            await interaction.response.send_message("No projects.")
            return

        lines: list[str] = []
        for name, proj in bot.projects.items():
            agent_count = len(proj.agents)
            running = sum(1 for a in proj.agents.values() if a.status == Status.RUNNING)
            lines.append(
                f"**{name}** — `{proj.path}` — "
                f"{agent_count} agent{'s' if agent_count != 1 else ''}"
                f"{f' ({running} running)' if running else ''}"
            )
        await interaction.response.send_message("\n".join(lines))

    @project_group.command(name="delete", description="Delete a project and kill its agents")
    @app_commands.describe(name="Project name")
    async def project_delete(interaction: discord.Interaction, name: str) -> None:
        await interaction.response.defer()
        proj = bot.projects.get(name)
        if proj is None:
            await interaction.followup.send(f"No project **{name}**.")
            return

        for agent_name, agent in list(proj.agents.items()):
            consumer = bot._consumer_tasks.pop(agent.full_name, None)
            if consumer:
                consumer.cancel()
            await agent.stop()

        del bot.projects[name]
        bot._save_state()

        guild = interaction.guild
        if guild:
            category = guild.get_channel(proj.category_id)
            if category and isinstance(category, discord.CategoryChannel):
                for ch in category.channels:
                    await ch.delete()
                await category.delete()

        await interaction.followup.send(f"Project **{name}** deleted.")

    tree.add_command(project_group)

    # --- /spawn ---
    @tree.command(name="spawn", description="Spawn a new agent in a project")
    @app_commands.describe(
        name="Agent name",
        project="Project name (auto-detected if run from a project channel)",
        task="Initial task to run",
        persona="Predefined persona (shapes agent behavior and identity)",
        extra_instr="Extra instructions to append to the persona or use as a custom role",
        session_id="Resume a specific Claude Code session by ID",
        cont="Continue the most recent Claude Code session in this project dir",
    )
    @app_commands.choices(persona=[
        app_commands.Choice(name="dev/python", value="dev/python"),
        app_commands.Choice(name="dev/cpp", value="dev/cpp"),
        app_commands.Choice(name="qa", value="qa"),
        app_commands.Choice(name="pm", value="pm"),
        app_commands.Choice(name="architect", value="architect"),
    ])
    async def spawn(
        interaction: discord.Interaction,
        name: str,
        project: str | None = None,
        task: str | None = None,
        persona: app_commands.Choice[str] | None = None,
        extra_instr: str | None = None,
        session_id: str | None = None,
        cont: bool = False,
    ) -> None:
        await interaction.response.defer()
        guild = interaction.guild
        if guild is None:
            await interaction.followup.send("Must be used in a server.")
            return

        proj = bot._resolve_project(interaction, project)
        if proj is None:
            await interaction.followup.send(
                "Could not determine project. Run this from a project's `#main` channel "
                "or pass `project:<name>`."
            )
            return

        if name in proj.agents:
            await interaction.followup.send(
                f"Agent **{name}** already exists in project **{proj.name}**."
            )
            return

        category = guild.get_channel(proj.category_id)
        if category is None or not isinstance(category, discord.CategoryChannel):
            await interaction.followup.send("Project category not found.")
            return

        channel = await guild.create_text_channel(name, category=category)

        persona_key = persona.value if persona else ""
        agent = Agent(
            name=name,
            project_name=proj.name,
            project_path=proj.path,
            channel_id=channel.id,
            system_prompt="",  # set below
            allowed_tools=proj.allowed_tools,
            role=extra_instr or "",
            persona=persona_key,
        )
        agent.system_prompt = bot._build_agent_system_prompt(proj, agent)

        try:
            await agent.start(
                resume_session=session_id,
                continue_conversation=cont and session_id is None,
                get_peers=lambda pn=proj.name: bot._get_peers_for_project(pn),
            )
        except Exception as exc:
            await interaction.followup.send(f"Failed to start agent: {exc}")
            await channel.delete()
            return

        proj.agents[name] = agent
        bot._save_state()

        bot._consumer_tasks[agent.full_name] = asyncio.create_task(
            consume_events(agent, channel),
            name=f"consumer-{agent.full_name}",
        )

        if session_id:
            label = f"spawned (resuming session `{session_id[:12]}…`)"
        elif cont:
            label = "spawned (continuing last session)"
        else:
            label = "spawned"
        msg_parts = [f"Agent **{name}** {label} in {channel.mention}"]
        if persona:
            msg_parts.append(f"Persona: **{persona.value}**")
        await interaction.followup.send("\n".join(msg_parts))

        if task:
            await agent.run_task_background(task)

    # --- /task ---
    @tree.command(name="task", description="Assign a task to an agent")
    @app_commands.describe(
        name="Agent name",
        task="Task description",
        project="Project name (auto-detected from channel)",
    )
    async def task_cmd(
        interaction: discord.Interaction,
        name: str,
        task: str,
        project: str | None = None,
    ) -> None:
        await interaction.response.defer()
        proj = bot._resolve_project(interaction, project)
        if proj is None:
            await interaction.followup.send("Could not determine project.")
            return
        agent = proj.agents.get(name)
        if agent is None:
            await interaction.followup.send(f"No agent **{name}** in project **{proj.name}**.")
            return
        await agent.run_task_background(task)
        await interaction.followup.send(
            f"Task assigned to **{name}** in <#{agent.channel_id}>"
        )

    # --- /kill ---
    @tree.command(name="kill", description="Stop an agent")
    @app_commands.describe(
        name="Agent name",
        project="Project name (auto-detected from channel)",
    )
    async def kill(
        interaction: discord.Interaction,
        name: str,
        project: str | None = None,
    ) -> None:
        await interaction.response.defer()
        proj = bot._resolve_project(interaction, project)
        if proj is None:
            await interaction.followup.send("Could not determine project.")
            return
        agent = proj.agents.get(name)
        if agent is None:
            await interaction.followup.send(f"No agent **{name}** in project **{proj.name}**.")
            return

        consumer = bot._consumer_tasks.pop(agent.full_name, None)
        if consumer:
            consumer.cancel()
        await agent.stop()
        del proj.agents[name]
        bot._save_state()
        await interaction.followup.send(f"Agent **{proj.name}/{name}** killed.")

    # --- /status ---
    @tree.command(name="status", description="Show status of all agents")
    async def status_cmd(interaction: discord.Interaction) -> None:
        embed = status_embed(bot.projects)
        await interaction.response.send_message(embed=embed)

    # --- /broadcast ---
    @tree.command(name="broadcast", description="Send a message to all running agents")
    @app_commands.describe(message="Message to broadcast")
    async def broadcast(interaction: discord.Interaction, message: str) -> None:
        await interaction.response.defer()
        active: list[Agent] = []
        for proj in bot.projects.values():
            for agent in proj.agents.values():
                if agent.status in (Status.RUNNING, Status.WAITING, Status.IDLE):
                    active.append(agent)

        if not active:
            await interaction.followup.send("No active agents.")
            return
        for agent in active:
            await agent.send_input_background(message)
        names = ", ".join(a.full_name for a in active)
        await interaction.followup.send(f"Broadcast sent to: {names}")

    # --- /sessions ---
    @tree.command(name="sessions", description="List Claude Code sessions for a project")
    @app_commands.describe(
        project="Project name (auto-detected from channel)",
    )
    async def sessions_cmd(
        interaction: discord.Interaction,
        project: str | None = None,
    ) -> None:
        proj = bot._resolve_project(interaction, project)
        if proj is None:
            await interaction.response.send_message(
                "Could not determine project. Run this from a project channel "
                "or pass `project:<name>`.",
                ephemeral=True,
            )
            return

        sessions = list_sessions(proj.path)
        if not sessions:
            await interaction.response.send_message(
                f"No sessions found for **{proj.name}** (`{proj.path}`).",
                ephemeral=True,
            )
            return

        lines: list[str] = [f"**Sessions for {proj.name}** (`{proj.path}`)\n"]
        for s in sessions:
            ts = s.timestamp[:16].replace("T", " ") if s.timestamp else "?"
            sid = s.session_id[:12]
            lines.append(f"`{sid}…` | {ts} | {s.task}")

        text = "\n".join(lines)
        if len(text) > 1900:
            text = text[:1900] + "\n…"
        await interaction.response.send_message(text, ephemeral=True)


# ======================================================================
# Entry point
# ======================================================================

def run_bot(config_path: str = "config.yaml") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    )

    config = Config.load(config_path)
    if not config.bot_token:
        raise RuntimeError(
            "DISCORD_BOT_TOKEN not set. Put it in .env or config.yaml"
        )

    bot = HivemindBot(config)
    bot.run(config.bot_token, log_handler=None)
