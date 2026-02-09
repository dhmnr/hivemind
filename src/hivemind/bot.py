"""Discord bot — slash commands, event handlers, and lifecycle management.

Organized around Projects (Discord category + directory) containing
independent Agents (each with its own channel and ClaudeSDKClient session).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

import discord
from discord import app_commands

from .agent import Agent, Status
from .config import Config
from .event_consumer import consume_approval_requests, consume_events
from .sessions import list_sessions
from .views import status_embed

log = logging.getLogger(__name__)

DEFAULT_TOOLS = [
    "Read", "Write", "Edit", "Bash", "Glob", "Grep",
    "mcp__human__ask_human",
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_agents(self) -> dict[str, Agent]:
        out: dict[str, Agent] = {}
        for proj in self.projects.values():
            for agent in proj.agents.values():
                out[agent.full_name] = agent
        return out

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
        self._approval_task = asyncio.create_task(
            consume_approval_requests(self, self),
            name="approval-consumer",
        )

    # ------------------------------------------------------------------
    # Channel messages → agent input
    # ------------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

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
`/spawn name:<str> [task:<str>] [session_id:<str>] [cont:True]` — Spawn a new agent. Pass `session_id` to resume a specific session, or `cont:True` to continue the most recent one.
`/task name:<str> task:<str>` — Assign a task to an existing agent
`/kill name:<str>` — Stop an agent
`/broadcast message:<str>` — Send a message to all active agents

**Info**
`/status` — Dashboard showing all projects and agents
`/sessions [project:<str>]` — List Claude Code sessions for a project (use in `#main`)
`/help` — This message

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
        session_id="Resume a specific Claude Code session by ID",
        cont="Continue the most recent Claude Code session in this project dir",
    )
    async def spawn(
        interaction: discord.Interaction,
        name: str,
        project: str | None = None,
        task: str | None = None,
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

        agent = Agent(
            name=name,
            project_name=proj.name,
            project_path=proj.path,
            channel_id=channel.id,
            system_prompt=proj.system_prompt,
            allowed_tools=proj.allowed_tools,
        )

        try:
            await agent.start(
                resume_session=session_id,
                continue_conversation=cont and session_id is None,
            )
        except Exception as exc:
            await interaction.followup.send(f"Failed to start agent: {exc}")
            await channel.delete()
            return

        proj.agents[name] = agent

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
        await interaction.followup.send(
            f"Agent **{name}** {label} in {channel.mention}"
        )

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
