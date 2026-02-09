"""Discord UI components — approval buttons, modals, status embeds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import discord

if TYPE_CHECKING:
    from .bot import Project


class ReplyModal(discord.ui.Modal, title="Reply to Agent"):
    """Free-form text reply modal."""

    answer = discord.ui.TextInput(
        label="Your response",
        style=discord.TextStyle.paragraph,
        placeholder="Type your response here…",
        required=True,
        max_length=2000,
    )

    def __init__(self, request_id: str, bridge, agent) -> None:
        super().__init__()
        self.request_id = request_id
        self.bridge = bridge
        self.agent = agent

    async def on_submit(self, interaction: discord.Interaction) -> None:
        text = self.answer.value
        self.bridge.resolve(self.request_id, text)
        await interaction.response.send_message(
            f"Sent reply: {text[:100]}{'…' if len(text) > 100 else ''}",
            ephemeral=True,
        )


class ApprovalView(discord.ui.View):
    """Buttons for approving / replying to an agent question."""

    def __init__(
        self,
        request_id: str,
        options: list[str],
        bridge,
        agent,
        *,
        timeout: float = 600,
    ) -> None:
        super().__init__(timeout=timeout)
        self.request_id = request_id
        self.bridge = bridge
        self.agent = agent

        for i, option in enumerate(options[:4]):
            btn = discord.ui.Button(
                label=option[:80],
                style=discord.ButtonStyle.primary,
                custom_id=f"approval_{request_id}_{i}",
            )
            btn.callback = self._make_option_callback(option)
            self.add_item(btn)

        custom_btn = discord.ui.Button(
            label="\U0001f4ac Custom Reply",
            style=discord.ButtonStyle.secondary,
            custom_id=f"approval_{request_id}_custom",
        )
        custom_btn.callback = self._custom_callback
        self.add_item(custom_btn)

    def _make_option_callback(self, option: str):
        async def callback(interaction: discord.Interaction) -> None:
            self.bridge.resolve(self.request_id, option)
            for item in self.children:
                if isinstance(item, discord.ui.Button):
                    item.disabled = True
            await interaction.response.edit_message(
                content=f"Selected: **{option}**", view=self
            )

        return callback

    async def _custom_callback(self, interaction: discord.Interaction) -> None:
        modal = ReplyModal(self.request_id, self.bridge, self.agent)
        await interaction.response.send_modal(modal)


# ---------------------------------------------------------------------------
# Embed helpers
# ---------------------------------------------------------------------------

STATUS_EMOJI = {
    "idle": "\u26aa",
    "running": "\U0001f7e2",
    "waiting": "\U0001f7e1",
    "done": "\u2705",
    "error": "\U0001f534",
}


def status_embed(projects: dict[str, Project]) -> discord.Embed:
    """Build a status overview embed for /status."""
    embed = discord.Embed(title="Hivemind Dashboard", color=0x5865F2)
    if not projects:
        embed.description = "No projects."
        return embed

    for proj_name, proj in projects.items():
        if not proj.agents:
            embed.add_field(
                name=f"\U0001f4c1 {proj_name}",
                value=f"`{proj.path}`\nNo agents",
                inline=False,
            )
            continue

        agent_lines: list[str] = []
        for agent in proj.agents.values():
            emoji = STATUS_EMOJI.get(agent.status.value, "\u2753")
            cost = f"${agent._total_cost:.4f}" if agent._total_cost else ""
            agent_lines.append(
                f"{emoji} **{agent.name}** <#{agent.channel_id}>"
                f"{f' — {cost}' if cost else ''}"
            )

        embed.add_field(
            name=f"\U0001f4c1 {proj_name}",
            value=f"`{proj.path}`\n" + "\n".join(agent_lines),
            inline=False,
        )
    return embed


def event_embed(kind: str, text: str, agent_name: str = "") -> discord.Embed:
    """Build an embed for an agent event."""
    colors = {
        "question": 0xF39C12,
        "error": 0xE74C3C,
    }
    embed = discord.Embed(
        description=text[:4096] if text else None,
        color=colors.get(kind, 0x95A5A6),
    )
    if agent_name:
        embed.set_footer(text=agent_name)
    return embed
