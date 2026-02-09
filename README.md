# Hivemind — Discord Agent Orchestrator for Claude Code

Orchestrate multiple Claude Code agents from Discord. Each agent runs in its own persistent `claude-code-sdk` session, reports progress to a dedicated channel, and can ask humans for input via Discord buttons.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Node.js and Claude Code CLI: `npm install -g @anthropic-ai/claude-code`
- A Discord bot token with Message Content intent enabled

## Setup

```bash
cp .env.example .env
# Edit .env with your tokens

uv sync
```

## Discord Server Structure

Create these channels manually (the bot creates agent channels automatically):

```
COMMAND CENTER
  #commander     — run slash commands here
  #approvals     — aggregated permission request pings

AGENTS
  (auto-created by /spawn)
```

## Run

```bash
make run
# or
uv run python -m hivemind
```

## Slash Commands

| Command | Description |
|---------|-------------|
| `/spawn name:<str> [project:<str>] [task:<str>] [tools:<str>]` | Spawn agent (use preset name or provide project path) |
| `/task name:<str> task:<str>` | Assign a new task to an existing agent |
| `/status` | Show status dashboard of all agents |
| `/kill name:<str>` | Stop an agent and clean up |
| `/broadcast message:<str>` | Send a message to all active agents |

## Agent Presets

Configure presets in `config.yaml`:

```yaml
agent_presets:
  my-project:
    project: ~/my-project
    system_prompt: "You are working on..."
    allowed_tools: [Read, Write, Edit, Bash, Glob, Grep, mcp__human__ask_human]
    max_turns: 100
```

Then spawn with just: `/spawn my-project`

## How It Works

1. `/spawn` creates a `#agent-{name}` channel and a `ClaudeSDKClient` session
2. The agent streams events (progress, tool calls, results) into the channel
3. The agent can call `ask_human` (MCP tool) to post interactive buttons to Discord
4. Humans respond via buttons or a free-form modal
5. Messages typed in an agent's channel are forwarded as input to that agent
6. `/status` shows a dashboard of all agents with status and cost
