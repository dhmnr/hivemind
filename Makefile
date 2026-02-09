.PHONY: run install lint sync

run:
	uv run python -m hivemind

sync:
	uv sync

install: sync

lint:
	uv run python -m py_compile src/hivemind/config.py
	uv run python -m py_compile src/hivemind/tools.py
	uv run python -m py_compile src/hivemind/agent.py
	uv run python -m py_compile src/hivemind/views.py
	uv run python -m py_compile src/hivemind/event_consumer.py
	uv run python -m py_compile src/hivemind/bot.py
	@echo "All files compile OK"
