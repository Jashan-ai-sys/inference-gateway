.PHONY: install fmt lint test sim-up sim-down integration run

install: ; uv sync
fmt: ; uv run ruff format . && uv run ruff check --fix .
lint: ; uv run ruff check . && uv run ruff format --check .
test: ; uv run pytest
sim-up: ; docker compose -f docker-compose.sim.yml up -d
sim-down: ; docker compose -f docker-compose.sim.yml down
integration: ; uv run pytest -m integration
run: ; GW_CONFIG=config/sim.yaml uv run python -m gateway.main
