"""Process entry point: `GW_CONFIG=config/sim.yaml python -m gateway.main`."""

from __future__ import annotations

import logging
import os

import httpx
import uvicorn

from gateway.analyzer import approx_tokens, qwen_token_counter
from gateway.api import create_app
from gateway.config import apply_env_overrides, load_config
from gateway.engine import Engine


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = apply_env_overrides(load_config(os.environ.get("GW_CONFIG", "config/sim.yaml")))
    count_tokens = (
        qwen_token_counter() if os.environ.get("GW_TOKENIZER") == "qwen" else approx_tokens
    )
    decision_log = open(cfg.decision_log, "a") if cfg.decision_log else None
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=5.0),
        limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
    )
    engine = Engine(cfg, client, count_tokens=count_tokens, decision_log=decision_log)
    uvicorn.run(
        create_app(cfg, engine),
        host=os.environ.get("GW_HOST", "127.0.0.1"),
        port=int(os.environ.get("GW_PORT", "8080")),
    )


if __name__ == "__main__":
    main()
