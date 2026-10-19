"""Remembers which backend served each conversation, for the model-switch cost (spec §4.3)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class SessionState:
    backend: str
    tokens: int  # prompt + output of the last turn: the prefix that backend may still have cached


class SessionStore:
    def __init__(self, max_sessions: int = 10_000) -> None:
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[tuple[str, str], SessionState] = OrderedDict()

    def get(self, tenant: str, session_id: str) -> SessionState | None:
        return self._sessions.get((tenant, session_id))

    def update(self, tenant: str, session_id: str, backend: str, tokens: int) -> None:
        key = (tenant, session_id)
        self._sessions[key] = SessionState(backend, tokens)
        self._sessions.move_to_end(key)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
