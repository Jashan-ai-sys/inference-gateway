"""Token buckets that mirror a hosted provider's per-minute request and token limits."""

from __future__ import annotations


class TokenBucket:
    """Refills continuously at `per_minute` units per minute and holds at most one minute's
    worth."""

    def __init__(self, per_minute: float) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.level = float(per_minute)
        self.updated: float | None = None

    def _refill(self, now: float) -> None:
        if self.updated is not None:
            self.level = min(self.capacity, self.level + (now - self.updated) * self.rate)
        self.updated = now

    def available(self, now: float) -> float:
        self._refill(now)
        return self.level

    def take(self, amount: float, now: float) -> None:
        self._refill(now)
        self.level -= amount


class HostedLimiter:
    def __init__(self, rpm: int | None, tpm: int | None) -> None:
        self.requests = TokenBucket(rpm) if rpm else None
        self.tokens = TokenBucket(tpm) if tpm else None

    def has_room(self, tokens: int, now: float) -> bool:
        if self.requests and self.requests.available(now) < 1:
            return False
        return not (self.tokens and self.tokens.available(now) < tokens)

    def take(self, tokens: int, now: float) -> None:
        if self.requests:
            self.requests.take(1, now)
        if self.tokens:
            self.tokens.take(tokens, now)
