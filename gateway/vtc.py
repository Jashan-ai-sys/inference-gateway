"""Virtual Token Counter fairness (Sheng et al., "Fairness in Serving LLMs", OSDI '24).

Each tenant has a counter of weighted tokens it has been served. The backlogged tenant with the
lowest counter goes next, so a tenant sending more requests does not get more service."""

from collections.abc import Iterable


class VTC:
    def __init__(self, prompt_weight: float = 1.0, output_weight: float = 2.0) -> None:
        self.prompt_weight = prompt_weight
        self.output_weight = output_weight
        self._counters: dict[str, float] = {}

    def counter(self, tenant: str) -> float:
        return self._counters.get(tenant, 0.0)

    def on_backlogged(self, tenant: str, backlogged: Iterable[str]) -> None:
        """Lift a tenant that just became backlogged to the lowest counter among the others, so
        credit saved while idle cannot be spent starving tenants that kept working."""
        others = [self.counter(t) for t in backlogged if t != tenant]
        if others:
            self._counters[tenant] = max(self.counter(tenant), min(others))

    def charge_prompt(self, tenant: str, tokens: int) -> None:
        self._counters[tenant] = self.counter(tenant) + self.prompt_weight * tokens

    def charge_output(self, tenant: str, tokens: int) -> None:
        self._counters[tenant] = self.counter(tenant) + self.output_weight * tokens

    def pick(self, backlogged: Iterable[str]) -> str | None:
        candidates = list(backlogged)
        if not candidates:
            return None
        return min(candidates, key=lambda t: (self.counter(t), t))
