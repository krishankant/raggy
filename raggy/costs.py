"""What a run cost (week 12): tokens × prices, per million tokens.

There's no built-in price list: prices change, and a stale number printed as a
fact is worse than none. Set them for the models you use:

    RAGGY_PRICES="openai:gpt-5.5=1.25/10/0.125, anthropic:claude-opus-5-5=5/25/0.5"

Each entry is ``provider:model=input/output[/cached input]``. Without a cached
price, cached tokens are charged at the input rate.
"""

from __future__ import annotations

from raggy.config import ConfigError


def parse_prices(raw: str | None) -> dict[str, tuple[float, float, float | None]]:
    prices = {}
    for entry in filter(None, (e.strip() for e in (raw or "").split(","))):
        model, _, rates = entry.partition("=")
        try:
            parts = [float(x) for x in rates.split("/")]
            if len(parts) not in (2, 3) or not model.strip():
                raise ValueError
        except ValueError:
            raise ConfigError(f"RAGGY_PRICES: can't read {entry!r} (use provider:model=input/output[/cached])") from None
        prices[model.strip()] = (parts[0], parts[1], parts[2] if len(parts) == 3 else None)
    return prices


def cost(usage: dict, model: str, prices: dict) -> float | None:
    """Dollars (or whatever unit the prices use) for this usage; None if the model has no price."""
    if model not in prices or not usage:
        return None
    rate_in, rate_out, rate_cached = prices[model]
    cached = usage.get("cache_read", 0)
    fresh = usage.get("input", 0) - cached
    return (fresh * rate_in + cached * (rate_cached if rate_cached is not None else rate_in)
            + usage.get("cache_write", 0) * rate_in + usage.get("output", 0) * rate_out) / 1_000_000
