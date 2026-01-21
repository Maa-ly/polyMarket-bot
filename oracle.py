from __future__ import annotations

from typing import Any


def get_probability_up(market: dict[str, Any]) -> float | None:
    """
    External oracle hook. Return a probability for "Up" in [0, 1], or None.
    """
    _ = market
    return None
