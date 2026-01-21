from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MergeResult:
    ok: bool
    tx_hash: str | None
    reason: str


def merge_full_set(
    market_id: str,
    up_token_id: str,
    down_token_id: str,
    shares: float,
    enabled: bool = False,
    **_kwargs: Any,
) -> MergeResult:
    """
    Optional on-chain merge stub (OFF by default).

    TODO: Integrate on-chain merge/redeem for matched full sets.
    - Validate wallet and chain config
    - Build and send merge transaction
    - Track tx hash and receipts
    """
    if not enabled:
        return MergeResult(False, None, "merge_disabled")
    return MergeResult(False, None, "merge_not_implemented")
