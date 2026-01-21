#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType


def main() -> int:
    load_dotenv()

    private_key = os.getenv("POLY_PRIVATE_KEY")
    if not private_key:
        print("error: POLY_PRIVATE_KEY is not set", file=sys.stderr)
        return 2

    chain_id = int(os.getenv("POLY_CHAIN_ID", "137"))
    signature_type = int(os.getenv("POLY_SIGNATURE_TYPE", "0"))
    funder = os.getenv("POLY_FUNDER") or None
    host = os.getenv("CLOB_HOST", "https://clob.polymarket.com")

    try:
        client = ClobClient(
            host,
            key=private_key,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder,
        )
        client.set_api_creds(client.create_or_derive_api_creds())

        params = BalanceAllowanceParams()
        params.asset_type = AssetType.COLLATERAL
        resp = client.get_balance_allowance(params)
    except Exception as exc:
        print(f"error: failed to fetch balance: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(resp, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
