#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE_URL = "https://data-api.polymarket.com"
DEFAULT_LIMIT = 50
MAX_OFFSET = 100000


def _normalize_ts(ts):
    if ts is None:
        return None
    # Heuristic: treat millisecond timestamps as seconds.
    if ts > 10**12:
        return int(ts / 1000)
    return int(ts)


def _iso(ts):
    ts = _normalize_ts(ts)
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _get_json(path, params):
    url = BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"User-Agent": "polymarket-win-analyzer"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} for {url}")
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def fetch_closed_positions(user, limit, start_ts=None, max_pages=None, max_rows=None):
    if limit > 50 or limit <= 0:
        raise ValueError("limit must be between 1 and 50")
    offset = 0
    page = 0
    results = []
    while True:
        if max_pages is not None and page >= max_pages:
            break
        params = {
            "user": user,
            "limit": limit,
            "offset": offset,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
        }
        batch = _get_json("/closed-positions", params)
        if not batch:
            break
        results.extend(batch)
        if max_rows is not None and len(results) >= max_rows:
            results = results[:max_rows]
            break
        if start_ts is not None:
            last_ts = _normalize_ts(batch[-1].get("timestamp"))
            if last_ts is not None and last_ts < start_ts:
                break
        if len(batch) < limit:
            break
        offset += limit
        page += 1
        if offset > MAX_OFFSET:
            break
    return results


def filter_by_time(rows, start_ts, end_ts):
    if start_ts is None and end_ts is None:
        return rows
    filtered = []
    for row in rows:
        ts = _normalize_ts(row.get("timestamp"))
        if ts is None:
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        filtered.append(row)
    return filtered


def realized_pnl(row):
    pnl = row.get("realizedPnl")
    return float(pnl) if pnl is not None else 0.0


def roi(row):
    pnl = realized_pnl(row)
    total = row.get("totalBought")
    if total in (None, 0):
        return None
    return pnl / float(total)


def group_stats(rows, key):
    groups = {}
    for row in rows:
        val = row.get(key) or "UNKNOWN"
        groups.setdefault(val, {"count": 0, "pnl": 0.0, "roi_sum": 0.0, "roi_n": 0})
        groups[val]["count"] += 1
        groups[val]["pnl"] += realized_pnl(row)
        r = roi(row)
        if r is not None:
            groups[val]["roi_sum"] += r
            groups[val]["roi_n"] += 1
    return groups


def print_group_table(title, groups, limit=10):
    print(f"\n{title}")
    items = sorted(groups.items(), key=lambda kv: kv[1]["pnl"], reverse=True)[:limit]
    for name, stats in items:
        avg_roi = stats["roi_sum"] / stats["roi_n"] if stats["roi_n"] else 0.0
        print(f"- {name}: count={stats['count']} pnl={stats['pnl']:.4f} avg_roi={avg_roi:.4f}")


def price_band_stats(rows):
    bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    stats = {}
    for low, high in bands:
        key = f"{low:.1f}-{high:.1f}"
        stats[key] = {"count": 0, "wins": 0}
    for row in rows:
        p = row.get("avgPrice")
        if p is None:
            continue
        for low, high in bands:
            if low <= float(p) < high:
                key = f"{low:.1f}-{high:.1f}"
                stats[key]["count"] += 1
                if realized_pnl(row) > 0:
                    stats[key]["wins"] += 1
                break
    return stats


def write_positions_csv(rows, out_path):
    headers = [
        "timestamp",
        "timestamp_iso",
        "conditionId",
        "title",
        "eventSlug",
        "outcome",
        "avgPrice",
        "totalBought",
        "realizedPnl",
        "roi",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            ts = _normalize_ts(row.get("timestamp"))
            writer.writerow(
                {
                    "timestamp": ts,
                    "timestamp_iso": _iso(ts),
                    "conditionId": row.get("conditionId", ""),
                    "title": row.get("title", ""),
                    "eventSlug": row.get("eventSlug", ""),
                    "outcome": row.get("outcome", ""),
                    "avgPrice": row.get("avgPrice", ""),
                    "totalBought": row.get("totalBought", ""),
                    "realizedPnl": row.get("realizedPnl", ""),
                    "roi": "" if roi(row) is None else f"{roi(row):.6f}",
                }
            )


def summarize(all_rows, wins):
    total = len(all_rows)
    win_count = len(wins)
    win_rate = (win_count / total) if total else 0.0
    pnl_total = sum(realized_pnl(r) for r in all_rows)
    pnl_wins = sum(realized_pnl(r) for r in wins)
    pnl_losses = pnl_total - pnl_wins
    print("Summary")
    print(f"- total_closed_positions: {total}")
    print(f"- winning_positions: {win_count}")
    print(f"- win_rate: {win_rate:.3f}")
    print(f"- total_realized_pnl: {pnl_total:.4f}")
    print(f"- realized_pnl_wins: {pnl_wins:.4f}")
    print(f"- realized_pnl_nonwins: {pnl_losses:.4f}")

    wins_roi = [roi(r) for r in wins if roi(r) is not None]
    if wins_roi:
        avg_roi = sum(wins_roi) / len(wins_roi)
        print(f"- avg_roi_wins: {avg_roi:.4f}")

    print_group_table("Top winning markets (by pnl)", group_stats(wins, "title"))
    print_group_table("Top winning event slugs (by pnl)", group_stats(wins, "eventSlug"))
    print_group_table("Top winning outcomes (by pnl)", group_stats(wins, "outcome"))

    band_stats = price_band_stats(all_rows)
    print("\nWin rate by avgPrice band")
    for band in sorted(band_stats.keys()):
        count = band_stats[band]["count"]
        wins_count = band_stats[band]["wins"]
        rate = (wins_count / count) if count else 0.0
        print(f"- {band}: count={count} wins={wins_count} win_rate={rate:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket closed positions and extract winning trades."
    )
    parser.add_argument("--user", required=True, help="User wallet address (0x...)")
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Filter positions with timestamp >= start (unix seconds)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Filter positions with timestamp <= end (unix seconds)",
    )
    parser.add_argument(
        "--min-pnl",
        type=float,
        default=0.0,
        help="Minimum realized PnL to count as a win",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Page size for API requests (1-50)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to fetch (each page is --limit rows)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to fetch total",
    )
    parser.add_argument(
        "--out",
        default="winning_trades.csv",
        help="Output CSV for winning positions",
    )
    parser.add_argument(
        "--out-all",
        default=None,
        help="Output CSV for all closed positions (full history)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.user.startswith("0x") or len(args.user) != 42:
        print("error: --user must be a 0x-prefixed address", file=sys.stderr)
        return 2

    try:
        rows = fetch_closed_positions(
            args.user,
            args.limit,
            start_ts=args.start,
            max_pages=args.max_pages,
            max_rows=args.max_rows,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows = filter_by_time(rows, args.start, args.end)
    wins = [r for r in rows if realized_pnl(r) > args.min_pnl]

    write_positions_csv(wins, args.out)
    if args.out_all:
        write_positions_csv(rows, args.out_all)
    summarize(rows, wins)
    print(f"\nWrote {len(wins)} winning positions to {args.out}")
    if args.out_all:
        print(f"Wrote {len(rows)} total positions to {args.out_all}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
