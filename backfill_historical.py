"""
One-time historical backfill: pull 1 year of daily data for all 19 active
commercial bank tickers, and load it into daily_prices.

Reuses the same upsert/company-list/log functions as daily_scrape.py —
this is deliberately NOT a separate code path, just a wider date range.

Usage:
    python backfill_historical.py               # defaults to 1 year back from today
    python backfill_historical.py --days 365
"""

import argparse
import os
import sys
import time
from datetime import date, timedelta

import psycopg2
from nepse import Nepse

from daily_scrape import get_active_bank_tickers, upsert_company, upsert_daily_price, log_run

MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 3

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is not set.")
    sys.exit(1)


def fetch_history_range(client, symbol, start_date, end_date):
    """
    Fetch a date range for one ticker, with retries. Returns the list of
    daily rows (each a dict with businessDate/highPrice/lowPrice/etc).

    Note: getCompanyPriceVolumeHistory pages at 500 rows internally. A
    1-year window is ~250 trading days, comfortably under that, so we
    expect a single page. We still check totalPages and warn if NEPSE
    ever returns more than one page here, since the wrapper we're using
    doesn't expose a way to request page 2+ directly — if this ever
    triggers, the date range needs to be split into smaller chunks.
    """
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = client.getCompanyPriceVolumeHistory(symbol, start_date=start_date, end_date=end_date)
            if not isinstance(result, dict):
                return []
            total_pages = result.get("totalPages", 1)
            if total_pages > 1:
                print(f"    WARNING: {symbol} has {total_pages} pages of data in this range — "
                      f"only page 1 is being fetched. Consider narrowing the date range.")
            return result.get("content", [])
        except Exception as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                print(f"    {symbol}: attempt {attempt} failed ({e}), retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
    print(f"    {symbol}: FAILED after {MAX_ATTEMPTS} attempts: {last_error}")
    raise last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365,
                         help="How many days back to backfill (default 365)")
    args = parser.parse_args()

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)
    print(f"=== Backfilling from {start_date} to {end_date} ===")

    client = Nepse()
    client.setTLSVerification(False)

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    print("Fetching active commercial bank ticker list...")
    banks = get_active_bank_tickers(client)
    print(f"Found {len(banks)} active commercial bank tickers.")

    security_id_map = client.getSecurityIDKeyMap()

    succeeded_symbols = []
    failed_symbols = []

    for company in banks:
        symbol = company["symbol"]
        print(f"  Processing {symbol}...")
        try:
            upsert_company(cur, company, security_id_map.get(symbol))

            rows = fetch_history_range(client, symbol, start_date, end_date)
            if not rows:
                print(f"    {symbol}: no historical rows returned.")
                failed_symbols.append(symbol)
                continue

            for row in rows:
                business_date = date.fromisoformat(row["businessDate"])
                upsert_daily_price(cur, symbol, business_date, row)

            conn.commit()
            succeeded_symbols.append(symbol)
            print(f"    {symbol}: inserted {len(rows)} days.")

        except Exception as e:
            print(f"    {symbol}: unexpected error, skipping this ticker: {e}")
            failed_symbols.append(symbol)
            conn.rollback()
            cur = conn.cursor()

    attempted = len(banks)
    succeeded = len(succeeded_symbols)
    failed = len(failed_symbols)
    status = "success" if failed == 0 else ("partial_failure" if succeeded > 0 else "failure")

    log_run(cur, status, attempted, succeeded, failed, failed_symbols)
    conn.commit()
    cur.close()
    conn.close()

    print(f"\n=== Done. status={status}, succeeded={succeeded}/{attempted} ===")


if __name__ == "__main__":
    main()
