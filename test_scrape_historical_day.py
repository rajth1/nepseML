"""
Test the scrape logic against a KNOWN PAST TRADING DAY, bypassing the
market-open check, since we can't force NEPSE to be open on demand.

This reuses the exact same fetch/upsert/logging functions as
daily_scrape.py — the only difference is which date we ask for, and that
we skip the "is the market open" question since we already know the
answer for a day in the past.

Usage:
    python test_scrape_historical_day.py                # defaults to 2026-08-07
    python test_scrape_historical_day.py --date 2026-08-06
"""

import argparse
import os
import sys
from datetime import date

import psycopg2
from nepse import Nepse

from daily_scrape import (
    get_active_bank_tickers,
    fetch_today_price,
    upsert_company,
    upsert_daily_price,
    log_run,
)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is not set.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-08-07",
                         help="A known past trading day, YYYY-MM-DD")
    args = parser.parse_args()
    test_date = date.fromisoformat(args.date)

    print(f"=== TEST RUN: pretending today is {test_date} (market-open check skipped) ===")

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

            row = fetch_today_price(client, symbol, test_date)
            if row is None:
                print(f"    {symbol}: no data returned for {test_date}.")
                failed_symbols.append(symbol)
                continue

            upsert_daily_price(cur, symbol, test_date, row)
            succeeded_symbols.append(symbol)
            print(f"    {symbol}: OK — close={row.get('closePrice')}")

        except Exception as e:
            print(f"    {symbol}: unexpected error, skipping: {e}")
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
    print("Check the companies, daily_prices, and scrape_log tables in your")
    print("database dashboard to confirm rows actually landed correctly.")


if __name__ == "__main__":
    main()
