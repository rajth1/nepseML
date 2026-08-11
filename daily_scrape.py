"""
Daily NEPSE commercial bank price scrape.

What this script does, in plain terms:
  1. Always attempts to fetch today's data for all active bank tickers —
     it does NOT pre-emptively skip based on any "is the market open"
     status check. That check turned out to answer "is trading happening
     at this exact instant," not "was today a trading day" — since this
     job runs well after market close, that check would read "closed"
     on every single day regardless of whether today was a holiday or a
     completely normal trading day. It's still called and logged for
     visibility, but it no longer gates anything.
  2. The REAL verdict comes from what the 19 fetch attempts actually
     returned: if every ticker cleanly reports "no data" with no errors,
     that's a strong, data-driven signal it was a genuine non-trading
     day. If any real fetch errors occurred, that's treated as an actual
     failure, not a holiday.
  3. Save whatever data came back into Postgres.
  4. Write a one-line summary of what happened so we have a paper trail.

Retry behaviour: if a single bank's fetch fails, we try it up to 2 times
total before giving up on JUST that bank — the other banks still get
processed normally.
"""

import os
import sys
import time
from datetime import date

import psycopg2
from nepse import Nepse

MAX_ATTEMPTS_PER_TICKER = 2
RETRY_DELAY_SECONDS = 3

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is not set.")
    sys.exit(1)


def get_active_bank_tickers(client):
    """Same filter locked in during Phase 1: Commercial Banks + Equity + Active."""
    company_list = client.getCompanyList()
    return [
        c for c in company_list
        if c.get("sectorName") == "Commercial Banks"
        and c.get("instrumentType") == "Equity"
        and c.get("status") == "A"
    ]


def log_market_status_for_visibility(client):
    """
    Calls NEPSE's live market-status endpoint purely for logging — NOT as
    a gate for anything. This turned out to answer "is trading happening
    right now" rather than "was today a trading day," which is the wrong
    question for a job that runs after close. Kept as a printed diagnostic
    in case its actual behavior is worth revisiting later, but nothing in
    this script's control flow depends on it anymore.
    """
    try:
        status = client.isNepseOpen()
        print(f"  (For reference only, not used for the decision below: "
              f"isNepseOpen() returned {status!r})")
    except Exception as e:
        print(f"  (Market status check failed, ignored: {e})")


def fetch_today_price(client, symbol, today):
    """Fetch today's OHLCV-ish row for one ticker, with retries."""
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS_PER_TICKER + 1):
        try:
            result = client.getCompanyPriceVolumeHistory(symbol, start_date=today, end_date=today)
            rows = result.get("content", []) if isinstance(result, dict) else []
            if not rows:
                return None  # no data for today — could be a holiday, or a new/thin listing
            return rows[0]
        except Exception as e:
            last_error = e
            if attempt < MAX_ATTEMPTS_PER_TICKER:
                print(f"    {symbol}: attempt {attempt} failed ({e}), retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
    print(f"    {symbol}: FAILED after {MAX_ATTEMPTS_PER_TICKER} attempts: {last_error}")
    raise last_error


def upsert_company(cur, company, security_id):
    cur.execute(
        """
        INSERT INTO companies (symbol, security_id, company_name, sector_name,
                                instrument_type, status, last_synced_at)
        VALUES (%s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (symbol) DO UPDATE SET
            security_id = EXCLUDED.security_id,
            company_name = EXCLUDED.company_name,
            status = EXCLUDED.status,
            last_synced_at = now()
        """,
        (
            company["symbol"],
            str(security_id),
            company["companyName"],
            company["sectorName"],
            company["instrumentType"],
            company["status"],
        ),
    )


def upsert_daily_price(cur, symbol, business_date, row):
    cur.execute(
        """
        INSERT INTO daily_prices (symbol, business_date, high_price, low_price,
                                   close_price, total_traded_quantity,
                                   total_traded_value, total_trades)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, business_date) DO UPDATE SET
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            total_traded_quantity = EXCLUDED.total_traded_quantity,
            total_traded_value = EXCLUDED.total_traded_value,
            total_trades = EXCLUDED.total_trades
        """,
        (
            symbol,
            business_date,
            row.get("highPrice"),
            row.get("lowPrice"),
            row.get("closePrice"),
            row.get("totalTradedQuantity"),
            row.get("totalTradedValue"),
            row.get("totalTrades"),
        ),
    )


def log_run(cur, status, attempted, succeeded, failed, failed_symbols, error_detail=None):
    cur.execute(
        """
        INSERT INTO scrape_log (run_finished_at, status, tickers_attempted,
                                 tickers_succeeded, tickers_failed,
                                 failed_symbols, error_detail)
        VALUES (now(), %s, %s, %s, %s, %s, %s)
        """,
        (status, attempted, succeeded, failed, failed_symbols or None, error_detail),
    )


def main():
    today = date.today()
    print(f"=== Daily scrape starting for {today} ===")

    client = Nepse()
    client.setTLSVerification(False)

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    log_market_status_for_visibility(client)

    print("Fetching active commercial bank ticker list...")
    banks = get_active_bank_tickers(client)
    print(f"Found {len(banks)} active commercial bank tickers.")

    security_id_map = client.getSecurityIDKeyMap()

    succeeded_symbols = []
    no_data_symbols = []   # clean "no rows for today" — expected on holidays
    error_symbols = []     # an actual exception after retries — a real problem

    for company in banks:
        symbol = company["symbol"]
        print(f"  Processing {symbol}...")
        try:
            upsert_company(cur, company, security_id_map.get(symbol))

            row = fetch_today_price(client, symbol, today)
            if row is None:
                print(f"    {symbol}: no data returned for today.")
                no_data_symbols.append(symbol)
                continue

            upsert_daily_price(cur, symbol, today, row)
            succeeded_symbols.append(symbol)

        except Exception as e:
            print(f"    {symbol}: unexpected error, skipping this ticker: {e}")
            error_symbols.append(symbol)
            conn.rollback()  # undo any partial work for this ticker, keep going
            cur = conn.cursor()

    attempted = len(banks)
    succeeded = len(succeeded_symbols)
    no_data = len(no_data_symbols)
    errored = len(error_symbols)
    failed = no_data + errored  # for the scrape_log column, which just wants a total

    # The actual decision: driven entirely by what happened, not by any
    # pre-fetch status check. Every ticker cleanly reporting "no data,"
    # with zero real errors, is what a genuine non-trading day looks like.
    if succeeded == 0 and errored == 0 and no_data == attempted and attempted > 0:
        print("Every ticker had no data today, with no fetch errors — "
              "treating this as a non-trading day.")
        status = "non_trading_day"
    elif errored == 0 and no_data == 0:
        status = "success"
    elif succeeded > 0:
        status = "partial_failure"
    else:
        status = "failure"

    log_run(cur, status, attempted, succeeded, failed, no_data_symbols + error_symbols)
    conn.commit()
    cur.close()
    conn.close()

    print(f"=== Done. status={status}, succeeded={succeeded}/{attempted} "
          f"(no_data={no_data}, errored={errored}) ===")


if __name__ == "__main__":
    main()
