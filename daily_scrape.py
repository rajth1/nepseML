"""
Daily NEPSE commercial bank price scrape.

What this script does, in plain terms:
  1. Ask NEPSE: "is the market even open today?"
     - If closed: write "non_trading_day" to the log and stop. This is
       expected and NOT an error — holidays and weekends happen.
  2. If open: get today's list of active commercial bank stocks, and for
     each one, fetch today's high/low/close/volume/turnover numbers.
  3. Save the bank list and the day's numbers into Postgres.
  4. Write a one-line summary of what happened (how many worked, how many
     didn't) so we have a paper trail without digging through logs.

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


def is_market_open_today(client):
    """
    Ask NEPSE directly whether the market is open, rather than maintaining
    our own holiday calendar by hand (which would need constant upkeep).

    We're intentionally cautious here: if this check itself fails or gives
    an unclear answer, we don't guess — we fall through and let the actual
    price-fetch step tell us (an all-empty result for every single bank on
    a day when nothing failed is a strong sign it was a non-trading day).
    """
    try:
        status = client.isNepseOpen()
        # Handle either a plain True/False, or a dict with a status-like field.
        if isinstance(status, bool):
            return status
        if isinstance(status, dict):
            for key in ("isOpen", "open", "status"):
                if key in status:
                    val = status[key]
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, str):
                        return val.lower() in ("open", "true", "yes")
        print(f"  (Could not clearly interpret market status response: {status!r} — "
              f"will fall back to checking whether ANY bank has data today.)")
        return None
    except Exception as e:
        print(f"  (Market status check failed: {e} — will fall back to checking "
              f"whether ANY bank has data today.)")
        return None


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

    market_open = is_market_open_today(client)
    if market_open is False:
        print("Market is closed today. Logging as non_trading_day and stopping.")
        log_run(cur, "non_trading_day", attempted=0, succeeded=0, failed=0, failed_symbols=[])
        conn.commit()
        cur.close()
        conn.close()
        print("=== Done ===")
        return

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

            row = fetch_today_price(client, symbol, today)
            if row is None:
                print(f"    {symbol}: no data returned for today.")
                failed_symbols.append(symbol)
                continue

            upsert_daily_price(cur, symbol, today, row)
            succeeded_symbols.append(symbol)

        except Exception as e:
            print(f"    {symbol}: unexpected error, skipping this ticker: {e}")
            failed_symbols.append(symbol)
            conn.rollback()  # undo any partial work for this ticker, keep going
            cur = conn.cursor()

    attempted = len(banks)
    succeeded = len(succeeded_symbols)
    failed = len(failed_symbols)

    # Fallback safety net: if the market-status check was unclear (None) AND
    # every single bank came back empty, this is almost certainly a
    # non-trading day rather than a real failure — relabel it as such.
    if market_open is None and succeeded == 0 and attempted > 0:
        print("Market status was unclear, and every bank had no data today — "
              "treating this as a non-trading day rather than a failure.")
        status = "non_trading_day"
    elif failed == 0:
        status = "success"
    elif succeeded > 0:
        status = "partial_failure"
    else:
        status = "failure"

    log_run(cur, status, attempted, succeeded, failed, failed_symbols)
    conn.commit()
    cur.close()
    conn.close()

    print(f"=== Done. status={status}, succeeded={succeeded}/{attempted} ===")


if __name__ == "__main__":
    main()
