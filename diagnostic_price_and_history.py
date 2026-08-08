"""
Phase 1 diagnostic — daily snapshot shape + native historical depth.

Same venv as diagnostic_current_site.py (basic-bgnr library already
installed and confirmed working).

What this checks:
  1. getPriceVolume() — is this the market-wide daily OHLCV snapshot we'd
     scrape once a day for every ticker?
  2. getCompanyPriceVolumeHistory('NABIL') / getDailyScripPriceGraph('NABIL')
     — do either of these return real multi-year per-ticker history? If
     yes, NEPSE-native backfill is viable and we can drop the
     ShareSansar/Merolagani fallback plan entirely.
  3. getSectorScrips() — cross-check against the sectorName filter from
     the previous diagnostic, and get the exact commercial-bank ticker
     list / count (brief expects ~20-25).
"""

import json

from nepse import Nepse

client = Nepse()
client.setTLSVerification(False)

def show(label, fn, *args):
    print("=" * 70)
    print(label)
    print("=" * 70)
    try:
        result = fn(*args)
        text = json.dumps(result, indent=2, default=str)
        print(text[:3000])
        if len(text) > 3000:
            print(f"... [truncated, {len(text)} chars total]")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
    print()

# 1. Daily snapshot — is this our once-a-day scrape target?
show("getPriceVolume() — market-wide daily snapshot", client.getPriceVolume)

# 2. Native historical for one bank ticker — the big question
show(
    "getCompanyPriceVolumeHistory('NABIL') — per-company history?",
    client.getCompanyPriceVolumeHistory, "NABIL",
)
show(
    "getDailyScripPriceGraph('NABIL') — per-scrip daily graph?",
    client.getDailyScripPriceGraph, "NABIL",
)

# 3. Confirm commercial-bank ticker list via sector scrips
show("getSectorScrips() — full sector breakdown", client.getSectorScrips)

try:
    company_list = client.getCompanyList()
    banks = [c for c in company_list if c.get("sectorName") == "Commercial Banks"]
    print("=" * 70)
    print(f"Commercial Banks tickers via getCompanyList() filter: {len(banks)}")
    print("=" * 70)
    for b in banks:
        print(f"  {b.get('symbol'):8s} {b.get('companyName')}  status={b.get('status')}")
except Exception as e:
    print(f"Ticker filter failed: {e}")

print("\nDONE. Paste this entire output back to me.")
