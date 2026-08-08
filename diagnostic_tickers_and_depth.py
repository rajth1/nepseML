"""
Phase 1 diagnostic — final two open questions:
  A. What's the REAL commercial-bank equity ticker list (filtering out
     debentures/promoter shares/bonds that share the same sectorName)?
  B. How far back does getCompanyPriceVolumeHistory actually go? Its
     response looks paginated — we need the pagination metadata, and to
     try pulling further back, before calling NEPSE-native backfill solved.

Same venv as the previous diagnostics.
"""

import inspect
import json

from nepse import Nepse

client = Nepse()
client.setTLSVerification(False)

print("=" * 70)
print("A. Refined ticker filter: Commercial Banks + Equity + Active")
print("=" * 70)
company_list = client.getCompanyList()

banks = [
    c for c in company_list
    if c.get("sectorName") == "Commercial Banks"
    and c.get("instrumentType") == "Equity"
    and c.get("status") == "A"
]
print(f"Count: {len(banks)}\n")
for b in banks:
    print(f"  {b.get('symbol'):8s} {b.get('companyName')}")

# Just in case instrumentType has other values we're not expecting for
# genuine common stock (e.g. spelled differently), show the full
# breakdown so we can sanity-check nothing real got excluded.
print("\nAll distinct instrumentType values within Commercial Banks sector:")
bank_sector_all = [c for c in company_list if c.get("sectorName") == "Commercial Banks"]
types_seen = {}
for c in bank_sector_all:
    t = c.get("instrumentType")
    types_seen[t] = types_seen.get(t, 0) + 1
for t, n in types_seen.items():
    print(f"  {t}: {n}")

print()
print("=" * 70)
print("B. Pagination / historical depth of getCompanyPriceVolumeHistory")
print("=" * 70)

print("Method signature:")
try:
    print(f"  {inspect.signature(client.getCompanyPriceVolumeHistory)}")
except Exception as e:
    print(f"  (could not introspect signature: {e})")

print("\nSource (if available):")
try:
    print(inspect.getsource(client.getCompanyPriceVolumeHistory))
except Exception as e:
    print(f"  (could not get source: {e})")

print("\nFull top-level keys of a raw call for NABIL:")
result = client.getCompanyPriceVolumeHistory("NABIL")
if isinstance(result, dict):
    for k in result.keys():
        v = result[k]
        if isinstance(v, list):
            print(f"  {k}: list of {len(v)} items")
        else:
            print(f"  {k}: {v!r}")
else:
    print(f"  (result is a {type(result)}, not a dict)")

print("\nFull JSON (top-level only, content truncated to first/last row):")
if isinstance(result, dict):
    shallow = {k: (v if not isinstance(v, list) else
                    (v[:1] + ["...", v[-1]] if len(v) > 2 else v))
               for k, v in result.items()}
    print(json.dumps(shallow, indent=2, default=str))

print("\nDONE. Paste this entire output back to me.")
