"""
Phase 1 diagnostic — CURRENT nepalstock.com site.

Run this in its own virtual environment, with ONLY this installed:

    pip install "git+https://github.com/basic-bgnr/NepseUnofficialApi"

Do NOT also `pip install nepse` in this same environment — there is an
unrelated, older PyPI package also named `nepse` (by a different author)
that we test separately in diagnostic_legacy_history.py. The two collide
on the same top-level module name, so they must live in separate venvs.

What this checks:
  1. Does the library still successfully authenticate (decode NEPSE's
     WASM-obfuscated token) against the live site?
  2. What methods does the client actually expose? (We only know
     getCompanyList() for certain from the README; everything else we
     discover here rather than guess.)
  3. What does one company record look like, and is there a sector
     field we can filter on to isolate commercial banks?
"""

import json
import sys

print("=" * 70)
print("STEP 1: Authenticate against nepalstock.com")
print("=" * 70)

try:
    from nepse import Nepse
except ImportError:
    print("Could not import `Nepse`. Install with:")
    print('  pip install "git+https://github.com/basic-bgnr/NepseUnofficialApi"')
    sys.exit(1)

client = Nepse()
client.setTLSVerification(False)  # library README: NEPSE's cert chain is incomplete

try:
    company_list = client.getCompanyList()
    print(f"AUTH OK — getCompanyList() returned {len(company_list)} companies.\n")
except Exception as e:
    print(f"AUTH FAILED: {type(e).__name__}: {e}")
    print("This likely means NEPSE changed its WASM token scheme since the")
    print("library's last patch. Check for open issues at:")
    print("  https://github.com/basic-bgnr/NepseUnofficialApi/issues")
    sys.exit(1)

print("=" * 70)
print("STEP 2: What methods does the client actually expose?")
print("=" * 70)
public_methods = sorted(
    m for m in dir(client) if not m.startswith("_") and callable(getattr(client, m))
)
for m in public_methods:
    print(f"  client.{m}(...)")

print()
print("=" * 70)
print("STEP 3: Shape of one company record")
print("=" * 70)
print(json.dumps(company_list[0], indent=2, default=str))

print()
print("=" * 70)
print("STEP 4: Look for a sector field to isolate commercial banks")
print("=" * 70)
sector_keys_seen = set()
for row in company_list:
    if isinstance(row, dict):
        for key in row:
            if "sector" in key.lower():
                sector_keys_seen.add(key)

if sector_keys_seen:
    for key in sector_keys_seen:
        values = sorted({row.get(key) for row in company_list if isinstance(row, dict)})
        print(f"Field '{key}' — distinct values ({len(values)}):")
        for v in values:
            print(f"    {v}")
else:
    print("No field containing 'sector' found directly on company records.")
    print("Look through the STEP 2 method list above for something like")
    print("'getSectorScrips' / 'getSectorList' / 'getSectorwiseSummary' and")
    print("try calling it — paste the result back to me and I'll adjust the")
    print("ticker-discovery logic accordingly.")

print()
print("DONE. Paste this entire output back to me.")
