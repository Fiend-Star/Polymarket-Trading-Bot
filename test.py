"""
Test script to directly query Gamma API for BTC markets.
Run this separately to debug the API response.
"""

import httpx
import asyncio
from datetime import datetime, timezone, timedelta
import json

async def _query_and_print(client, base_url, label, params, btc_filter=False, max_show=5):
    """Query Gamma API and print results."""
    print(f"\n{label}")
    resp = await client.get(f"{base_url}/markets", params=params)
    if resp.status_code != 200:
        print(f"   Error: {resp.status_code}"); return
    data = resp.json()
    print(f"   Found {len(data)} markets")
    if btc_filter:
        btc = [m for m in data if 'btc' in m.get('slug', '').lower()]
        print(f"   BTC markets: {len(btc)}")
        for m in btc:
            print(f"   - {m.get('slug')}: expires {m.get('endDate', '')}")
    else:
        for m in data[:max_show]:
            print(f"   - {m.get('slug')}: {m.get('question', '')}")


async def test_gamma_api():
    """Test different filtering approaches with Gamma API."""
    base_url = "https://gamma-api.polymarket.com"
    now = datetime.now(timezone.utc)
    base = {"active": "true", "closed": "false", "archived": "false"}
    time_params = {"end_date_min": now.isoformat(),
                   "end_date_max": (now + timedelta(minutes=30)).isoformat()}
    print("=" * 80 + "\nTESTING GAMMA API FILTERING\n" + "=" * 80)
    async with httpx.AsyncClient() as client:
        await _query_and_print(client, base_url,
            "1. All active BTC markets (slug filter)",
            {**base, "limit": 50, "slug": "btc-updown-15m-1771140600"})
        await _query_and_print(client, base_url,
            "2. Markets with crypto tag (744)",
            {**base, "tag_id": 744, "limit": 20})
        await _query_and_print(client, base_url,
            "3. Time filter (next 30 minutes)",
            {**base, "limit": 50, **time_params}, btc_filter=True)
        await _query_and_print(client, base_url,
            "4. Time filter + crypto tag",
            {**base, "tag_id": 744, "limit": 50, **time_params}, btc_filter=True)

if __name__ == "__main__":
    asyncio.run(test_gamma_api())