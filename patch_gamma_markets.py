"""
Enhanced patch for Polymarket gamma_markets.py and provider.py
- Fixes array parameter handling in gamma_markets.py
- Forces load_all_async to use Gamma API with time filters
"""

import os
from typing import Any, Dict, List, Tuple, Union
import logging
import asyncio

logger = logging.getLogger(__name__)


# =========================================================================
# Patch function 1: Fixed query builder
# =========================================================================

_SCALAR_KEYS = (
    "active", "archived", "closed", "limit", "offset", "order",
    "ascending", "liquidity_num_min", "liquidity_num_max",
    "volume_num_min", "volume_num_max", "start_date_min",
    "start_date_max", "end_date_min", "end_date_max", "tag_id", "related_tags",
)

_ARRAY_KEYS = (
    "id", "slug", "clob_token_ids", "condition_ids",
    "question_ids", "market_maker_address",
)


def _build_base_params(filters):
    """Build base params from is_active flag."""
    params = {}
    if filters.get("is_active") is True:
        params.update({"active": "true", "archived": "false", "closed": "false"})
    return params


def patched_build_markets_query(filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Patched version that properly handles array parameters."""
    if not filters:
        return {}
    params = _build_base_params(filters)
    for key in _SCALAR_KEYS:
        if key in filters and filters[key] is not None:
            params[key] = filters[key]
    for key in _ARRAY_KEYS:
        if key in filters and filters[key] is not None:
            value = filters[key]
            params[key] = list(value) if isinstance(value, (tuple, list)) else [value]
            if key == "slug" and params[key]:
                logger.debug(f"Added {len(params[key])} slug filters")
    return params


# =========================================================================
# Patch function 2: Patched load_all_async
# =========================================================================

async def _load_via_slug_builder(self):
    """Use event_slug_builder to get slugs and load via Gamma Markets API."""
    from nautilus_trader.common.config import resolve_path
    slug_builder = resolve_path(self._config.event_slug_builder)
    market_slugs = slug_builder()
    self._log.info(f"Slug builder returned {len(market_slugs)} market slugs")
    if market_slugs:
        self._log.info(f"  First: {market_slugs[0]}, Last: {market_slugs[-1]}")
    slug_filters = {"active": True, "closed": False, "archived": False,
                    "slug": list(market_slugs), "limit": 100}
    await self._load_all_using_gamma_markets(slug_filters)


async def patched_load_all_async(self, filters: dict | None = None) -> None:
    """Patched load_all_async that supports event_slug_builder for MARKET slugs."""
    from nautilus_trader.adapters.polymarket.providers import PolymarketInstrumentProviderConfig
    if (isinstance(self._config, PolymarketInstrumentProviderConfig)
            and self._config.event_slug_builder):
        await _load_via_slug_builder(self)
        return
    self._log.info("=" * 80)
    self._log.info("LOADING MARKETS VIA GAMMA API (PATCHED)")
    if filters:
        self._log.info(f"Filters: {filters}")
    self._log.info("=" * 80)
    if getattr(self._config, 'use_gamma_markets', False):
        await self._load_all_using_gamma_markets(filters)
    else:
        self._log.warning("Falling back to CLOB API (slow, may ignore filters)")
        await self._load_markets([], filters)


# =========================================================================
# Patch function 3: Gamma Markets loader
# =========================================================================

def _count_market_types(markets):
    """Count BTC/ETH/SOL markets for logging."""
    counts = {"btc": 0, "eth": 0, "sol": 0}
    for m in markets:
        slug = m.get("slug", "").lower()
        for key in counts:
            if key in slug:
                counts[key] += 1; break
    other = len(markets) - sum(counts.values())
    return counts["btc"], counts["eth"], counts["sol"], other


def _process_market(self, market, gamma_markets_mod):
    """Process a single market into instruments. Returns count loaded."""
    normalized = gamma_markets_mod.normalize_gamma_market_to_clob_format(market)
    slug = market.get("slug", "")
    if "btc" in slug.lower() and "15m" in slug.lower():
        self._log.info(f"✓ Found BTC 15-min market: {slug}")
    count = 0
    for token_info in normalized.get("tokens", []):
        token_id = token_info["token_id"]
        if not token_id:
            continue
        self._load_instrument(normalized, token_id, token_info["outcome"])
        count += 1
    return count


async def _load_all_using_gamma_markets(self, filters: dict | None = None) -> None:
    """Load instruments using Gamma API with proper server-side filtering."""
    from nautilus_trader.adapters.polymarket.common import gamma_markets as gm
    filters = (filters or {}).copy()
    filters.setdefault("limit", 1000)
    self._log.info(f"Requesting markets from Gamma API with filters: {filters}")
    try:
        markets = await gm.list_markets(
            http_client=self._http_client, filters=filters, timeout=120.0)
        self._log.info(f"✓ Gamma API returned {len(markets)} markets")
        if not markets:
            self._log.warning("No markets found with current filters"); return
        btc, eth, sol, other = _count_market_types(markets)
        self._log.info(f"Market breakdown: {btc} BTC, {eth} ETH, {sol} SOL, {other} other")
        loaded = 0
        for market in markets:
            try:
                loaded += _process_market(self, market, gm)
            except Exception as e:
                self._log.error(f"Error processing market {market.get('slug', '?')}: {e}")
        self._log.info(f"Loaded {loaded} instruments from {len(markets)} markets")
        if btc == 0:
            self._log.warning("No BTC markets found in this batch")
    except Exception as e:
        self._log.error(f"Gamma API request failed: {e}")
        import traceback; traceback.print_exc()


# =========================================================================
# Apply all patches
# =========================================================================

def apply_gamma_markets_patch():
    """Monkey-patch gamma_markets.py and provider.py for proper filtering."""
    try:
        from nautilus_trader.adapters.polymarket.common import gamma_markets
        from nautilus_trader.adapters.polymarket import providers

        logger.info("=" * 80)
        logger.info("Applying enhanced patches for Polymarket filtering")
        logger.info("=" * 80)

        gamma_markets.build_markets_query = patched_build_markets_query
        logger.info("✓ Patched gamma_markets.build_markets_query")

        providers.PolymarketInstrumentProvider.load_all_async = patched_load_all_async
        providers.PolymarketInstrumentProvider._load_all_using_gamma_markets = _load_all_using_gamma_markets
        logger.info("✓ Patched PolymarketInstrumentProvider.load_all_async")
        logger.info("=" * 80)
        return True
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}"); return False
    except Exception as e:
        logger.error(f"Failed to apply patch: {e}")
        import traceback; traceback.print_exc()
        return False


def verify_patch():
    """Verify that the patch is working."""
    try:
        from nautilus_trader.adapters.polymarket.common import gamma_markets
        from nautilus_trader.adapters.polymarket import providers
        test_filters = {"active": True, "closed": False, "archived": False,
                        "slug": ("test-slug-1", "test-slug-2"),
                        "end_date_min": "2026-01-01T00:00:00Z"}
        params = gamma_markets.build_markets_query(test_filters)
        logger.info(f"Query builder test: {test_filters} -> {params}")
        has_patched = hasattr(providers.PolymarketInstrumentProvider, '_load_all_using_gamma_markets')
        logger.info(f"Provider has patched method: {has_patched}")
        return has_patched
    except Exception as e:
        logger.error(f"Failed to verify patch: {e}"); return False
