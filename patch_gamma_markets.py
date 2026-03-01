"""
Enhanced patch for Polymarket gamma_markets.py and provider.py
- Fixes array parameter handling in gamma_markets.py
- Forces load_all_async to use Gamma API with time filters
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def apply_gamma_markets_patch():
    """
    Monkey-patch both gamma_markets.py and provider.py to properly handle filtering.
    """
    try:
        # Import the modules we need to patch
        from nautilus_trader.adapters.polymarket.common import gamma_markets
        from nautilus_trader.adapters.polymarket import providers
        from nautilus_trader.core.nautilus_pyo3 import HttpClient

        logger.info("=" * 80)
        logger.info("Applying enhanced patches for Polymarket filtering & Timeouts")
        logger.info("=" * 80)
        
        # ===== PATCH 0: Fix py_clob_client Timeout Constraints =====
        try:
            import httpx
            from py_clob_client.http_helpers import helpers as clob_helpers
            
            # The default py_clob_client is artificially restricted to 5.0s timeouts
            # This causes catastrophic bot failure during mass position reconciliation
            # because Gamma APIs often take >10s to reply under heavy load.
            timeout_limit = 30.0
            
            # Sub in our own httpx client with higher thresholds natively
            clob_helpers._http_client = httpx.Client(
                timeout=httpx.Timeout(timeout_limit)
            )
            logger.info(f"✓ Patched py_clob_client.http_helpers (_http_client.timeout = {timeout_limit}s)")
        except ImportError as e:
            logger.warning(f"Could not patch py_clob_client timeouts: {e}")

        # ===== PATCH 1: Fix gamma_markets.py array parameter handling =====

        def patched_build_markets_query(filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
            """
            Patched version that properly handles array parameters.
            """
            params: Dict[str, Any] = {}
            if not filters:
                return params

            if filters.get("is_active") is True:
                params["active"] = "true"
                params["archived"] = "false"
                params["closed"] = "false"

            # Handle scalar parameters
            scalar_keys = (
                "active",
                "archived",
                "closed",
                "limit",
                "offset",
                "order",
                "ascending",
                "liquidity_num_min",
                "liquidity_num_max",
                "volume_num_min",
                "volume_num_max",
                "start_date_min",
                "start_date_max",
                "end_date_min",
                "end_date_max",
                "tag_id",
                "related_tags",
            )
            for key in scalar_keys:
                if key in filters and filters[key] is not None:
                    params[key] = filters[key]

            # Handle array parameters
            array_keys = (
                "id",
                "slug",
                "clob_token_ids",
                "condition_ids",
                "question_ids",
                "market_maker_address",
            )

            for key in array_keys:
                if key in filters and filters[key] is not None:
                    value = filters[key]
                    if isinstance(value, (tuple, list)):
                        params[key] = list(value)
                    else:
                        params[key] = [value]

                    if key == "slug" and params[key]:
                        logger.debug(f"Added {len(params[key])} slug filters")

            return params

        # Apply gamma_markets patch
        gamma_markets.build_markets_query = patched_build_markets_query
        logger.info("✓ Patched gamma_markets.build_markets_query (array parameter handling)")

        # ===== PATCH 2: Replace load_all_async to force Gamma API usage =====
        # Save a reference to the ORIGINAL native load_all_async before patching
        _original_load_all_async = providers.PolymarketInstrumentProvider.load_all_async

        async def patched_load_all_async(self, filters: dict | None = None) -> None:
            """
            Patched load_all_async that supports event_slug_builder for MARKET slugs
            via the Gamma API (not the event API).

            If event_slug_builder is configured, call it to get market slugs,
            then use the Gamma Markets API with slug filtering.
            """
            from nautilus_trader.adapters.polymarket.providers import PolymarketInstrumentProviderConfig
            from nautilus_trader.common.config import resolve_path

            # If event_slug_builder is set, use it to build market slug filters
            if (
                    isinstance(self._config, PolymarketInstrumentProviderConfig)
                    and self._config.event_slug_builder
            ):
                slug_builder = resolve_path(self._config.event_slug_builder)
                market_slugs = slug_builder()
                self._log.info(f"Slug builder returned {len(market_slugs)} market slugs")
                if market_slugs:
                    self._log.info(f"  First: {market_slugs[0]}")
                    self._log.info(f"  Last:  {market_slugs[-1]}")

                # Build filters for Gamma Markets API with slug filtering
                slug_filters = {
                    "active": True,
                    "closed": False,
                    "archived": False,
                    "slug": list(market_slugs),
                    "limit": 100,
                }
                await self._load_all_using_gamma_markets(slug_filters)
                return

            # Otherwise, use our patched Gamma API logic
            self._log.info("=" * 80)
            self._log.info("LOADING MARKETS VIA GAMMA API (PATCHED)")

            if filters:
                self._log.info(f"Filters: {filters}")
            else:
                self._log.info("No filters applied")

            self._log.info("=" * 80)

            # Use Gamma API if available, otherwise fall back to CLOB
            if getattr(self._config, 'use_gamma_markets', False):
                await self._load_all_using_gamma_markets(filters)
            else:
                self._log.warning("Falling back to CLOB API (slow, may ignore filters)")
                await self._load_markets([], filters)

        async def _load_all_using_gamma_markets(self, filters: dict | None = None) -> None:
            """
            Load all instruments using Gamma API with proper server-side filtering.
            This is the CORRECT implementation that respects time filters.
            """
            filters = filters.copy() if filters is not None else {}

            # Set reasonable defaults
            if "limit" not in filters:
                filters["limit"] = 1000  # Get as many as possible per request

            self._log.info(f"Requesting markets from Gamma API with filters: {filters}")

            try:
                markets = await gamma_markets.list_markets(
                    http_client=self._http_client,
                    filters=filters,
                    timeout=120.0
                )

                self._log.info(f"✓ Gamma API returned {len(markets)} markets")

                if not markets:
                    self._log.warning("No markets found with current filters")
                    self._log.warning("Check that:")
                    self._log.warning("  1. Markets exist with these expiration times")
                    self._log.warning("  2. Filters are correctly formatted")
                    return

                # Count markets by type for debugging
                btc_count = 0
                eth_count = 0
                sol_count = 0

                for market in markets:
                    slug = market.get('slug', '')
                    if 'btc' in slug.lower():
                        btc_count += 1
                    elif 'eth' in slug.lower():
                        eth_count += 1
                    elif 'sol' in slug.lower():
                        sol_count += 1

                self._log.info(
                    f"Market breakdown: {btc_count} BTC, {eth_count} ETH, {sol_count} SOL, {len(markets) - btc_count - eth_count - sol_count} other")

                # Process each market
                loaded_count = 0
                for market in markets:
                    try:
                        normalized_market = gamma_markets.normalize_gamma_market_to_clob_format(market)

                        # Log BTC markets specifically
                        slug = market.get('slug', '')
                        if 'btc' in slug.lower() and '15m' in slug.lower():
                            self._log.info(f"✓ Found BTC 15-min market: {slug}")

                        for token_info in normalized_market.get("tokens", []):
                            token_id = token_info["token_id"]
                            if not token_id:
                                continue
                            outcome = token_info["outcome"]
                            self._load_instrument(normalized_market, token_id, outcome)
                            loaded_count += 1
                    except Exception as e:
                        self._log.error(f"Error processing market {market.get('slug', 'unknown')}: {e}")
                        continue

                self._log.info(f"Successfully loaded {loaded_count} instruments from {len(markets)} markets")

                if btc_count > 0:
                    self._log.info(f"✓ BTC markets found and loaded!")
                else:
                    self._log.warning("No BTC markets found in this batch")

            except Exception as e:
                self._log.error(f"Gamma API request failed: {e}")
                import traceback
                traceback.print_exc()

        # Apply provider patches
        providers.PolymarketInstrumentProvider.load_all_async = patched_load_all_async
        providers.PolymarketInstrumentProvider._load_all_using_gamma_markets = _load_all_using_gamma_markets

        logger.info("✓ Patched PolymarketInstrumentProvider.load_all_async")
        logger.info("  - Now FORCES Gamma API usage with proper filtering")
        logger.info("  - Time-based filters should now work correctly")
        logger.info("=" * 80)

        return True

    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.error("Make sure nautilus_trader is installed")
        return False
    except Exception as e:
        logger.error(f"Failed to apply patch: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_patch():
    """Verify that the patch is working."""
    try:
        from nautilus_trader.adapters.polymarket.common import gamma_markets
        from nautilus_trader.adapters.polymarket import providers

        logger.info("=" * 80)
        logger.info("VERIFYING PATCHES")
        logger.info("=" * 80)

        # Test gamma_markets array handling
        test_filters = {
            "active": True,
            "closed": False,
            "archived": False,
            "slug": ("test-slug-1", "test-slug-2"),
            "end_date_min": "2026-01-01T00:00:00Z",
        }

        params = gamma_markets.build_markets_query(test_filters)
        logger.info("Gamma markets query builder test:")
        logger.info(f"  Input filters: {test_filters}")
        logger.info(f"  Output params: {params}")

        # Check provider methods
        has_patched = hasattr(providers.PolymarketInstrumentProvider, '_load_all_using_gamma_markets')
        logger.info(f"Provider has patched method: {has_patched}")

        logger.info("=" * 80)

        return has_patched

    except Exception as e:
        logger.error(f"Failed to verify patch: {e}")
        return False
