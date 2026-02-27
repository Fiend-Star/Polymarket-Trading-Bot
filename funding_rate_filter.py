"""
Binance Funding Rate Regime Filter

Fetches BTCUSDT perpetual funding rate from Binance Futures API.
Provides regime classification for the trading strategy:

  - EXTREME_POSITIVE (>+0.05%/8h): Strong long bias → favor mean-reversion SHORT
  - HIGH_POSITIVE (>+0.02%): Mild long bias → slight mean-reversion SHORT preference
  - NEUTRAL: No regime signal
  - HIGH_NEGATIVE (<-0.02%): Mild short bias → slight mean-reversion LONG preference
  - EXTREME_NEGATIVE (<-0.05%/8h): Strong short bias → favor mean-reversion LONG

Research:
  - β coefficient for funding rate → next-8h BTC return: −0.087 (p=0.008)
  - R² only 0.003 standalone, but useful as regime filter
  - Extreme funding = crowded positioning → likely reversal
  - Updates every 8 hours, we poll every 5 minutes for "predicted" funding

No authentication required — Binance public API.

USAGE:
    funder = FundingRateFilter()
    await funder.update()  # or funder.update_sync()
    
    regime = funder.get_regime()
    # regime.classification = "EXTREME_POSITIVE"
    # regime.mean_reversion_bias = -0.02  # Subtract from YES probability
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional

from loguru import logger

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    try:
        import requests as httpx
        HTTPX_AVAILABLE = True
    except ImportError:
        HTTPX_AVAILABLE = False
        logger.warning("Neither httpx nor requests available — FundingRateFilter disabled")


# =============================================================================
# Constants
# =============================================================================
BINANCE_MARK_PRICE_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
BINANCE_FUNDING_RATE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
UPDATE_INTERVAL_SEC = 300  # Poll every 5 minutes
CACHE_TTL_SEC = 300


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class FundingRegime:
    """Current funding rate regime classification."""
    funding_rate: float             # Raw funding rate (e.g., 0.0001 = 0.01%)
    funding_rate_pct: float         # As percentage (e.g., 0.01)
    predicted_rate: Optional[float]  # Predicted next funding rate
    classification: str             # EXTREME_POSITIVE, HIGH_POSITIVE, NEUTRAL, etc.
    mean_reversion_bias: float      # Additive probability adjustment
    index_price: float              # Binance index price
    mark_price: float               # Binance mark price
    basis_bps: float                # (mark - index) / index × 10000
    next_funding_time: int          # Unix ms
    last_update: float              # When we fetched this


# =============================================================================
# Funding Rate Filter
# =============================================================================

class FundingRateFilter:
    """
    Fetches Binance BTCUSDT perpetual funding rate and classifies
    the market regime for mean-reversion bias.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        extreme_threshold: float = 0.0005,   # 0.05% per 8h
        high_threshold: float = 0.0002,      # 0.02% per 8h
        max_bias: float = 0.02,              # Max probability adjustment ±2%
    ):
        self.symbol = symbol
        self.extreme_threshold = extreme_threshold
        self.high_threshold = high_threshold
        self.max_bias = max_bias

        self._lock = threading.Lock()
        self._regime: Optional[FundingRegime] = None
        self._last_fetch: float = 0.0
        self._fetch_count: int = 0

        logger.info(
            f"Initialized FundingRateFilter: {symbol}, "
            f"extreme={extreme_threshold:.4%}, high={high_threshold:.4%}, "
            f"max_bias={max_bias:.1%}"
        )

    @property
    def regime(self) -> Optional[FundingRegime]:
        with self._lock:
            return self._regime

    def get_regime(self) -> FundingRegime:
        """
        Get current funding regime. Returns cached value if fresh,
        otherwise returns a neutral default.
        """
        with self._lock:
            if self._regime and (time.time() - self._regime.last_update) < CACHE_TTL_SEC * 3:
                return self._regime

        # Stale or never fetched — return neutral
        return FundingRegime(
            funding_rate=0.0, funding_rate_pct=0.0,
            predicted_rate=None, classification="NEUTRAL",
            mean_reversion_bias=0.0, index_price=0.0, mark_price=0.0,
            basis_bps=0.0, next_funding_time=0, last_update=0,
        )

    def update_sync(self) -> Optional[FundingRegime]:
        """Synchronous funding rate fetch (for non-async contexts)."""
        if not HTTPX_AVAILABLE:
            return None

        try:
            if hasattr(httpx, 'Client'):
                # httpx
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(
                        BINANCE_MARK_PRICE_URL,
                        params={"symbol": self.symbol},
                    )
                    resp.raise_for_status()
                    data = resp.json()
            else:
                # requests fallback
                resp = httpx.get(
                    BINANCE_MARK_PRICE_URL,
                    params={"symbol": self.symbol},
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()

            return self._process_response(data)

        except Exception as e:
            logger.warning(f"Funding rate fetch failed: {e}")
            return None

    async def update(self) -> Optional[FundingRegime]:
        """Async funding rate fetch."""
        if not HTTPX_AVAILABLE:
            return None

        try:
            if hasattr(httpx, 'AsyncClient'):
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        BINANCE_MARK_PRICE_URL,
                        params={"symbol": self.symbol},
                    )
                    resp.raise_for_status()
                    data = resp.json()
            else:
                # Fallback to sync in thread
                import asyncio
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.update_sync)

            return self._process_response(data)

        except Exception as e:
            logger.warning(f"Async funding rate fetch failed: {e}")
            return None

    def _process_response(self, data: dict) -> FundingRegime:
        """Process Binance premiumIndex response into regime classification."""
        funding_rate = float(data.get("lastFundingRate", 0))
        mark_price = float(data.get("markPrice", 0))
        index_price = float(data.get("indexPrice", 0))
        next_funding_time = int(data.get("nextFundingTime", 0))

        classification, bias = self._classify(funding_rate)
        regime = self._build_regime(
            funding_rate, mark_price, index_price,
            next_funding_time, classification, bias,
        )

        with self._lock:
            self._regime = regime
            self._last_fetch = time.time()
            self._fetch_count += 1

        logger.info(
            f"Funding rate: {funding_rate:+.6f} ({regime.funding_rate_pct:+.4f}%) → "
            f"{classification}, bias={bias:+.3f}, basis={regime.basis_bps:+.1f}bps"
        )
        return regime

    def _build_regime(self, funding_rate, mark_price, index_price,
                      next_funding_time, classification, bias):
        """Build a FundingRegime from parsed data."""
        funding_pct = funding_rate * 100
        basis_bps = 0.0
        if index_price > 0:
            basis_bps = (mark_price - index_price) / index_price * 10000

        return FundingRegime(
            funding_rate=funding_rate,
            funding_rate_pct=funding_pct,
            predicted_rate=None,
            classification=classification,
            mean_reversion_bias=bias,
            index_price=index_price,
            mark_price=mark_price,
            basis_bps=basis_bps,
            next_funding_time=next_funding_time,
            last_update=time.time(),
        )


    def _classify(self, rate: float) -> tuple:
        """Classify funding rate into regime + compute mean-reversion bias."""
        if rate > self.extreme_threshold:
            # Crowded long → expect reversal DOWN
            return "EXTREME_POSITIVE", -self.max_bias
        elif rate > self.high_threshold:
            scale = (rate - self.high_threshold) / (self.extreme_threshold - self.high_threshold)
            return "HIGH_POSITIVE", -self.max_bias * 0.5 * scale
        elif rate < -self.extreme_threshold:
            # Crowded short → expect reversal UP
            return "EXTREME_NEGATIVE", +self.max_bias
        elif rate < -self.high_threshold:
            scale = (abs(rate) - self.high_threshold) / (self.extreme_threshold - self.high_threshold)
            return "HIGH_NEGATIVE", +self.max_bias * 0.5 * scale
        else:
            return "NEUTRAL", 0.0

    def should_update(self) -> bool:
        """Check if it's time to re-fetch."""
        return (time.time() - self._last_fetch) > UPDATE_INTERVAL_SEC

    def get_stats(self) -> dict:
        return {
            "fetch_count": self._fetch_count,
            "last_fetch": self._last_fetch,
            "regime": self._regime.classification if self._regime else "UNKNOWN",
            "funding_rate": self._regime.funding_rate if self._regime else None,
            "basis_bps": self._regime.basis_bps if self._regime else None,
        }


# =============================================================================
# Singleton
# =============================================================================
_funder_instance = None

def get_funding_rate_filter() -> FundingRateFilter:
    global _funder_instance
    if _funder_instance is None:
        _funder_instance = FundingRateFilter()
    return _funder_instance
