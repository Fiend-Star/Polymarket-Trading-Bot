"""
DataSourceManager — owns external data-source connections and caching.

SRP: This class's sole responsibility is connecting to external price/sentiment
     feeds (RTDS, Coinbase, news) and providing thread-safe cached reads.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Optional
from loguru import logger

from interfaces import IPriceSource, ISentimentSource, IVolEstimator


class DataSourceManager:
    """Manages connections and caching for external market data."""

    def __init__(self, vol_estimator: IVolEstimator, rtds=None, funding_filter=None):
        self._vol_estimator = vol_estimator
        self.rtds = rtds
        self.funding_filter = funding_filter
        self._coinbase: Optional[IPriceSource] = None
        self._news: Optional[ISentimentSource] = None
        self._initialized = False
        self._lock = threading.Lock()
        # Cached values
        self._spot: Optional[float] = None
        self._sentiment: Optional[float] = None
        self._sentiment_class: Optional[str] = None

    # ── Public API ───────────────────────────────────────────────────────

    async def init_sources(self):
        """Connect to Coinbase and news/social data sources."""
        if self._initialized:
            return
        self._coinbase = await self._try_connect_source(
            "data_sources.coinbase.adapter", "CoinbaseDataSource", "Coinbase")
        self._news = await self._try_connect_source(
            "data_sources.news_social.adapter", "NewsSocialDataSource", "News/Social")
        self._initialized = True

    async def teardown(self):
        """Disconnect all external sources."""
        for src in (self._news, self._coinbase):
            if src:
                try:
                    await src.disconnect()
                except Exception:
                    pass
        self._initialized = False

    async def refresh(self):
        """Fetch latest spot + sentiment, update cache thread-safely."""
        spot = await self._fetch_spot()
        sent, sent_cls = await self._fetch_sentiment()
        if self.funding_filter and self.funding_filter.should_update():
            try:
                self.funding_filter.update_sync()
            except Exception as e:
                logger.debug(f"Funding update failed: {e}")
        with self._lock:
            if spot is not None:
                self._spot = spot
            if sent is not None:
                self._sentiment = sent
                self._sentiment_class = sent_cls

    def get_cached(self) -> dict:
        """Thread-safe read of cached market data."""
        with self._lock:
            return {
                "spot_price": self._spot,
                "sentiment_score": self._sentiment,
                "sentiment_classification": self._sentiment_class,
            }

    @property
    def cached_spot(self) -> Optional[float]:
        with self._lock:
            return self._spot

    @cached_spot.setter
    def cached_spot(self, value: float):
        with self._lock:
            self._spot = value

    # ── Pre-seeding ──────────────────────────────────────────────────────

    async def preseed_vol(self, vol_method: str):
        """Warm up vol estimator with recent Coinbase candles."""
        import aiohttp
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, params={"granularity": 60},
                                 timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        return
                    candles = await r.json()
            if not candles or not isinstance(candles, list):
                return
            candles.sort(key=lambda c: c[0])
            fed = 0
            for c in candles:
                try:
                    self._vol_estimator.add_price(float(c[4]), float(c[0]))
                    fed += 1
                except (ValueError, IndexError, TypeError):
                    continue
            vol = self._vol_estimator.get_vol(vol_method)
            logger.info(f"Vol pre-seeded: {fed} candles → RV={vol.annualized_vol:.1%}")
            if candles:
                with self._lock:
                    if self._spot is None:
                        self._spot = float(candles[-1][4])
        except Exception as e:
            logger.warning(f"Vol pre-seed failed: {e}")

    # ── Private ──────────────────────────────────────────────────────────

    async def _try_connect_source(self, module, cls_name, label):
        try:
            mod = __import__(module, fromlist=[cls_name])
            src = getattr(mod, cls_name)()
            await src.connect()
            logger.info(f"✓ {label} connected")
            return src
        except Exception as e:
            logger.warning(f"Could not connect {label}: {e}")
            return None

    async def _fetch_spot(self) -> Optional[float]:
        if self.rtds and self.rtds.chainlink_btc_price > 0 and self.rtds.chainlink_age_ms < 30000:
            return self.rtds.chainlink_btc_price
        if self._coinbase:
            try:
                p = await self._coinbase.get_current_price()
                if p:
                    spot = float(p)
                    self._vol_estimator.add_price(spot)
                    return spot
            except Exception as e:
                logger.debug(f"Coinbase refresh failed: {e}")
        return None

    async def _fetch_sentiment(self):
        if not self._news:
            return None, None
        try:
            fg = await self._news.get_fear_greed_index()
            if fg and "value" in fg:
                return float(fg["value"]), fg.get("classification", "")
        except Exception as e:
            logger.debug(f"Sentiment refresh failed: {e}")
        return None, None

