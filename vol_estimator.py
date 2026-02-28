"""
Realized Volatility Estimator V2 — EGARCH-style EWMA + jump detection.

Key upgrades: leverage effect, jump detection (3σ), return stats for overlays,
VRP tracking (IV-RV). Analytics methods in vol_analytics.py (VolAnalyticsMixin).
Typical BTC 15-min realized vol: 40-100% annualized.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from loguru import logger
from vol_analytics import VolAnalyticsMixin
from vol_methods import VolMethodsMixin

SECONDS_PER_YEAR = 365.25 * 24 * 3600
MINUTES_PER_YEAR = 365.25 * 24 * 60


@dataclass
class VolEstimate:
    """Volatility estimate with metadata."""
    annualized_vol: float
    sample_minutes: float
    num_returns: int
    vol_per_minute: float
    method: str
    confidence: float           # 0-1, based on sample size
    timestamp: float

    def __repr__(self):
        return f"Vol({self.annualized_vol:.1%} ann, n={self.num_returns}, {self.method})"


@dataclass
class JumpEstimate:
    """Jump parameters for Merton jump-diffusion pricer."""
    intensity: float        # λ: jumps per year
    mean: float             # μ_J: mean log-jump size
    vol: float              # σ_J: jump size std dev
    recent_jump: bool       # Whether a jump was detected in the last N bars
    num_jumps_detected: int # Jumps in current window
    confidence: float       # 0-1

    def __repr__(self):
        return f"Jumps(λ={self.intensity:.0f}/yr, n={self.num_jumps_detected}, recent={self.recent_jump})"


@dataclass
class ReturnStats:
    """Return statistics for mean reversion and other overlays."""
    recent_return: float            # Return since some reference (e.g., market open)
    recent_return_sigma: float      # That return in σ units
    rolling_autocorr: float         # First-order autocorrelation of recent returns
    mean_reversion_active: bool     # Whether |return_sigma| > threshold
    num_returns: int

    def __repr__(self):
        return f"Returns(ret={self.recent_return:.4%}, σ={self.recent_return_sigma:.2f}, mr={self.mean_reversion_active})"


@dataclass
class PriceSample:
    """A single price observation."""
    timestamp: float
    price: float
    high: Optional[float] = None
    low: Optional[float] = None


class VolEstimator(VolAnalyticsMixin, VolMethodsMixin):
    """Real-time RV estimator: EGARCH-style EWMA + jump detection + VRP tracking."""

    def __init__(
        self,
        window_minutes: float = 60.0,
        resample_interval_sec: float = 60.0,
        ewma_halflife_samples: int = 20,
        min_samples: int = 5,
        default_vol: float = 0.65,
        # V2: EGARCH parameters
        leverage_gamma: float = 0.15,       # Asymmetric response to negative returns
        # V2: Jump detection
        jump_threshold_sigma: float = 3.0,  # Flag returns > 3σ as jumps
    ):
        self.window_minutes = window_minutes
        self.resample_interval_sec = resample_interval_sec
        self.ewma_halflife_samples = ewma_halflife_samples
        self.min_samples = min_samples
        self.default_vol = default_vol
        self.leverage_gamma = leverage_gamma
        self.jump_threshold_sigma = jump_threshold_sigma

        self._ewma_lambda = 1.0 - math.log(2) / max(ewma_halflife_samples, 1)
        self._simulated_time: Optional[float] = None

        self._init_buffers()

        logger.info(f"VolEstimator V2: w={window_minutes}m, resamp={resample_interval_sec}s, "
                    f"vol={default_vol:.0%}, γ={leverage_gamma}, jump={jump_threshold_sigma}σ")

    def _init_buffers(self):
        """Initialize all internal data buffers."""
        max_ticks = int(self.window_minutes * 60 / max(self.resample_interval_sec, 1) * 10)
        self._raw_ticks: deque = deque(maxlen=max(max_ticks, 1000))
        max_bars = int(self.window_minutes / (self.resample_interval_sec / 60)) + 10
        self._bars: deque = deque(maxlen=max(max_bars, 200))
        self._last_bar_time: float = 0.0
        self._cached_vol: Optional[VolEstimate] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 10.0
        self._detected_jumps: deque = deque(maxlen=100)
        self._recent_jump_window_sec: float = 300.0
        self._iv_observations: deque = deque(maxlen=50)
        self._rv_observations: deque = deque(maxlen=50)

    def add_price(self, price: float, timestamp: Optional[float] = None):
        """Add a new BTC price observation."""
        if price <= 0:
            return

        ts = timestamp or time.time()
        self._raw_ticks.append(PriceSample(timestamp=ts, price=price))

        bar_boundary = int(ts / self.resample_interval_sec) * self.resample_interval_sec

        if bar_boundary > self._last_bar_time:
            # Detect jumps on bar close
            if self._bars and self._bars[-1].price > 0:
                log_ret = math.log(price / self._bars[-1].price)
                self._check_jump(ts, log_ret)

            self._bars.append(PriceSample(
                timestamp=bar_boundary, price=price,
                high=price, low=price,
            ))
            self._last_bar_time = bar_boundary
        else:
            if self._bars:
                self._bars[-1].price = price
                if self._bars[-1].high is None or price > self._bars[-1].high:
                    self._bars[-1].high = price
                if self._bars[-1].low is None or price < self._bars[-1].low:
                    self._bars[-1].low = price

        self._cached_vol = None

    def add_prices_bulk(self, prices: List[Tuple[float, float]]):
        """Add multiple (timestamp, price) pairs at once."""
        for ts, price in prices:
            self.add_price(price, ts)

    def set_simulated_time(self, ts: Optional[float]):
        """Set simulated time for backtesting. Pass None to use real time."""
        self._simulated_time = ts

    def _now(self) -> float:
        """Current time (real or simulated for backtesting)."""
        return self._simulated_time if self._simulated_time is not None else time.time()

    def _default_vol_estimate(self, now):
        """Return a default vol estimate when insufficient data."""
        return VolEstimate(
            annualized_vol=self.default_vol, sample_minutes=0.0,
            num_returns=len(self._bars) - 1 if self._bars else 0,
            vol_per_minute=self.default_vol / math.sqrt(MINUTES_PER_YEAR),
            method="default", confidence=0.0, timestamp=now)

    def get_vol(self, method: str = "ewma") -> VolEstimate:
        """Get current realized vol. Methods: 'close_to_close', 'ewma', 'parkinson'."""
        now = self._now()
        if (self._cached_vol is not None and now - self._cache_time < self._cache_ttl
                and self._cached_vol.method == method):
            return self._cached_vol

        cutoff = now - self.window_minutes * 60
        while self._bars and self._bars[0].timestamp < cutoff:
            self._bars.popleft()

        if len(self._bars) < self.min_samples + 1:
            est = self._default_vol_estimate(now)
        elif method == "ewma":
            est = self._ewma_vol_v2()
        elif method == "parkinson":
            est = self._parkinson_vol()
        else:
            est = self._close_to_close_vol()

        self._cached_vol = est
        self._cache_time = now
        return est

    def _compute_ewma_variance(self, log_returns: list) -> float:
        """Compute EWMA variance with EGARCH-style leverage effect."""
        lam = self._ewma_lambda
        gamma = self.leverage_gamma
        ewma_var = log_returns[0] ** 2

        for lr in log_returns[1:]:
            innovation = lr * lr
            if lr < 0:
                innovation *= (1.0 + gamma)
            ewma_var = lam * ewma_var + (1 - lam) * innovation

        return ewma_var

    def _ewma_vol_v2(self) -> VolEstimate:
        """EWMA vol with EGARCH leverage: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t · (1 + γ·I(r<0))."""
        now = self._now()
        bars = list(self._bars)

        log_returns = []
        for i in range(1, len(bars)):
            if bars[i - 1].price > 0 and bars[i].price > 0:
                lr = math.log(bars[i].price / bars[i - 1].price)
                log_returns.append(lr)

        if len(log_returns) < self.min_samples:
            return self._default_vol_estimate(now, len(log_returns))

        ewma_var = self._compute_ewma_variance(log_returns)
        return self._build_vol_estimate(
            ewma_var, bars, log_returns, "ewma", now
        )

    def _default_vol_estimate(self, now: float, num_returns: int) -> VolEstimate:
        """Return default vol estimate when insufficient data."""
        return VolEstimate(
            annualized_vol=self.default_vol, sample_minutes=0.0,
            num_returns=num_returns,
            vol_per_minute=self.default_vol / math.sqrt(MINUTES_PER_YEAR),
            method="default", confidence=0.0, timestamp=now,
        )

    def _build_vol_estimate(
        self, variance: float, bars: list, log_returns: list,
        method: str, now: float,
    ) -> VolEstimate:
        """Build a VolEstimate from computed variance."""
        vol_per_interval = math.sqrt(max(variance, 1e-20))
        interval_minutes = self.resample_interval_sec / 60.0
        vol_per_minute = vol_per_interval / math.sqrt(interval_minutes) if interval_minutes > 0 else 0.0
        annualized = vol_per_minute * math.sqrt(MINUTES_PER_YEAR)
        annualized = max(0.10, min(3.0, annualized))
        sample_minutes = (bars[-1].timestamp - bars[0].timestamp) / 60.0
        confidence = min(1.0, len(log_returns) / 30.0)
        return VolEstimate(
            annualized_vol=annualized, sample_minutes=sample_minutes,
            num_returns=len(log_returns), vol_per_minute=vol_per_minute,
            method=method, confidence=confidence, timestamp=now,
        )

    # Vol methods inherited from VolMethodsMixin:
    # _close_to_close_vol, _parkinson_vol, _parkinson_sum_squares

    def _check_jump(self, timestamp: float, log_return: float):
        """Check if a return qualifies as a jump (> threshold σ)."""
        vol_est = self._cached_vol
        if vol_est is None or vol_est.vol_per_minute <= 0:
            return

        # Convert return to σ units using current vol estimate
        interval_minutes = self.resample_interval_sec / 60.0
        expected_std = vol_est.vol_per_minute * math.sqrt(interval_minutes)

        if expected_std > 0:
            sigma_units = abs(log_return) / expected_std
            if sigma_units > self.jump_threshold_sigma:
                self._detected_jumps.append((timestamp, log_return))
                logger.warning(
                    f"⚡ JUMP DETECTED: {log_return:+.4%} "
                    f"({sigma_units:.1f}σ) at {timestamp:.0f}"
                )

    # Analytics methods inherited from VolAnalyticsMixin:
    # get_jump_params, _estimate_jump_distribution, get_return_stats,
    # _compute_recent_sigma, _compute_autocorr, record_iv, record_rv,
    # get_vrp, get_stats, reset


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_vol_instance = None

def get_vol_estimator() -> VolEstimator:
    global _vol_instance
    if _vol_instance is None:
        _vol_instance = VolEstimator()
    return _vol_instance







