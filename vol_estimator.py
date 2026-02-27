"""
Realized Volatility Estimator for BTC — V2

RESEARCH-DRIVEN UPGRADES over V1:
  1. EGARCH-style leverage effect in EWMA
     - EGARCH(1,1) with Student-t is best BTC vol model (Springer 2025)
     - Negative returns increase vol more than positive (asymmetric response)
     - Persistence parameter 0.91-0.99 for BTC

  2. Jump detection
     - Flags returns exceeding 3σ threshold as "jumps"
     - Exports jump intensity/mean/vol for Merton jump-diffusion pricer
     - Tracks recent jump state for conditional parameter adjustment

  3. Return statistics for overlays
     - Recent return since market open (for mean reversion signal)
     - Return in sigma units (standardized)
     - Rolling autocorrelation (confirms mean-reversion regime)

  4. Variance Risk Premium (VRP) tracking
     - IV overprices RV ~70% of the time by ~15 vol points for BTC
     - BTC annualized VRP ≈ 14% (vs 2% for S&P 500)
     - When VRP is wide, fading extreme market probabilities has structural edge

Typical BTC 15-min realized vol: 40-100% annualized
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECONDS_PER_YEAR = 365.25 * 24 * 3600
MINUTES_PER_YEAR = 365.25 * 24 * 60


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
        return (
            f"Vol({self.annualized_vol:.1%} ann, "
            f"{self.vol_per_minute:.4%}/min, "
            f"n={self.num_returns}, "
            f"{self.sample_minutes:.1f}min window, "
            f"method={self.method})"
        )


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
        return (
            f"Jumps(λ={self.intensity:.0f}/yr, μ_J={self.mean:.4f}, "
            f"σ_J={self.vol:.4f}, recent={self.recent_jump}, "
            f"n_jumps={self.num_jumps_detected})"
        )


@dataclass
class ReturnStats:
    """Return statistics for mean reversion and other overlays."""
    recent_return: float            # Return since some reference (e.g., market open)
    recent_return_sigma: float      # That return in σ units
    rolling_autocorr: float         # First-order autocorrelation of recent returns
    mean_reversion_active: bool     # Whether |return_sigma| > threshold
    num_returns: int

    def __repr__(self):
        return (
            f"Returns(ret={self.recent_return:.4%}, "
            f"σ={self.recent_return_sigma:.2f}, "
            f"autocorr={self.rolling_autocorr:+.3f}, "
            f"mr_active={self.mean_reversion_active})"
        )


@dataclass
class PriceSample:
    """A single price observation."""
    timestamp: float
    price: float
    high: Optional[float] = None
    low: Optional[float] = None


# ---------------------------------------------------------------------------
# Vol Estimator V2
# ---------------------------------------------------------------------------

class VolEstimator:
    """
    Real-time realized volatility estimator with jump detection.

    V2 additions:
      - EGARCH-style leverage effect (negative returns → higher vol)
      - Jump detection (flags 3σ outliers, exports jump parameters)
      - Return statistics (recent return, σ-units, autocorrelation)
      - VRP tracking (implied vol vs realized vol spread)
    """

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

        # Raw tick buffer
        max_ticks = int(window_minutes * 60 / max(resample_interval_sec, 1) * 10)
        self._raw_ticks: deque = deque(maxlen=max(max_ticks, 1000))

        # Resampled price series (1-minute bars)
        max_bars = int(window_minutes / (resample_interval_sec / 60)) + 10
        self._bars: deque = deque(maxlen=max(max_bars, 200))
        self._last_bar_time: float = 0.0

        # Cache
        self._cached_vol: Optional[VolEstimate] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 10.0

        # EWMA decay factor
        self._ewma_lambda = 1.0 - math.log(2) / max(ewma_halflife_samples, 1)

        # V2: Jump tracking
        self._detected_jumps: deque = deque(maxlen=100)  # (timestamp, return_size)
        self._recent_jump_window_sec: float = 300.0  # 5 minutes

        # V2: VRP tracking
        self._iv_observations: deque = deque(maxlen=50)  # (timestamp, iv)
        self._rv_observations: deque = deque(maxlen=50)  # (timestamp, rv)

        # V2: Simulated time for backtesting
        # When set, get_vol() uses this instead of time.time() for pruning
        self._simulated_time: Optional[float] = None

        logger.info(
            f"Initialized VolEstimator V2: window={window_minutes}min, "
            f"resample={resample_interval_sec}s, default_vol={default_vol:.0%}, "
            f"leverage_gamma={leverage_gamma}, jump_threshold={jump_threshold_sigma}σ"
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Vol estimation
    # ------------------------------------------------------------------

    def get_vol(self, method: str = "ewma") -> VolEstimate:
        """Get current realized vol. Methods: 'close_to_close', 'ewma', 'parkinson'."""
        now = self._now()

        if (self._cached_vol is not None
                and now - self._cache_time < self._cache_ttl
                and self._cached_vol.method == method):
            return self._cached_vol

        # Prune old bars
        cutoff = now - self.window_minutes * 60
        while self._bars and self._bars[0].timestamp < cutoff:
            self._bars.popleft()

        if len(self._bars) < self.min_samples + 1:
            estimate = VolEstimate(
                annualized_vol=self.default_vol,
                sample_minutes=0.0,
                num_returns=len(self._bars) - 1 if self._bars else 0,
                vol_per_minute=self.default_vol / math.sqrt(MINUTES_PER_YEAR),
                method="default",
                confidence=0.0,
                timestamp=now,
            )
            self._cached_vol = estimate
            self._cache_time = now
            return estimate

        if method == "ewma":
            estimate = self._ewma_vol_v2()
        elif method == "parkinson":
            estimate = self._parkinson_vol()
        else:
            estimate = self._close_to_close_vol()

        self._cached_vol = estimate
        self._cache_time = now
        return estimate

    def _ewma_vol_v2(self) -> VolEstimate:
        """
        EWMA vol with EGARCH-style leverage effect.

        Standard EWMA: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t
        V2 with leverage: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t + γ·I(r<0)·r²_t

        The γ (leverage_gamma) term makes negative returns contribute MORE
        to volatility than positive returns of the same magnitude.
        This matches EGARCH(1,1) behavior found optimal for BTC.
        """
        now = self._now()
        bars = list(self._bars)

        log_returns = []
        for i in range(1, len(bars)):
            if bars[i - 1].price > 0 and bars[i].price > 0:
                lr = math.log(bars[i].price / bars[i - 1].price)
                log_returns.append(lr)

        if len(log_returns) < self.min_samples:
            return VolEstimate(
                annualized_vol=self.default_vol, sample_minutes=0.0,
                num_returns=len(log_returns),
                vol_per_minute=self.default_vol / math.sqrt(MINUTES_PER_YEAR),
                method="default", confidence=0.0, timestamp=now,
            )

        lam = self._ewma_lambda
        gamma = self.leverage_gamma
        ewma_var = log_returns[0] ** 2

        for lr in log_returns[1:]:
            # Standard EWMA component
            innovation = lr * lr
            # EGARCH leverage: negative returns get extra weight
            if lr < 0:
                innovation *= (1.0 + gamma)
            ewma_var = lam * ewma_var + (1 - lam) * innovation

        vol_per_interval = math.sqrt(max(ewma_var, 1e-20))

        interval_minutes = self.resample_interval_sec / 60.0
        vol_per_minute = vol_per_interval / math.sqrt(interval_minutes) if interval_minutes > 0 else 0.0
        annualized = vol_per_minute * math.sqrt(MINUTES_PER_YEAR)
        annualized = max(0.10, min(3.0, annualized))

        sample_minutes = (bars[-1].timestamp - bars[0].timestamp) / 60.0
        confidence = min(1.0, len(log_returns) / 30.0)

        return VolEstimate(
            annualized_vol=annualized, sample_minutes=sample_minutes,
            num_returns=len(log_returns), vol_per_minute=vol_per_minute,
            method="ewma", confidence=confidence, timestamp=now,
        )

    def _close_to_close_vol(self) -> VolEstimate:
        """Standard close-to-close realized vol from log returns."""
        now = self._now()
        bars = list(self._bars)

        log_returns = []
        for i in range(1, len(bars)):
            if bars[i - 1].price > 0 and bars[i].price > 0:
                log_returns.append(math.log(bars[i].price / bars[i - 1].price))

        if len(log_returns) < self.min_samples:
            return VolEstimate(
                annualized_vol=self.default_vol, sample_minutes=0.0,
                num_returns=len(log_returns),
                vol_per_minute=self.default_vol / math.sqrt(MINUTES_PER_YEAR),
                method="default", confidence=0.0, timestamp=now,
            )

        mean_lr = sum(log_returns) / len(log_returns)
        variance = sum((lr - mean_lr) ** 2 for lr in log_returns) / (len(log_returns) - 1)
        vol_per_interval = math.sqrt(variance)

        interval_minutes = self.resample_interval_sec / 60.0
        vol_per_minute = vol_per_interval / math.sqrt(interval_minutes) if interval_minutes > 0 else 0.0
        annualized = vol_per_minute * math.sqrt(MINUTES_PER_YEAR)
        annualized = max(0.10, min(3.0, annualized))

        sample_minutes = (bars[-1].timestamp - bars[0].timestamp) / 60.0
        confidence = min(1.0, len(log_returns) / 30.0)

        return VolEstimate(
            annualized_vol=annualized, sample_minutes=sample_minutes,
            num_returns=len(log_returns), vol_per_minute=vol_per_minute,
            method="close_to_close", confidence=confidence, timestamp=now,
        )

    def _parkinson_vol(self) -> VolEstimate:
        """Parkinson high-low vol estimator."""
        now = self._now()
        bars = [b for b in self._bars if b.high is not None and b.low is not None]

        if len(bars) < self.min_samples:
            return self._close_to_close_vol()

        sum_sq = 0.0
        valid_bars = 0
        for bar in bars:
            if bar.high > 0 and bar.low > 0 and bar.high >= bar.low:
                hl_ratio = bar.high / bar.low
                if hl_ratio > 0:
                    sum_sq += math.log(hl_ratio) ** 2
                    valid_bars += 1

        if valid_bars < self.min_samples:
            return self._close_to_close_vol()

        parkinson_var = sum_sq / (4.0 * valid_bars * math.log(2))
        vol_per_interval = math.sqrt(parkinson_var)

        interval_minutes = self.resample_interval_sec / 60.0
        vol_per_minute = vol_per_interval / math.sqrt(interval_minutes) if interval_minutes > 0 else 0.0
        annualized = vol_per_minute * math.sqrt(MINUTES_PER_YEAR)
        annualized = max(0.10, min(3.0, annualized))

        sample_minutes = (bars[-1].timestamp - bars[0].timestamp) / 60.0
        confidence = min(1.0, valid_bars / 30.0)

        return VolEstimate(
            annualized_vol=annualized, sample_minutes=sample_minutes,
            num_returns=valid_bars, vol_per_minute=vol_per_minute,
            method="parkinson", confidence=confidence, timestamp=now,
        )

    # ------------------------------------------------------------------
    # V2: Jump detection
    # ------------------------------------------------------------------

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

    def get_jump_params(self) -> JumpEstimate:
        """
        Export jump parameters for Merton jump-diffusion pricer.

        Estimates λ (intensity), μ_J (mean), σ_J (vol) from detected jumps.
        Falls back to conservative defaults when insufficient data.
        """
        now = self._now()

        # Prune old jumps
        while self._detected_jumps and self._detected_jumps[0][0] < now - self.window_minutes * 60:
            self._detected_jumps.popleft()

        num_jumps = len(self._detected_jumps)
        window_years = self.window_minutes / MINUTES_PER_YEAR

        # Recent jump check (last 5 minutes)
        recent_jump = any(
            ts > now - self._recent_jump_window_sec
            for ts, _ in self._detected_jumps
        )

        if num_jumps < 2:
            # Conservative defaults: ~4 jumps/day, slight negative bias
            return JumpEstimate(
                intensity=1500.0,   # ~4/day
                mean=-0.0008,       # Slight negative (liquidation cascades)
                vol=0.003,          # ~0.3% typical jump size
                recent_jump=recent_jump,
                num_jumps_detected=num_jumps,
                confidence=0.1,
            )

        # Estimate from data
        jump_returns = [lr for _, lr in self._detected_jumps]

        intensity = num_jumps / window_years if window_years > 0 else 1500.0
        mean_jump = sum(jump_returns) / len(jump_returns)
        var_jump = (
            sum((jr - mean_jump) ** 2 for jr in jump_returns) / (len(jump_returns) - 1)
            if len(jump_returns) > 1 else 0.003 ** 2
        )
        vol_jump = math.sqrt(max(var_jump, 1e-10))

        # Clamp to reasonable ranges
        intensity = max(100.0, min(50000.0, intensity))
        vol_jump = max(0.001, min(0.05, vol_jump))

        confidence = min(1.0, num_jumps / 10.0)

        # If we just had a jump, increase intensity (Hawkes self-excitation)
        if recent_jump:
            intensity *= 2.0  # Jumps cluster: double intensity after recent jump

        return JumpEstimate(
            intensity=intensity,
            mean=mean_jump,
            vol=vol_jump,
            recent_jump=recent_jump,
            num_jumps_detected=num_jumps,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # V2: Return statistics for overlays
    # ------------------------------------------------------------------

    def get_return_stats(
        self, reference_price: Optional[float] = None
    ) -> ReturnStats:
        """
        Compute return statistics for mean reversion and other overlays.

        Args:
            reference_price: BTC price at market open (strike). If None,
                            uses the first bar's price as reference.
        """
        bars = list(self._bars)

        if len(bars) < 2:
            return ReturnStats(
                recent_return=0.0, recent_return_sigma=0.0,
                rolling_autocorr=0.0, mean_reversion_active=False,
                num_returns=0,
            )

        # Recent return vs reference
        current_price = bars[-1].price
        ref_price = reference_price or bars[0].price

        if ref_price > 0 and current_price > 0:
            recent_return = (current_price - ref_price) / ref_price
        else:
            recent_return = 0.0

        # Return in sigma units
        vol_est = self.get_vol(method="ewma")
        elapsed_minutes = (bars[-1].timestamp - bars[0].timestamp) / 60.0
        if vol_est.vol_per_minute > 0 and elapsed_minutes > 0:
            expected_move = vol_est.vol_per_minute * math.sqrt(max(elapsed_minutes, 0.1))
            recent_sigma = recent_return / expected_move if expected_move > 0 else 0.0
        else:
            recent_sigma = 0.0

        # Rolling first-order autocorrelation of returns
        log_returns = []
        for i in range(1, len(bars)):
            if bars[i - 1].price > 0 and bars[i].price > 0:
                log_returns.append(math.log(bars[i].price / bars[i - 1].price))

        autocorr = self._compute_autocorr(log_returns) if len(log_returns) > 5 else 0.0

        mean_reversion_active = abs(recent_sigma) > 1.5

        return ReturnStats(
            recent_return=recent_return,
            recent_return_sigma=recent_sigma,
            rolling_autocorr=autocorr,
            mean_reversion_active=mean_reversion_active,
            num_returns=len(log_returns),
        )

    def _compute_autocorr(self, returns: list) -> float:
        """Compute first-order autocorrelation of return series."""
        if len(returns) < 3:
            return 0.0

        n = len(returns)
        mean = sum(returns) / n
        var = sum((r - mean) ** 2 for r in returns) / n

        if var < 1e-20:
            return 0.0

        cov = sum(
            (returns[i] - mean) * (returns[i - 1] - mean)
            for i in range(1, n)
        ) / (n - 1)

        return cov / var

    # ------------------------------------------------------------------
    # V2: VRP tracking
    # ------------------------------------------------------------------

    def record_iv(self, iv: float, timestamp: Optional[float] = None):
        """Record an implied vol observation for VRP tracking."""
        ts = timestamp or time.time()
        self._iv_observations.append((ts, iv))

    def record_rv(self, rv: float, timestamp: Optional[float] = None):
        """Record a realized vol observation for VRP tracking."""
        ts = timestamp or time.time()
        self._rv_observations.append((ts, rv))

    def get_vrp(self) -> float:
        """
        Get current Variance Risk Premium (IV - RV).

        Positive VRP means IV > RV (market overpricing vol).
        BTC average: ~14% annualized (7x larger than S&P 500).
        When VRP is wide, market binary prices overstate move probability.
        """
        if not self._iv_observations or not self._rv_observations:
            return 0.15  # Default: assume ~15% VRP (research average)

        # Use most recent observations
        recent_iv = self._iv_observations[-1][1]
        recent_rv = self._rv_observations[-1][1]

        return recent_iv - recent_rv

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "raw_ticks": len(self._raw_ticks),
            "bars": len(self._bars),
            "window_minutes": self.window_minutes,
            "last_bar_time": self._last_bar_time,
            "cached_vol": self._cached_vol.annualized_vol if self._cached_vol else None,
            "jumps_detected": len(self._detected_jumps),
            "iv_observations": len(self._iv_observations),
            "rv_observations": len(self._rv_observations),
        }

    def reset(self):
        """Clear all data (e.g., on market switch)."""
        self._raw_ticks.clear()
        self._bars.clear()
        self._last_bar_time = 0.0
        self._cached_vol = None
        self._cache_time = 0.0
        # NOTE: Do NOT clear jump history or VRP — those persist across markets
        # self._detected_jumps.clear()  # Keep!
        # self._iv_observations.clear()  # Keep!


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_vol_instance = None

def get_vol_estimator() -> VolEstimator:
    global _vol_instance
    if _vol_instance is None:
        _vol_instance = VolEstimator()
    return _vol_instance