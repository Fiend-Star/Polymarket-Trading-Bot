"""
Typed configuration — single source of truth for all bot settings.

SRP: This module's sole responsibility is loading and validating configuration.
All env-var reads are consolidated here; no other module should call os.getenv().
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: str = "true") -> bool:
    return _env(key, default).lower() == "true"


def _env_int(key: str, default: str) -> int:
    return int(_env(key, default))


def _env_float(key: str, default: str) -> float:
    return float(_env(key, default))


def _env_decimal(key: str, default: str) -> Decimal:
    return Decimal(_env(key, default))


# ── Trading Window & Order Config ────────────────────────────────────────────

@dataclass(frozen=True)
class TradingConfig:
    """Immutable trading window and order parameters."""
    trade_window_start_sec: int = _env_int("TRADE_WINDOW_START", "60")
    trade_window_end_sec: int = _env_int("TRADE_WINDOW_END", "180")
    position_size_usd: Decimal = _env_decimal("MARKET_BUY_USD", "1.00")
    use_limit_orders: bool = _env_bool("USE_LIMIT_ORDERS", "true")
    limit_order_offset: Decimal = _env_decimal("LIMIT_ORDER_OFFSET", "0.01")
    max_retries_per_window: int = _env_int("MAX_RETRIES_PER_WINDOW", "3")
    market_interval_seconds: int = 900
    quote_stability_required: int = 3
    quote_min_spread: float = 0.001


# ── Fusion / Signal Config ───────────────────────────────────────────────────

@dataclass(frozen=True)
class FusionConfig:
    """Signal fusion thresholds."""
    min_fusion_signals: int = _env_int("MIN_FUSION_SIGNALS", "2")
    min_fusion_score: float = _env_float("MIN_FUSION_SCORE", "55.0")
    trend_up_threshold: float = _env_float("TREND_UP_THRESHOLD", "0.60")
    trend_down_threshold: float = _env_float("TREND_DOWN_THRESHOLD", "0.40")
    spike_threshold: float = _env_float("SPIKE_THRESHOLD", "0.05")
    divergence_threshold: float = _env_float("DIVERGENCE_THRESHOLD", "0.05")


# ── V3 Quant Config ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QuantConfig:
    """Binary option pricing model parameters."""
    min_edge_cents: float = _env_float("MIN_EDGE_CENTS", "0.02")
    take_profit_pct: float = _env_float("TAKE_PROFIT_PCT", "0.30")
    cut_loss_pct: float = _env_float("CUT_LOSS_PCT", "-0.50")
    vol_method: str = _env("VOL_METHOD", "ewma")
    default_vol: float = _env_float("DEFAULT_VOL", "0.65")


# ── V3.1 RTDS + Funding Config ──────────────────────────────────────────────

@dataclass(frozen=True)
class RTDSConfig:
    """RTDS Chainlink oracle and funding-rate filter settings."""
    use_rtds: bool = _env_bool("USE_RTDS", "true")
    late_window_sec: float = _env_float("RTDS_LATE_WINDOW_SEC", "15")
    late_window_min_bps: float = _env_float("RTDS_LATE_WINDOW_MIN_BPS", "3.0")
    divergence_threshold_bps: float = _env_float("RTDS_DIVERGENCE_THRESHOLD_BPS", "5.0")
    use_funding_filter: bool = _env_bool("USE_FUNDING_FILTER", "true")
    late_window_enabled: bool = _env_bool("LATE_WINDOW_ENABLED", "true")


# ── Polymarket API Credentials ───────────────────────────────────────────────

@dataclass(frozen=True)
class PolymarketCreds:
    """Polymarket API credentials (read-only from env)."""
    private_key: str = _env("POLYMARKET_PK", "")
    api_key: str = _env("POLYMARKET_API_KEY", "")
    api_secret: str = _env("POLYMARKET_API_SECRET", "")
    passphrase: str = _env("POLYMARKET_PASSPHRASE", "")
    signature_type: int = 1


# ── Redis Config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RedisConfig:
    """Redis connection settings."""
    host: str = _env("REDIS_HOST", "localhost")
    port: int = _env_int("REDIS_PORT", "6379")
    db: int = _env_int("REDIS_DB", "2")


# ── Top-level aggregate ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class BotConfig:
    """
    Root configuration object — compose all sub-configs.

    Usage:
        cfg = BotConfig()              # loads from env
        print(cfg.trading.position_size_usd)
        print(cfg.quant.vol_method)
    """
    trading: TradingConfig = field(default_factory=TradingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    rtds: RTDSConfig = field(default_factory=RTDSConfig)
    creds: PolymarketCreds = field(default_factory=PolymarketCreds)
    redis: RedisConfig = field(default_factory=RedisConfig)
    restart_after_minutes: int = 90


# Module-level singleton (immutable, safe to share)
_cfg: BotConfig | None = None


def get_config() -> BotConfig:
    """Get the global immutable config. Created once, never mutated."""
    global _cfg
    if _cfg is None:
        _cfg = BotConfig()
    return _cfg

