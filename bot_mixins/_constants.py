"""
Shared constants for bot mixins — derived from BotConfig.

Single source of truth: config.py. This module exposes backward-compat
module-level names so mixins can ``from bot_mixins._constants import X``.
"""
from config import get_config

_CFG = get_config()

# Trading
QUOTE_STABILITY_REQUIRED = _CFG.trading.quote_stability_required
QUOTE_MIN_SPREAD = _CFG.trading.quote_min_spread
MARKET_INTERVAL_SECONDS = _CFG.trading.market_interval_seconds
TRADE_WINDOW_START_SEC = _CFG.trading.trade_window_start_sec
TRADE_WINDOW_END_SEC = _CFG.trading.trade_window_end_sec
POSITION_SIZE_USD = _CFG.trading.position_size_usd
USE_LIMIT_ORDERS = _CFG.trading.use_limit_orders
LIMIT_ORDER_OFFSET = _CFG.trading.limit_order_offset
MAX_RETRIES_PER_WINDOW = _CFG.trading.max_retries_per_window

# Fusion
MIN_FUSION_SIGNALS = _CFG.fusion.min_fusion_signals
MIN_FUSION_SCORE = _CFG.fusion.min_fusion_score
TREND_UP_THRESHOLD = _CFG.fusion.trend_up_threshold
TREND_DOWN_THRESHOLD = _CFG.fusion.trend_down_threshold

# Quant
MIN_EDGE_CENTS = _CFG.quant.min_edge_cents
TAKE_PROFIT_PCT = _CFG.quant.take_profit_pct
CUT_LOSS_PCT = _CFG.quant.cut_loss_pct
VOL_METHOD = _CFG.quant.vol_method
DEFAULT_VOL = _CFG.quant.default_vol

# RTDS
USE_RTDS = _CFG.rtds.use_rtds
RTDS_LATE_WINDOW_SEC = _CFG.rtds.late_window_sec
RTDS_LATE_WINDOW_MIN_BPS = _CFG.rtds.late_window_min_bps
RTDS_DIVERGENCE_THRESHOLD_BPS = _CFG.rtds.divergence_threshold_bps
USE_FUNDING_FILTER = _CFG.rtds.use_funding_filter
LATE_WINDOW_ENABLED = _CFG.rtds.late_window_enabled

