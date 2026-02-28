"""
Backtest fusion confirmation — mirrors V3 bot signal processors.

SRP: Signal confirmation via old fusion processors.
"""
import math
import os
from datetime import timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from loguru import logger
from backtest_models import Candle
from backtest_sim import compute_rsi

try:
    from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
    from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
    from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
    from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False


def create_fusion_processors() -> Optional[Dict]:
    if not FUSION_AVAILABLE: return None
    return {
        "spike": SpikeDetectionProcessor(
            spike_threshold=float(os.getenv("SPIKE_THRESHOLD", "0.05")), lookback_periods=20),
        "sentiment": SentimentProcessor(extreme_fear_threshold=25, extreme_greed_threshold=75),
        "divergence": PriceDivergenceProcessor(
            divergence_threshold=float(os.getenv("DIVERGENCE_THRESHOLD", "0.05"))),
        "tick_velocity": TickVelocityProcessor(
            velocity_threshold_60s=0.015, velocity_threshold_30s=0.010),
    }


def _collect_signals(processors, current_price, price_history, pm):
    signals = []
    for name in ["spike", "sentiment", "divergence", "tick_velocity"]:
        proc = processors.get(name)
        if proc is None: continue
        try:
            sig = proc.process(current_price=current_price, historical_prices=price_history, metadata=pm)
            if sig: signals.append(sig)
        except Exception: pass
    return signals


def _count_confirmations(signals, model_bullish):
    conf = contra = 0
    for sig in signals:
        if ("BULLISH" in str(sig.direction).upper()) == model_bullish:
            conf += 1
        else:
            contra += 1
    return conf, contra


def run_confirmation(processors, current_price, price_history, metadata, model_bullish):
    if processors is None: return (0, 0)
    pm = {k: (Decimal(str(v)) if isinstance(v, float) else v) for k, v in metadata.items()}
    signals = _collect_signals(processors, current_price, price_history, pm)
    return _count_confirmations(signals, model_bullish)


def _build_tick_buffer(window_candles, dm):
    buf = []
    start = max(0, dm - 2)
    for i in range(start, min(dm + 1, len(window_candles))):
        c = window_candles[i]
        for j, px in enumerate([c.open, (c.open + c.close) / 2, c.close]):
            buf.append({"ts": c.timestamp + timedelta(seconds=j * 20),
                        "price": Decimal(str(round(px, 4)))})
    return buf


def _classify_sentiment(score):
    if score < 25: return "Extreme Fear"
    elif score < 45: return "Fear"
    elif score < 55: return "Neutral"
    elif score < 75: return "Greed"
    return "Extreme Greed"


def build_confirmation_metadata(window_candles, dm, historical_btc_closes):
    if len(window_candles) <= dm: return {}
    cc = window_candles[dm].close
    recent = historical_btc_closes[-20:]
    sma = sum(recent) / len(recent) if recent else cc
    dev = (cc - sma) / sma if sma != 0 else 0.0
    p5 = historical_btc_closes[-5] if len(historical_btc_closes) >= 5 else cc
    mom = (cc - p5) / p5 if p5 != 0 else 0.0
    var = sum((p - sma)**2 for p in recent) / len(recent) if recent else 0.0
    rsi = compute_rsi(historical_btc_closes, period=14)
    return {
        "deviation": dev, "momentum": mom,
        "volatility": math.sqrt(var), "sentiment_score": rsi,
        "sentiment_classification": _classify_sentiment(rsi),
        "spot_price": cc,
        "tick_buffer": _build_tick_buffer(window_candles, dm),
        "yes_token_id": None,
    }

