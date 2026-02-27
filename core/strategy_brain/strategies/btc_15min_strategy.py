"""
15-Minute BTC Trading Strategy
Main strategy that coordinates signal processing and trading decisions
"""
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import deque
from loguru import logger
import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine, FusedSignal
from core.strategy_brain.signal_processors.base_processor import SignalDirection


class BTCStrategy15Min:
    """
    15-minute BTC trading strategy.
    
    Workflow:
    1. Collect price data every 15 minutes
    2. Run all signal processors
    3. Fuse signals into consensus
    4. Make trading decision
    5. Manage positions
    """
    
    def _init_processors(self):
        """Initialize signal processors and fusion engine."""
        self.spike_detector = SpikeDetectionProcessor(spike_threshold=0.15, lookback_periods=20)
        self.sentiment_processor = SentimentProcessor(extreme_fear_threshold=25, extreme_greed_threshold=75)
        self.divergence_processor = PriceDivergenceProcessor(divergence_threshold=0.05)
        self.fusion_engine = get_fusion_engine()

    def _init_state(self):
        """Initialize price history, position tracking, and statistics."""
        self.price_history = deque(maxlen=100)
        self._current_price: Optional[Decimal] = None
        self._spot_price_consensus: Optional[Decimal] = None
        self._sentiment_score: Optional[float] = None
        self.open_positions: List[Dict[str, Any]] = []
        self._is_running = False
        self._last_decision_time: Optional[datetime] = None
        self._signals_processed = self._trades_executed = 0
        self._total_pnl = Decimal("0")

    def __init__(self, max_position_size: Decimal = Decimal("10.0"),
                 stop_loss_pct: float = 0.30, take_profit_pct: float = 0.20,
                 max_positions: int = 2):
        """Initialize 15-minute BTC strategy."""
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self._init_processors()
        self._init_state()
        logger.info(f"15-Min Strategy: max=${max_position_size}, "
                    f"SL={stop_loss_pct:.0%}, TP={take_profit_pct:.0%}")

    async def start(self) -> None:
        """Start the strategy."""
        if self._is_running:
            logger.warning("Strategy already running")
            return
        
        self._is_running = True
        logger.info("Strategy started")
        
        # Start decision loop
        asyncio.create_task(self._decision_loop())
    
    async def stop(self) -> None:
        """Stop the strategy."""
        self._is_running = False
        logger.info("Strategy stopped")
    
    def update_market_data(
        self,
        price: Decimal,
        spot_consensus: Optional[Decimal] = None,
        sentiment: Optional[float] = None,
    ) -> None:
        """
        Update current market data.
        
        Args:
            price: Current Polymarket price
            spot_consensus: Spot exchange consensus price
            sentiment: Fear & Greed sentiment score (0-100)
        """
        self._current_price = price
        self.price_history.append(price)
        
        if spot_consensus:
            self._spot_price_consensus = spot_consensus
        
        if sentiment is not None:
            self._sentiment_score = sentiment
    
    async def _decision_loop(self) -> None:
        """
        Main decision loop - runs every 15 minutes.
        """
        while self._is_running:
            try:
                # Wait until next 15-minute mark
                await self._wait_for_next_interval()
                
                # Make trading decision
                await self._make_decision()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in decision loop: {e}")
                await asyncio.sleep(60)
    
    async def _wait_for_next_interval(self) -> None:
        """Wait until next 15-minute interval."""
        now = datetime.now()
        
        # Calculate next 15-minute mark
        minutes_past = now.minute % 15
        if minutes_past == 0:
            # Already at 15-min mark, wait 15 minutes
            wait_minutes = 15
        else:
            # Wait until next 15-min mark
            wait_minutes = 15 - minutes_past
        
        next_time = now + timedelta(minutes=wait_minutes)
        next_time = next_time.replace(second=0, microsecond=0)
        
        wait_seconds = (next_time - now).total_seconds()
        
        logger.info(f"Waiting {wait_seconds:.0f}s until next decision ({next_time.strftime('%H:%M')})")
        
        await asyncio.sleep(wait_seconds)
    
    async def _make_decision(self) -> None:
        """Make trading decision based on fused signals."""
        logger.info("=" * 60 + " MAKING DECISION " + "=" * 60)
        if not self._current_price:
            logger.warning("No current price data"); return
        signals = self._process_signals()
        if not signals:
            logger.info("No signals generated"); return
        for sig in signals:
            logger.info(f"  [{sig.source}] {sig.direction.value}: "
                       f"score={sig.score:.1f}, conf={sig.confidence:.2%}")
        fused = self.fusion_engine.fuse_signals(signals, min_signals=1, min_score=60.0)
        if not fused or not fused.is_actionable:
            logger.info("No actionable fused signal"); return
        if len(self.open_positions) >= self.max_positions:
            logger.warning(f"Max positions ({self.max_positions})"); return
        await self._execute_trade(fused)
        self._last_decision_time = datetime.now()
    
    def _run_processor(self, processor, metadata, require_spot=False):
        """Run a single signal processor. Returns signal or None."""
        if require_spot and not self._spot_price_consensus:
            return None
        return processor.process(
            current_price=self._current_price,
            historical_prices=list(self.price_history), metadata=metadata)

    def _process_signals(self) -> List:
        """Run all signal processors and return generated signals."""
        if len(self.price_history) < 20:
            return []
        meta = {}
        if self._spot_price_consensus:
            meta['spot_price'] = float(self._spot_price_consensus)
        if self._sentiment_score is not None:
            meta['sentiment_score'] = self._sentiment_score
        results = [
            self._run_processor(self.spike_detector, meta),
            self._run_processor(self.sentiment_processor, meta),
            self._run_processor(self.divergence_processor, meta, require_spot=True)]
        signals = [s for s in results if s is not None]
        self._signals_processed += len(signals)
        return signals
    
    def _compute_sl_tp(self, direction, entry_price):
        """Compute stop loss and take profit prices."""
        if direction == SignalDirection.BULLISH:
            sl = entry_price * Decimal(str(1 - self.stop_loss_pct))
            tp = entry_price * Decimal(str(1 + self.take_profit_pct))
        else:
            sl = entry_price * Decimal(str(1 + self.stop_loss_pct))
            tp = entry_price * Decimal(str(1 - self.take_profit_pct))
        return sl, tp

    async def _execute_trade(self, signal: FusedSignal) -> None:
        """Execute trade based on fused signal."""
        entry = self._current_price
        sl, tp = self._compute_sl_tp(signal.direction, entry)
        size = min(self.max_position_size, Decimal("10.0"))
        position = {
            "id": f"pos_{datetime.now().timestamp()}", "direction": signal.direction.value,
            "entry_price": entry, "size": size, "stop_loss": sl, "take_profit": tp,
            "entry_time": datetime.now(), "signal_score": signal.score, "status": "open"}
        self.open_positions.append(position)
        self._trades_executed += 1
        logger.info(f"Trade: {signal.direction.value} entry=${float(entry):,.2f} "
                    f"size=${float(size):,.2f} SL=${float(sl):,.2f} TP=${float(tp):,.2f}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            "is_running": self._is_running,
            "signals_processed": self._signals_processed,
            "trades_executed": self._trades_executed,
            "open_positions": len(self.open_positions),
            "total_pnl": float(self._total_pnl),
            "last_decision": self._last_decision_time.isoformat() if self._last_decision_time else None,
            "processors": {
                "spike_detector": self.spike_detector.get_stats(),
                "sentiment": self.sentiment_processor.get_stats(),
                "divergence": self.divergence_processor.get_stats(),
            },
            "fusion_engine": self.fusion_engine.get_statistics(),
        }


# Singleton instance
_strategy_instance = None

def get_btc_strategy() -> BTCStrategy15Min:
    """Get singleton strategy instance."""
    global _strategy_instance
    if _strategy_instance is None:
        _strategy_instance = BTCStrategy15Min()
    return _strategy_instance





