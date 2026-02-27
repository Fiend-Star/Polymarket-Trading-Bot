"""
Data Validator
Validates incoming market data for quality and anomalies
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class ValidationRule:
    """Data validation rule."""
    name: str
    min_value: Optional[Decimal] = None
    max_value: Optional[Decimal] = None
    max_change_percent: Optional[float] = None
    required: bool = True


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class DataValidator:
    """
    Validates market data for:
    - Price sanity checks
    - Anomaly detection
    - Data completeness
    - Timestamp validation
    """
    
    def __init__(self):
        """Initialize data validator."""
        # Price history for anomaly detection
        self._price_history: Dict[str, List[Decimal]] = {}
        self._max_history_size = 100
        
        # Validation rules
        self.btc_rules = {
            "price": ValidationRule(
                name="BTC Price",
                min_value=Decimal("1000"),  # BTC never below $1k (sanity check)
                max_value=Decimal("1000000"),  # BTC never above $1M (yet!)
                max_change_percent=20.0,  # Max 20% change between updates
            ),
            "volume": ValidationRule(
                name="Volume",
                min_value=Decimal("0"),
                required=False,
            ),
        }
        
        logger.info("Initialized Data Validator")
    
    def _validate_price_range(self, price, errors):
        """Check price is within configured bounds."""
        rule = self.btc_rules["price"]
        if price < rule.min_value:
            errors.append(f"Price ${price:,.2f} below minimum ${rule.min_value:,.2f}")
        if price > rule.max_value:
            errors.append(f"Price ${price:,.2f} above maximum ${rule.max_value:,.2f}")

    def _validate_timestamp(self, timestamp, warnings, metadata):
        """Check timestamp freshness."""
        diff = abs((datetime.now() - timestamp).total_seconds())
        if diff > 300:
            warnings.append(f"Timestamp is {diff:.0f}s old (stale data)")
            metadata["timestamp_age_seconds"] = diff

    def _validate_price_change(self, source, price, warnings, metadata):
        """Check for anomalous price changes."""
        if source in self._price_history and self._price_history[source]:
            last = self._price_history[source][-1]
            chg = abs((price - last) / last) * 100
            rule = self.btc_rules["price"]
            if chg > rule.max_change_percent:
                warnings.append(f"Large price change: {chg:.2f}% (${last:,.2f}→${price:,.2f})")
                metadata["price_change_percent"] = float(chg)

    def _validate_spread(self, bid, ask, errors, warnings, metadata):
        """Validate bid/ask spread."""
        if not (bid and ask):
            return
        spread_pct = ((ask - bid) / bid) * 100
        if spread_pct > 1.0:
            warnings.append(f"Wide spread: {spread_pct:.2f}% (${ask - bid:,.2f})")
            metadata["spread_percent"] = float(spread_pct)
        if bid > ask:
            errors.append(f"Bid ${bid:,.2f} > Ask ${ask:,.2f} (crossed market)")

    def _update_price_history(self, source, price):
        """Record price in history for anomaly detection."""
        if source not in self._price_history:
            self._price_history[source] = []
        self._price_history[source].append(price)
        if len(self._price_history[source]) > self._max_history_size:
            self._price_history[source].pop(0)

    def validate_market_data(self, source: str, price: Decimal, timestamp: datetime,
                             volume: Optional[Decimal] = None,
                             bid: Optional[Decimal] = None,
                             ask: Optional[Decimal] = None) -> ValidationResult:
        """Validate market data and return ValidationResult."""
        errors, warnings, metadata = [], [], {}
        self._validate_price_range(price, errors)
        self._validate_timestamp(timestamp, warnings, metadata)
        self._validate_price_change(source, price, warnings, metadata)
        self._validate_spread(bid, ask, errors, warnings, metadata)
        if volume is not None and volume < self.btc_rules["volume"].min_value:
            errors.append(f"Negative volume: ${volume:,.2f}")
        self._update_price_history(source, price)
        if errors:
            logger.error(f"Validation FAILED for {source}: {errors}")
        if warnings:
            logger.warning(f"Validation warnings for {source}: {warnings}")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors,
                               warnings=warnings, metadata=metadata)

    def validate_sentiment_data(self, score: float, timestamp: datetime) -> ValidationResult:
        """Validate sentiment data (score 0-100, timestamp freshness)."""
        errors, warnings = [], []
        if score < 0 or score > 100:
            errors.append(f"Sentiment score {score} out of range [0-100]")
        age = abs((datetime.now() - timestamp).total_seconds())
        if age > 3600:
            warnings.append(f"Sentiment data is {age/3600:.1f}h old")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors,
                                warnings=warnings, metadata={})

    def detect_anomaly(self, source: str, current_price: Decimal,
                       z_score_threshold: float = 3.0) -> Optional[Dict[str, Any]]:
        """Detect price anomalies using Z-score. Returns anomaly dict or None."""
        if source not in self._price_history:
            return None
        history = self._price_history[source]
        if len(history) < 10:
            return None
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std_dev = Decimal(str(float(variance) ** 0.5))
        if std_dev == 0:
            return None
        z_score = abs((current_price - mean) / std_dev)

        if float(z_score) > z_score_threshold:
            return {"source": source, "current_price": float(current_price),
                    "mean_price": float(mean), "std_dev": float(std_dev),
                    "z_score": float(z_score), "threshold": z_score_threshold,
                    "anomaly_type": "price_spike" if current_price > mean else "price_drop"}
        return None
    
    def get_price_statistics(self, source: str) -> Optional[Dict[str, Any]]:
        """
        Get price statistics for a source.
        
        Args:
            source: Data source
            
        Returns:
            Statistics dict or None
        """
        if source not in self._price_history or not self._price_history[source]:
            return None
        
        history = self._price_history[source]
        
        mean = sum(history) / len(history)
        min_price = min(history)
        max_price = max(history)
        current_price = history[-1]
        
        return {
            "source": source,
            "count": len(history),
            "current": float(current_price),
            "mean": float(mean),
            "min": float(min_price),
            "max": float(max_price),
            "range": float(max_price - min_price),
            "range_percent": float((max_price - min_price) / min_price * 100),
        }
    
    def clear_history(self, source: Optional[str] = None) -> None:
        """
        Clear price history.
        
        Args:
            source: Specific source to clear, or None for all
        """
        if source:
            if source in self._price_history:
                self._price_history[source].clear()
                logger.info(f"Cleared price history for {source}")
        else:
            self._price_history.clear()
            logger.info("Cleared all price history")


# Singleton instance
_validator_instance: Optional[DataValidator] = None

def get_validator() -> DataValidator:
    """Get singleton instance of data validator."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = DataValidator()
    return _validator_instance