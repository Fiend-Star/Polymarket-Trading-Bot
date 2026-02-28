"""
BTC Instrument Definitions for NautilusTrader
"""
from decimal import Decimal
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.currencies import USDC, BTC
from loguru import logger


def _create_btc_instrument(symbol_str, venue_str, price_prec=2, size_prec=8,
                           price_inc="0.01", size_inc="0.00000001",
                           max_qty="1000", min_qty="0.001",
                           max_price="1000000.00", min_price="1.00",
                           margin_init="0.05", margin_maint="0.03",
                           maker_fee="0.001", taker_fee="0.002") -> CryptoPerpetual:
    """Create a BTC CryptoPerpetual instrument with given parameters."""
    inst = CryptoPerpetual(
        instrument_id=InstrumentId(symbol=Symbol(symbol_str), venue=Venue(venue_str)),
        raw_symbol=Symbol(symbol_str), base_currency=BTC,
        quote_currency=USDC, settlement_currency=USDC, is_inverse=False,
        price_precision=price_prec, size_precision=size_prec,
        price_increment=Price.from_str(price_inc), size_increment=Quantity.from_str(size_inc),
        max_quantity=Quantity.from_str(max_qty), min_quantity=Quantity.from_str(min_qty),
        max_price=Price.from_str(max_price), min_price=Price.from_str(min_price),
        margin_init=Decimal(margin_init), margin_maint=Decimal(margin_maint),
        maker_fee=Decimal(maker_fee), taker_fee=Decimal(taker_fee),
        ts_event=0, ts_init=0)
    logger.info(f"Created instrument: {inst.id}")
    return inst


def create_btc_polymarket_instrument() -> CryptoPerpetual:
    """Create BTC prediction market instrument for Polymarket."""
    return _create_btc_instrument(
        "BTC-POLYMARKET", "POLYMARKET", size_prec=4, size_inc="0.0001",
        max_qty="1000000", min_qty="0.01", max_price="1.00", min_price="0.00")


def create_btc_spot_instrument() -> CryptoPerpetual:
    """Create BTC spot price instrument (Coinbase reference)."""
    return _create_btc_instrument(
        "BTC-USD", "COINBASE", maker_fee="0.005", taker_fee="0.005")


def create_btc_binance_instrument() -> CryptoPerpetual:
    """Create Binance BTC instrument."""
    return _create_btc_instrument(
        "BTCUSDT", "BINANCE", max_qty="9000", min_qty="0.00001",
        margin_init="0.01", margin_maint="0.005", taker_fee="0.001")


class InstrumentRegistry:
    """Registry for all trading instruments."""
    
    def __init__(self):
        """Initialize instrument registry."""
        self.instruments = {}
        self._setup_instruments()
        
        logger.info(f"Initialized instrument registry with {len(self.instruments)} instruments")
    
    def _setup_instruments(self) -> None:
        """Setup all instruments."""
        # Polymarket prediction market
        polymarket = create_btc_polymarket_instrument()
        self.instruments[str(polymarket.id)] = polymarket
        
        # Spot reference instruments
        coinbase = create_btc_spot_instrument()
        self.instruments[str(coinbase.id)] = coinbase
        
        binance = create_btc_binance_instrument()
        self.instruments[str(binance.id)] = binance
    
    def get(self, instrument_id: str) -> CryptoPerpetual:
        """Get instrument by ID."""
        return self.instruments.get(instrument_id)
    
    def get_polymarket(self) -> CryptoPerpetual:
        """Get Polymarket BTC instrument."""
        return self.get("BTC-POLYMARKET.POLYMARKET")
    
    def get_coinbase(self) -> CryptoPerpetual:
        """Get Coinbase BTC instrument."""
        return self.get("BTC-USD.COINBASE")
    
    def get_binance(self) -> CryptoPerpetual:
        """Get Binance BTC instrument."""
        return self.get("BTCUSDT.BINANCE")
    
    def get_all(self) -> list:
        """Get all instruments."""
        return list(self.instruments.values())


# Singleton instance
_registry_instance = None

def get_instrument_registry() -> InstrumentRegistry:
    """Get singleton instrument registry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = InstrumentRegistry()
    return _registry_instance