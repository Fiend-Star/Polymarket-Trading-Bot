"""
Patch for PolymarketExecutionClient to support $1 market buys.

The Polymarket adapter normally requires quote_quantity=True for BUY market orders,
but our strategy sends token quantities (quote_quantity=False).
This patch intercepts BUY market orders and forces them to use the configured
USD amount ($1 default) via the create_market_order API call.
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_patch_applied = False
_DEFAULT_USD_AMOUNT = float(os.getenv("MARKET_BUY_USD", "1.0"))


async def _sign_and_submit_market_order(self, order, amount, instrument, modules):
    """Sign and submit a market order with the given amount."""
    get_token_id, convert_tif, order_side_str, MarketOrderArgs, PartialCreateOrderOptions, LogColor = modules
    order_type = convert_tif(order.time_in_force)
    market_order_args = MarketOrderArgs(
        token_id=get_token_id(order.instrument_id),
        amount=amount, side=order_side_str(order.side), order_type=order_type)
    neg_risk = self._get_neg_risk_for_instrument(instrument)
    options = PartialCreateOrderOptions(neg_risk=neg_risk)
    signing_start = self._clock.timestamp()
    signed_order = await asyncio.to_thread(
        self._http_client.create_market_order, market_order_args, options=options)
    interval = self._clock.timestamp() - signing_start
    return signed_order, interval


async def _patched_submit_market_order(self, command, instrument):
    """Patched market order handler: BUY uses USD amount, SELL uses token qty."""
    from nautilus_trader.adapters.polymarket.common.symbol import get_polymarket_token_id
    from nautilus_trader.adapters.polymarket.http.conversion import convert_tif_to_polymarket_order_type
    from nautilus_trader.model.enums import OrderSide, order_side_to_str
    from nautilus_trader.common.enums import LogColor
    from py_clob_client.client import MarketOrderArgs, PartialCreateOrderOptions

    modules = (get_polymarket_token_id, convert_tif_to_polymarket_order_type,
               order_side_to_str, MarketOrderArgs, PartialCreateOrderOptions, LogColor)
    order = command.order

    if order.side == OrderSide.BUY:
        self._log.info(f"[PATCH] BUY market → ${_DEFAULT_USD_AMOUNT:.2f} USD (qty {float(order.quantity):.6f} ignored)", LogColor.MAGENTA)
        amount = _DEFAULT_USD_AMOUNT
    else:
        if order.is_quote_quantity:
            self._deny_market_order_quantity(
                order, "Polymarket SELL orders require base-denominated quantities"); return
        amount = float(order.quantity)

    signed_order, interval = await _sign_and_submit_market_order(self, order, amount, instrument, modules)
    side_label = "BUY" if order.side == OrderSide.BUY else "SELL"
    self._log.info(f"[PATCH] Signed market {side_label} in {interval:.3f}s", LogColor.BLUE)
    self.generate_order_submitted(
        strategy_id=order.strategy_id, instrument_id=order.instrument_id,
        client_order_id=order.client_order_id, ts_event=self._clock.timestamp_ns())
    await self._post_signed_order(order, signed_order)


def apply_market_order_patch():
    """Apply monkey patch to PolymarketExecutionClient."""
    global _patch_applied
    if _patch_applied:
        logger.info("Market order patch already applied"); return True
    try:
        from nautilus_trader.adapters.polymarket.execution import PolymarketExecutionClient
        logger.info(f"Market BUY USD amount: ${_DEFAULT_USD_AMOUNT:.2f}")
        PolymarketExecutionClient._submit_market_order = _patched_submit_market_order
        _patch_applied = True
        logger.info("Market order patch applied — BUY orders use $MARKET_BUY_USD")
        return True
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}"); return False
    except Exception as e:
        logger.error(f"Failed to apply market order patch: {e}")
        import traceback; traceback.print_exc()
        return False
