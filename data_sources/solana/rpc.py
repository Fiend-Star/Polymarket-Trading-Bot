"""
Solana RPC Data Source
Connects to Solana blockchain for on-chain BTC/crypto data
"""
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
import httpx
from loguru import logger


class SolanaRPCDataSource:
    """
    Solana RPC data source.
    
    Note: While Solana doesn't have native BTC, it can provide:
    - Wrapped BTC (wBTC/renBTC) on-chain data
    - DEX price feeds
    - Oracle data (Pyth Network)
    - Transaction volume metrics
    """
    
    def __init__(
        self,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        use_pyth: bool = True,
    ):
        """
        Initialize Solana RPC source.
        
        Args:
            rpc_url: Solana RPC endpoint
            use_pyth: Whether to use Pyth Network for price data
        """
        self.rpc_url = rpc_url
        self.use_pyth = use_pyth
        self.session: Optional[httpx.AsyncClient] = None
        
        # Pyth BTC/USD price feed address (mainnet)
        self.pyth_btc_address = "GVXRSBjFk6e6J3NbVPXohDJetcTjaeeuykUpbQF8UoMU"
        
        # Cache
        self._last_price: Optional[Decimal] = None
        self._last_update: Optional[datetime] = None
        
        logger.info("Initialized Solana RPC data source")
    
    async def _rpc_call(self, method, params=None):
        """Make a JSON-RPC call. Returns result value or None."""
        payload = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params:
            payload["params"] = params
        resp = await self.session.post(self.rpc_url, json=payload)
        resp.raise_for_status()
        return resp.json().get("result")

    async def connect(self) -> bool:
        """Connect to Solana RPC."""
        try:
            self.session = httpx.AsyncClient(timeout=30.0, headers={"Content-Type": "application/json"})
            result = await self._rpc_call("getSlot")
            if result is not None:
                logger.info(f"✓ Connected to Solana RPC (Slot: {result})")
                return True
            logger.error("Solana RPC unexpected response"); return False
        except Exception as e:
            logger.error(f"Solana connect failed: {e}"); return False

    async def disconnect(self) -> None:
        """Close connection."""
        if self.session:
            await self.session.aclose()
            logger.info("Disconnected from Solana RPC")

    async def get_pyth_price(self) -> Optional[Decimal]:
        """Get BTC price from Pyth Network oracle."""
        if not self.use_pyth:
            return None
        try:
            result = await self._rpc_call("getAccountInfo",
                                           [self.pyth_btc_address, {"encoding": "base64"}])
            if result and result.get("value"):
                logger.debug("Fetched Pyth data (requires Pyth SDK to parse)")
            return None  # Placeholder - needs Pyth SDK
        except Exception as e:
            logger.error(f"Pyth price error: {e}"); return None

    async def get_slot(self) -> Optional[int]:
        """Get current slot number."""
        try:
            return await self._rpc_call("getSlot")
        except Exception as e:
            logger.error(f"Slot error: {e}"); return None

    async def get_block_time(self, slot: int) -> Optional[datetime]:
        """Get block timestamp for a slot."""
        try:
            ts = await self._rpc_call("getBlockTime", [slot])
            return datetime.fromtimestamp(ts) if ts else None
        except Exception as e:
            logger.error(f"Block time error: {e}"); return None

    async def get_token_supply(self, token_mint: str) -> Optional[Dict[str, Any]]:
        """Get token supply information."""
        try:
            result = await self._rpc_call("getTokenSupply", [token_mint])
            if result and "value" in result:
                v = result["value"]
                return {"amount": v["amount"], "decimals": v["decimals"], "ui_amount": v["uiAmount"]}
            return None
        except Exception as e:
            logger.error(f"Token supply error: {e}"); return None

    async def get_network_stats(self) -> Optional[Dict[str, Any]]:
        """Get Solana network statistics (slot, TPS, etc.)."""
        try:
            slot = await self.get_slot()
            if not slot: return None
            bt = await self.get_block_time(slot)
            result = await self._rpc_call("getRecentPerformanceSamples", [1])
            if result and len(result) > 0:
                p = result[0]
                return {"timestamp": bt or datetime.now(), "current_slot": slot,
                        "num_transactions": p.get("numTransactions"),
                        "sample_period_secs": p.get("samplePeriodSecs"),
                        "tps": p.get("numTransactions", 0) / max(p.get("samplePeriodSecs", 1), 1)}
            return None
        except Exception as e:
            logger.error(f"Network stats error: {e}"); return None

    @property
    def last_price(self) -> Optional[Decimal]:
        """Get cached last price."""
        return self._last_price
    
    @property
    def last_update(self) -> Optional[datetime]:
        """Get time of last price update."""
        return self._last_update
    
    async def health_check(self) -> bool:
        """
        Check if data source is healthy.
        
        Returns:
            True if healthy
        """
        try:
            slot = await self.get_slot()
            return slot is not None
        except:
            return False


# Singleton instance
_solana_instance: Optional[SolanaRPCDataSource] = None

def get_solana_source() -> SolanaRPCDataSource:
    """Get singleton instance of Solana RPC data source."""
    global _solana_instance
    if _solana_instance is None:
        _solana_instance = SolanaRPCDataSource()
    return _solana_instance