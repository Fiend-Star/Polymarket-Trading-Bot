"""
Helper functions for interacting with Polymarket Gamma API and redemption.

New in V3.2:
- fetch_price_to_beat: Get authoritative strike price from Gamma API
- fetch_condition_id: Get condition ID for market redemption
- auto_redeem_winnings: Automatic gasless redemption of winnings
"""
import os
import time
from typing import Optional
import requests
from loguru import logger

try:
    from polymarket_apis import PolymarketGaslessWeb3Client
except ImportError:
    PolymarketGaslessWeb3Client = None

# Persistent HTTP session for Gamma API calls (reuses TCP/TLS connections)
_gamma_session: Optional[requests.Session] = None


def _get_gamma_session() -> requests.Session:
    """Return a module-level persistent requests.Session for Gamma API."""
    global _gamma_session
    if _gamma_session is None:
        _gamma_session = requests.Session()
        _gamma_session.headers.update({"Accept": "application/json"})
    return _gamma_session


def fetch_price_to_beat(slug: str, timeout: float = 10.0, max_retries: int = 3) -> Optional[float]:
    """
    Fetch the authoritative 'priceToBeat' for a BTC 15-min UpDown market.

    Polymarket stores the exact Chainlink BTC/USD price at eventStartTime in
    the event metadata field ``priceToBeat``.  This is the REAL strike that
    determines resolution — NOT the Chainlink price we capture ourselves.

    Path: GET /markets?slug=<slug>  →  events[0].eventMetadata.priceToBeat
    """
    session = _get_gamma_session()
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"slug": slug},
                timeout=timeout,
            )
            if r.status_code != 200:
                logger.debug(f"Gamma API returned {r.status_code} for slug={slug} (attempt {attempt})")
                # Retry on server errors / non-200 up to max_retries
                if attempt < max_retries:
                    logger.info(f"Retrying fetch_price_to_beat in {backoff}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return None

            data = r.json()
            if not data:
                return None

            market = data[0]
            events = market.get("events", [])
            if not events:
                return None

            metadata = events[0].get("eventMetadata")
            if not metadata or not isinstance(metadata, dict):
                return None

            ptb = metadata.get("priceToBeat")
            if ptb is not None:
                return float(ptb)

            return None
        except Exception as e:
            # Handle request-level failures with exponential backoff
            logger.debug(f"fetch_price_to_beat({slug}) attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"fetch_price_to_beat retry in {backoff}s (attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                return None


def fetch_condition_id(slug: str, timeout: float = 10.0) -> Optional[str]:
    """Fetch the specific condition ID needed to redeem a market."""
    session = _get_gamma_session()
    try:
        r = session.get(
            "https://gamma-api.polymarket.com/markets",
            params={"slug": slug},
            timeout=timeout,
        )
        if r.status_code == 200:
            data = r.json()
            if data and len(data) > 0:
                return data[0].get("conditionId")
    except Exception as e:
        logger.error(f"Failed to fetch condition ID for {slug}: {e}")
    return None


def auto_redeem_winnings(slug: str, direction: str, amount_won: float):
    """Redeems winning shares for USDC via gasless proxy transaction."""
    if not PolymarketGaslessWeb3Client:
        logger.error("polymarket-apis not installed. Cannot auto-redeem.")
        return

    private_key = os.getenv("POLYMARKET_PK")
    proxy_address = os.getenv("POLYMARKET_FUNDER")

    if not private_key or not proxy_address:
        logger.error("Missing POLYMARKET_PK or POLYMARKET_FUNDER in .env")
        return

    logger.info(f"🔄 Fetching condition ID to auto-redeem winnings for {slug}...")
    condition_id = fetch_condition_id(slug)

    if not condition_id:
        logger.error("Could not find condition_id. Redemption aborted.")
        return

    try:
        from polymarket_apis.types.clob_types import ApiCreds
        api_key = os.getenv("POLYMARKET_API_KEY")
        api_secret = os.getenv("POLYMARKET_API_SECRET")
        api_passphrase = os.getenv("POLYMARKET_PASSPHRASE")

        creds = None
        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(
                key=api_key,
                secret=api_secret,
                passphrase=api_passphrase
            )

        client = PolymarketGaslessWeb3Client(
            private_key=private_key,
            signature_type=1,
            builder_creds=creds
        )

        # direction: "long" = Bought YES (index 0), "short" = Bought NO (index 1)
        amounts = [float(amount_won), 0.0] if direction == "long" else [0.0, float(amount_won)]

        receipt = client.redeem_position(
            condition_id=condition_id,
            amounts=amounts,
            neg_risk=False
        )
        logger.info(f"💰 Successfully Redeemed {slug}! Tx Hash: {receipt}")
    except Exception as e:
        logger.error(f"Failed to auto-redeem winnings: {e}")
