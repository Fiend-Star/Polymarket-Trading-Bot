# In a separate file, e.g., slug_builders.py
from datetime import UTC, datetime, timedelta

def build_btc_15min_slugs() -> list[str]:
    """Build slugs for BTC 15-minute UpDown markets."""
    slugs = []
    now = datetime.now(tz=UTC)
    # Generate slugs for upcoming 15-min windows
    for i in range(10):  # next ~2.5 hours of markets
        target = now + timedelta(minutes=15 * i)
        # Adjust slug format to match Polymarket's naming convention
        slug = f"btc-updown-15m-{int(target.timestamp())}"
        slugs.append(slug)
    return slugs