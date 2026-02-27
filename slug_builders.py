from datetime import UTC, datetime


def build_btc_15min_slugs() -> list[str]:
    """
    Build slugs for BTC 15-minute UpDown markets.

    Aligns to proper 15-min boundaries (unix // 900 * 900) and includes
    1 prior interval + 24 hours of future markets.
    """
    now = datetime.now(tz=UTC)
    unix_interval_start = (int(now.timestamp()) // 900) * 900  # current 15-min boundary

    slugs = []
    for i in range(-1, 97):  # 1 prior + current + 96 future (~24 hours)
        timestamp = unix_interval_start + (i * 900)
        slugs.append(f"btc-updown-15m-{timestamp}")
    return slugs