from py_clob_client.client import ClobClient
import os
from dotenv import load_dotenv

load_dotenv()


def _normalize_private_key(value: str) -> str:
    return value[2:] if value.startswith("0x") else value


def main() -> int:
    private_key = os.getenv("POLYMARKET_PK")
    if not private_key:
        print("Missing POLYMARKET_PK in environment (.env).")
        return 1

    private_key = _normalize_private_key(private_key.strip())

    client = ClobClient("https://clob.polymarket.com", key=private_key, chain_id=137)
    creds = client.create_or_derive_api_creds()
    print(creds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
