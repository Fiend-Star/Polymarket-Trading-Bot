"""
Redis Control Script for BTC Bot Simulation Mode
Toggle between simulation and live trading without restarting
"""
import redis
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def get_redis_client():
    """Get Redis client."""
    try:
        client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 2)),
            decode_responses=True,
            socket_connect_timeout=5
        )
        client.ping()
        return client
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        print(f"  Make sure Redis is running: redis-server")
        return None


def get_current_mode(client):
    """Get current simulation mode."""
    try:
        mode = client.get('btc_trading:simulation_mode')
        if mode is None:
            return None
        return mode == '1'
    except Exception as e:
        print(f"✗ Error reading mode: {e}")
        return None


def set_simulation_mode(client, simulation: bool):
    """Set simulation mode."""
    try:
        client.set('btc_trading:simulation_mode', '1' if simulation else '0')
        mode_text = "SIMULATION" if simulation else "LIVE TRADING"
        print(f"✓ Mode set to: {mode_text}")
        return True
    except Exception as e:
        print(f"✗ Error setting mode: {e}")
        return False


def display_status(client):
    """Display current status."""
    mode = get_current_mode(client)
    
    print("\n" + "=" * 60)
    print("BTC BOT - CURRENT STATUS")
    print("=" * 60)
    
    if mode is None:
        print("Status: ⚪ Not set (using default from .env)")
    elif mode:
        print("Status: 🟡 SIMULATION MODE")
        print("  - No real trades will be placed")
        print("  - Safe for testing")
    else:
        print("Status: 🔴 LIVE TRADING MODE")
        print("  - REAL MONEY AT RISK!")
        print("  - Real orders will be placed")
    
    print("=" * 60 + "\n")


def _handle_cli_command(client, command):
    """Handle a single CLI command argument."""
    if command in ['sim', 'simulation', 'on']:
        print("Switching to SIMULATION mode...")
        set_simulation_mode(client, True)
        display_status(client)

    elif command in ['live', 'off']:
        print("\n⚠️  WARNING: Switching to LIVE TRADING mode!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            set_simulation_mode(client, False)
            display_status(client)
        else:
            print("Cancelled.")

    elif command in ['status', 'check']:
        pass  # Already displayed

    else:
        print(f"Unknown command: {command}")
        print("\nUsage:")
        print("  python redis_control.py sim       - Enable simulation mode")
        print("  python redis_control.py live      - Enable live trading")
        print("  python redis_control.py status    - Show current status")


def _confirm_live_switch(client):
    """Prompt user to confirm live trading switch."""
    print("\n⚠️  WARNING: This will enable LIVE TRADING!")
    confirm = input("Type 'yes' to confirm: ")
    if confirm.lower() == 'yes':
        set_simulation_mode(client, False)
        display_status(client)
    else:
        print("Cancelled.")


def _interactive_menu_loop(client):
    """Run the interactive menu loop."""
    print("Commands:")
    print("  1. Enable simulation mode")
    print("  2. Enable live trading (⚠️ DANGEROUS!)")
    print("  3. Check status")
    print("  4. Exit")

    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()

            if choice == '1':
                set_simulation_mode(client, True)
                display_status(client)
            elif choice == '2':
                _confirm_live_switch(client)
            elif choice == '3':
                display_status(client)
            elif choice == '4':
                print("Goodbye!")
                break
            else:
                print("Invalid choice!")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    """Main control interface."""
    print("\n" + "=" * 60)
    print("BTC BOT - SIMULATION MODE CONTROL")
    print("=" * 60)

    client = get_redis_client()
    if not client:
        return

    display_status(client)

    if len(sys.argv) > 1:
        _handle_cli_command(client, sys.argv[1].lower())
    else:
        _interactive_menu_loop(client)


if __name__ == "__main__":
    main()