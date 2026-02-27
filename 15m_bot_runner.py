import subprocess
import time
import sys
import os
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

BOT_SCRIPT = "bot.py"


def _validate_bot_script():
    """Check if bot script exists, exit with diagnostics if not."""
    if not os.path.exists(BOT_SCRIPT):
        print(f"ERROR: Bot script '{BOT_SCRIPT}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print("\nAvailable .py files:")
        for file in os.listdir('.'):
            if file.endswith('.py'):
                print(f"  - {file}")
        print("\nPlease set BOT_SCRIPT to your bot filename")
        sys.exit(1)


def _print_startup_banner(bot_args):
    """Print the startup banner."""
    print("=" * 80)
    print("BTC 15-MIN TRADING BOT - AUTO-RESTART WRAPPER")
    print(f"Bot: {BOT_SCRIPT} {' '.join(bot_args)}")
    print("=" * 80)
    print()


def _handle_bot_exit(exit_code):
    """Handle bot exit code and return wait time before restart."""
    print()
    print("=" * 80)
    print(f"Bot stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit code: {exit_code}")
    print("=" * 80)

    if exit_code in [0, 143, 15, -15]:
        print("✅ Normal auto-restart - loading fresh filters...")
        return 2
    else:
        print(f"⚠️ Error detected (code {exit_code}) - waiting before retry...")
        return 10


def _log_separator(msg):
    """Print separator line with message."""
    print(f"\n{'=' * 80}\n{msg}\n{'=' * 80}\n")


def run_bot():
    """Run the bot with auto-restart using the SAME Python environment."""
    python_cmd = sys.executable
    bot_args = sys.argv[1:] if len(sys.argv) > 1 else []
    _print_startup_banner(bot_args)
    _validate_bot_script()
    restart_count = 0
    while True:
        restart_count += 1
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting bot (#{restart_count})...")
        try:
            result = subprocess.run([python_cmd, BOT_SCRIPT] + bot_args, check=False)
            wait = _handle_bot_exit(result.returncode)
            print(f"Restarting in {wait}s...\n"); time.sleep(wait)
        except KeyboardInterrupt:
            _log_separator("Keyboard interrupt - stopping wrapper"); break
        except Exception as e:
            _log_separator(f"ERROR: {e}"); time.sleep(10)

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("\nStopped by user")
        sys.exit(0)