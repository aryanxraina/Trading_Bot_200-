"""
telegram_alerts.py
Sends trade alerts to Telegram. One-message kill switch support.
"""

import requests
from src.utils.config_loader import CONFIG
from src.utils.logger import get_logger

logger = get_logger("telegram")

TOKEN = CONFIG["telegram"]["bot_token"]
CHAT_ID = CONFIG["telegram"]["chat_id"]
ENABLED = CONFIG["telegram"]["send_alerts"]

BASE_URL = f"https://api.telegram.org/bot{TOKEN}"


def send_message(text: str) -> bool:
    if not ENABLED:
        return False
    try:
        resp = requests.post(
            f"{BASE_URL}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=5
        )
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def alert_trade_entry(symbol: str, side: str, price: float, qty: int, stop_loss: float):
    msg = (
        f"🟢 *ENTRY* | {symbol}\n"
        f"Side: {side} | Qty: {qty}\n"
        f"Price: ₹{price:.2f} | SL: ₹{stop_loss:.2f}"
    )
    send_message(msg)


def alert_trade_exit(symbol: str, pnl: float):
    icon = "✅" if pnl >= 0 else "🔴"
    msg = f"{icon} *EXIT* | {symbol}\nP&L: ₹{pnl:.2f}"
    send_message(msg)


def alert_drawdown_halt(reason: str):
    msg = f"🚨 *BOT HALTED* — {reason}"
    send_message(msg)


def alert_daily_summary(total_pnl: float, trades: int, capital: float):
    icon = "📈" if total_pnl >= 0 else "📉"
    msg = (
        f"{icon} *Daily Summary*\n"
        f"P&L: ₹{total_pnl:.2f} | Trades: {trades}\n"
        f"Capital: ₹{capital:.2f}"
    )
    send_message(msg)
