"""
drawdown_monitor.py
Tracks daily and weekly P&L.
Auto-halts the bot if drawdown limits are breached.

Hard rules (non-negotiable per your project doc):
  - -2% daily loss  → halt for the day
  - -5% weekly loss → halt for 3 days, mandatory human review
"""

from datetime import datetime, timedelta
from src.utils.config_loader import CONFIG
from src.utils.logger import get_logger
from src.utils.telegram_alerts import alert_drawdown_halt

logger = get_logger("drawdown_monitor")

MAX_DAILY_LOSS_PCT = CONFIG["risk"]["max_daily_loss_pct"]    # 0.02
MAX_WEEKLY_LOSS_PCT = CONFIG["risk"]["max_weekly_loss_pct"]  # 0.05


class DrawdownMonitor:
    def __init__(self, starting_capital: float):
        self.starting_capital = starting_capital
        self.capital = starting_capital

        self.day_start_capital = starting_capital
        self.week_start_capital = starting_capital

        self.halted = False
        self.halt_reason = ""
        self.halt_until: datetime | None = None

    def update_capital(self, new_capital: float):
        self.capital = new_capital

    def check(self) -> bool:
        """
        Call this before every trade.
        Returns True if trading is allowed, False if halted.
        """
        # Check if halt period has expired
        if self.halted and self.halt_until:
            if datetime.now() >= self.halt_until:
                logger.info("Halt period expired — resuming trading")
                self.halted = False
                self.halt_reason = ""
                self.halt_until = None

        if self.halted:
            logger.warning(f"Bot is halted: {self.halt_reason}")
            return False

        # Daily drawdown check
        daily_loss_pct = (self.capital - self.day_start_capital) / self.day_start_capital
        if daily_loss_pct <= -MAX_DAILY_LOSS_PCT:
            self._halt(
                reason=f"Daily loss limit hit ({daily_loss_pct:.2%})",
                days=1
            )
            return False

        # Weekly drawdown check
        weekly_loss_pct = (self.capital - self.week_start_capital) / self.week_start_capital
        if weekly_loss_pct <= -MAX_WEEKLY_LOSS_PCT:
            self._halt(
                reason=f"Weekly loss limit hit ({weekly_loss_pct:.2%})",
                days=3
            )
            return False

        return True

    def on_market_open(self):
        """Call at 9:15am every trading day."""
        self.day_start_capital = self.capital
        logger.info(f"Market open — Day start capital: ₹{self.capital:.2f}")

    def on_week_start(self):
        """Call on Monday morning."""
        self.week_start_capital = self.capital
        logger.info(f"Week start capital: ₹{self.capital:.2f}")

    def _halt(self, reason: str, days: int):
        self.halted = True
        self.halt_reason = reason
        self.halt_until = datetime.now() + timedelta(days=days)
        logger.critical(f"🚨 BOT HALTED — {reason} — Resumes: {self.halt_until.date()}")
        alert_drawdown_halt(reason)

    def status(self) -> dict:
        daily_pnl_pct = (self.capital - self.day_start_capital) / self.day_start_capital
        weekly_pnl_pct = (self.capital - self.week_start_capital) / self.week_start_capital
        return {
            "capital": self.capital,
            "daily_pnl_pct": round(daily_pnl_pct * 100, 2),
            "weekly_pnl_pct": round(weekly_pnl_pct * 100, 2),
            "halted": self.halted,
            "halt_reason": self.halt_reason
        }
