"""
paper_trade.py
Simulates trades without real money.
Use this for all of Week 4 before going live.

Tracks: entries, exits, P&L, win rate, drawdown
"""

import json
import os
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger("paper_trade")

PAPER_TRADE_LOG = "logs/trades/paper_trades.json"


class PaperTrader:
    def __init__(self, starting_capital: float = 100000):
        self.capital = starting_capital
        self.starting_capital = starting_capital
        self.open_positions: dict = {}  # symbol → position dict
        self.closed_trades: list = []
        self._load()

    def buy(self, symbol: str, quantity: int, price: float, stop_loss: float) -> dict:
        if symbol in self.open_positions:
            logger.warning(f"Already have an open position in {symbol}")
            return {}

        cost = quantity * price
        if cost > self.capital:
            logger.warning(f"Insufficient capital: need ₹{cost:.2f}, have ₹{self.capital:.2f}")
            return {}

        self.capital -= cost
        position = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": price,
            "stop_loss": stop_loss,
            "entry_time": datetime.now().isoformat(),
            "cost": cost
        }
        self.open_positions[symbol] = position
        logger.info(f"PAPER BUY  | {symbol} x{quantity} @ ₹{price:.2f} | SL: ₹{stop_loss:.2f}")
        self._save()
        return position

    def sell(self, symbol: str, price: float, reason: str = "target") -> dict:
        if symbol not in self.open_positions:
            logger.warning(f"No open position for {symbol}")
            return {}

        pos = self.open_positions.pop(symbol)
        proceeds = pos["quantity"] * price
        pnl = proceeds - pos["cost"]
        self.capital += proceeds

        trade = {**pos, "exit_price": price, "exit_time": datetime.now().isoformat(),
                 "pnl": round(pnl, 2), "reason": reason}
        self.closed_trades.append(trade)

        icon = "✅" if pnl >= 0 else "🔴"
        logger.info(f"PAPER SELL | {symbol} @ ₹{price:.2f} | P&L: ₹{pnl:.2f} {icon}")
        self._save()
        return trade

    def check_stop_losses(self, current_prices: dict[str, float]):
        """Call this every candle with latest prices."""
        for symbol, pos in list(self.open_positions.items()):
            price = current_prices.get(symbol)
            if price and price <= pos["stop_loss"]:
                logger.info(f"Stop-loss triggered for {symbol} at ₹{price:.2f}")
                self.sell(symbol, price, reason="stop_loss")

    def summary(self) -> dict:
        if not self.closed_trades:
            return {"message": "No closed trades yet"}

        wins = [t for t in self.closed_trades if t["pnl"] > 0]
        losses = [t for t in self.closed_trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in self.closed_trades)

        return {
            "total_trades": len(self.closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(self.closed_trades) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "capital": round(self.capital, 2),
            "return_pct": round((self.capital - self.starting_capital) / self.starting_capital * 100, 2),
            "open_positions": list(self.open_positions.keys())
        }

    def _save(self):
        os.makedirs("logs/trades", exist_ok=True)
        data = {
            "capital": self.capital,
            "open_positions": self.open_positions,
            "closed_trades": self.closed_trades
        }
        with open(PAPER_TRADE_LOG, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if os.path.exists(PAPER_TRADE_LOG):
            with open(PAPER_TRADE_LOG) as f:
                data = json.load(f)
            self.capital = data.get("capital", self.starting_capital)
            self.open_positions = data.get("open_positions", {})
            self.closed_trades = data.get("closed_trades", [])
            logger.info(f"Loaded paper trade state: ₹{self.capital:.2f} capital, "
                        f"{len(self.closed_trades)} closed trades")


if __name__ == "__main__":
    trader = PaperTrader(100000)
    trader.buy("RELIANCE.NS", 5, 2900, 2850)
    trader.sell("RELIANCE.NS", 2950, reason="target")
    print(trader.summary())
