"""
position_sizer.py
Calculates position size based on:
  - Current phase (1–4) from config
  - Risk per trade % for that phase
  - Stop-loss distance
  - Regime multiplier (reduces size if VIX high or bear market)
"""

from src.utils.config_loader import CONFIG
from src.utils.logger import get_logger

logger = get_logger("position_sizer")

PHASE = CONFIG["risk"]["phase"]
PHASE_SETTINGS = CONFIG["risk"]["phase_settings"]
CASH_RESERVE_PCT = CONFIG["capital"]["cash_reserve_pct"]


def get_risk_per_trade_pct(phase: int = None) -> float:
    phase = phase or PHASE
    return PHASE_SETTINGS[phase]["risk_per_trade_pct"]


def get_max_positions(phase: int = None) -> int:
    phase = phase or PHASE
    return PHASE_SETTINGS[phase]["max_open_positions"]


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss_price: float,
    regime_multiplier: float = 1.0,
    phase: int = None
) -> dict:
    """
    Kelly-lite position sizing based on fixed risk %.

    Args:
        capital: Current total capital
        entry_price: Planned entry price per share
        stop_loss_price: Stop-loss price per share
        regime_multiplier: 0.5 if high VIX/bear, 1.0 otherwise
        phase: Override current phase (optional)

    Returns:
        {
            "quantity": int,
            "risk_amount": float,
            "max_loss": float,
            "position_value": float
        }
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("Prices must be positive")
    if stop_loss_price >= entry_price:
        raise ValueError("Stop-loss must be below entry price for long trades")

    # Deployable capital (excluding cash reserve)
    deployable = capital * (1 - CASH_RESERVE_PCT)

    # Risk amount in ₹
    risk_pct = get_risk_per_trade_pct(phase)
    risk_amount = deployable * risk_pct * regime_multiplier

    # Stop-loss distance per share
    sl_distance = entry_price - stop_loss_price

    if sl_distance <= 0:
        raise ValueError("Stop-loss distance must be positive")

    # Quantity = risk_amount / sl_distance
    quantity = int(risk_amount / sl_distance)

    if quantity <= 0:
        logger.warning("Calculated quantity is 0 — position too small for this capital/risk setting")
        return {"quantity": 0, "risk_amount": 0, "max_loss": 0, "position_value": 0}

    position_value = quantity * entry_price
    max_loss = quantity * sl_distance

    result = {
        "quantity": quantity,
        "risk_amount": round(risk_amount, 2),
        "max_loss": round(max_loss, 2),
        "position_value": round(position_value, 2),
        "risk_pct_used": round(risk_pct * regime_multiplier * 100, 3)
    }

    logger.info(
        f"Position size: {quantity} shares @ ₹{entry_price} "
        f"| SL: ₹{stop_loss_price} | Max loss: ₹{max_loss:.2f}"
    )
    return result


if __name__ == "__main__":
    # Example: ₹1,00,000 capital, Phase 1, buying at ₹2500 with SL at ₹2450
    result = calculate_position_size(
        capital=100000,
        entry_price=2500,
        stop_loss_price=2450,
        regime_multiplier=1.0,
        phase=1
    )
    print(result)
