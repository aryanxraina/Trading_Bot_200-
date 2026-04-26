"""
market_bias_calculator.py
Rule-based market bias engine. ZERO external dependencies.

Takes macro data (VIX, FII, S&P, crude, Nifty) and produces a market-level
bias score from -1.0 (very bearish) to +1.0 (very bullish).

The rules are based on your config.yaml thresholds + well-established
institutional trading logic:

  1. FII flow:   FII buying > 500cr = bullish, selling > 3000cr = very bearish
  2. India VIX:  < 13 = calm (bullish), 13-18 = normal, > 18 = fear (bearish)
  3. US markets: S&P up > 0.5% = tailwind, down > 1% = headwind
  4. Crude oil:  Big jump (>3%) = bearish for India (import-dependent)
  5. Nifty:      Yesterday's momentum — up > 0.5% = bullish continuation
  6. USD/INR:    Rupee weakening > 0.3% = bearish for equities

Each factor contributes a weighted score. Total is clipped to [-1, +1].

This is Mode 1 (rule-based). Always works. Zero cost. No LLM needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("market_bias")

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class MarketBias:
    date: str
    bias_label: str          # "bullish" | "bearish" | "neutral"
    bias_score: float        # [-1.0, +1.0]
    confidence: str          # "high" | "medium" | "low"

    # Individual factor scores (for transparency/debugging)
    fii_signal: float = 0.0
    vix_signal: float = 0.0
    us_market_signal: float = 0.0
    crude_signal: float = 0.0
    nifty_signal: float = 0.0
    rupee_signal: float = 0.0

    # Raw data (for logging)
    fii_net_cr: Optional[float] = None
    india_vix: Optional[float] = None
    sp500_change: Optional[float] = None
    crude_change: Optional[float] = None
    nifty_change: Optional[float] = None
    usdinr_change: Optional[float] = None

    # Meta
    factors_available: int = 0
    factors_total: int = 6
    recommendation: str = "hold"  # "increase_exposure" | "reduce_exposure" | "hold" | "no_trade"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Factor weights — how much each factor influences the final score
# These sum to ~1.0. FII has the most weight because it's the strongest
# predictor of next-day Nifty direction for Indian markets.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "fii":       0.30,   # FII flow is the single strongest signal
    "vix":       0.20,   # VIX predicts regime, not direction — lower weight
    "us_market": 0.20,   # S&P sets the tone for Asian open
    "crude":     0.10,   # India is a net importer — crude spikes are bad
    "nifty":     0.10,   # Yesterday's momentum (mean reversion is weak)
    "rupee":     0.10,   # Rupee weakness signals FII outflows
}


def _fii_signal(fii_net_cr: Optional[float]) -> float:
    """FII net flow in crores -> signal in [-1, +1]."""
    if fii_net_cr is None:
        return 0.0
    if fii_net_cr > 2000:
        return 1.0        # strong institutional buying
    elif fii_net_cr > 500:
        return 0.5        # moderate buying (your config threshold)
    elif fii_net_cr > 0:
        return 0.2
    elif fii_net_cr > -1000:
        return -0.2
    elif fii_net_cr > -3000:
        return -0.5       # moderate selling
    else:
        return -1.0       # heavy selling (your config: fii_sell_threshold=3000)


def _vix_signal(india_vix: Optional[float]) -> float:
    """India VIX level -> signal. Low VIX = calm = bullish."""
    if india_vix is None:
        return 0.0
    if india_vix < 13:
        return 0.6        # very calm — bullish
    elif india_vix < 15:
        return 0.3
    elif india_vix < 18:
        return 0.0        # neutral zone
    elif india_vix < 20:
        return -0.3       # getting fearful (your config: vix_threshold=20)
    elif india_vix < 25:
        return -0.6
    else:
        return -1.0       # panic territory


def _us_market_signal(sp500_change_pct: Optional[float]) -> float:
    """S&P 500 previous session change -> signal."""
    if sp500_change_pct is None:
        return 0.0
    if sp500_change_pct > 1.5:
        return 1.0        # strong US rally
    elif sp500_change_pct > 0.5:
        return 0.5
    elif sp500_change_pct > -0.5:
        return 0.0        # flat
    elif sp500_change_pct > -1.5:
        return -0.5
    else:
        return -1.0       # US crash — Indian open will gap down


def _crude_signal(crude_change_pct: Optional[float]) -> float:
    """Crude oil price change -> signal. Big crude spike = bad for India."""
    if crude_change_pct is None:
        return 0.0
    if crude_change_pct > 3.0:
        return -0.8       # crude spike — import bill goes up
    elif crude_change_pct > 1.0:
        return -0.3
    elif crude_change_pct < -3.0:
        return 0.5        # crude drop — good for India
    elif crude_change_pct < -1.0:
        return 0.2
    return 0.0            # small moves don't matter


def _nifty_signal(nifty_change_pct: Optional[float]) -> float:
    """Yesterday's Nifty change -> weak momentum signal."""
    if nifty_change_pct is None:
        return 0.0
    if nifty_change_pct > 1.0:
        return 0.5
    elif nifty_change_pct > 0.3:
        return 0.2
    elif nifty_change_pct < -1.0:
        return -0.5
    elif nifty_change_pct < -0.3:
        return -0.2
    return 0.0


def _rupee_signal(usdinr_change_pct: Optional[float]) -> float:
    """USD/INR change -> signal. Rupee weakening = bearish."""
    if usdinr_change_pct is None:
        return 0.0
    # USDINR UP means rupee is weakening
    if usdinr_change_pct > 0.5:
        return -0.7       # sharp rupee fall — FII outflow likely
    elif usdinr_change_pct > 0.2:
        return -0.3
    elif usdinr_change_pct < -0.3:
        return 0.3        # rupee strengthening — inflows
    return 0.0


def calculate_bias(market_data: dict) -> MarketBias:
    """
    Takes the output of fetch_all_market_data() and produces a MarketBias.
    Gracefully handles missing data — uses only what's available.
    """
    today = datetime.now(IST).strftime("%Y-%m-%d")

    # Calculate individual signals
    fii = _fii_signal(market_data.get("fii_net_cr"))
    vix = _vix_signal(market_data.get("india_vix"))
    us = _us_market_signal(market_data.get("sp500_change_pct"))
    crude = _crude_signal(market_data.get("crude_change_pct"))
    nifty = _nifty_signal(market_data.get("nifty_change_pct"))
    rupee = _rupee_signal(market_data.get("usdinr_change_pct"))

    signals = {
        "fii": fii,
        "vix": vix,
        "us_market": us,
        "crude": crude,
        "nifty": nifty,
        "rupee": rupee,
    }

    # Count available factors (non-zero means data was present)
    raw_values = {
        "fii": market_data.get("fii_net_cr"),
        "vix": market_data.get("india_vix"),
        "us_market": market_data.get("sp500_change_pct"),
        "crude": market_data.get("crude_change_pct"),
        "nifty": market_data.get("nifty_change_pct"),
        "rupee": market_data.get("usdinr_change_pct"),
    }
    factors_available = sum(1 for v in raw_values.values() if v is not None)

    # Weighted sum
    total_score = sum(signals[k] * WEIGHTS[k] for k in WEIGHTS)
    # Clip to [-1, 1]
    total_score = max(-1.0, min(1.0, total_score))

    # Label
    if total_score > 0.25:
        label = "bullish"
    elif total_score < -0.25:
        label = "bearish"
    else:
        label = "neutral"

    # Confidence based on data availability
    if factors_available >= 5:
        confidence = "high"
    elif factors_available >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    # Recommendation
    if total_score > 0.4:
        recommendation = "increase_exposure"
    elif total_score < -0.4:
        recommendation = "reduce_exposure"
    elif total_score < -0.6:
        recommendation = "no_trade"
    else:
        recommendation = "hold"

    bias = MarketBias(
        date=today,
        bias_label=label,
        bias_score=round(total_score, 4),
        confidence=confidence,
        fii_signal=round(fii, 3),
        vix_signal=round(vix, 3),
        us_market_signal=round(us, 3),
        crude_signal=round(crude, 3),
        nifty_signal=round(nifty, 3),
        rupee_signal=round(rupee, 3),
        fii_net_cr=market_data.get("fii_net_cr"),
        india_vix=market_data.get("india_vix"),
        sp500_change=market_data.get("sp500_change_pct"),
        crude_change=market_data.get("crude_change_pct"),
        nifty_change=market_data.get("nifty_change_pct"),
        usdinr_change=market_data.get("usdinr_change_pct"),
        factors_available=factors_available,
        recommendation=recommendation,
    )

    logger.info(
        f"Market bias: {label.upper()} ({total_score:+.2f}) | "
        f"confidence={confidence} | {factors_available}/6 factors | "
        f"rec={recommendation}"
    )
    return bias


if __name__ == "__main__":
    # Test with sample data
    sample = {
        "fii_net_cr": -1500,
        "india_vix": 16.5,
        "sp500_change_pct": -0.8,
        "crude_change_pct": 1.2,
        "nifty_change_pct": -0.5,
        "usdinr_change_pct": 0.3,
    }
    bias = calculate_bias(sample)
    print(f"\nBias: {bias.bias_label} ({bias.bias_score:+.3f})")
    print(f"Confidence: {bias.confidence}")
    print(f"Recommendation: {bias.recommendation}")
    print(f"\nFactor breakdown:")
    print(f"  FII:       {bias.fii_signal:+.2f} (net {bias.fii_net_cr}Cr)")
    print(f"  VIX:       {bias.vix_signal:+.2f} ({bias.india_vix})")
    print(f"  US market: {bias.us_market_signal:+.2f} (S&P {bias.sp500_change:+.1f}%)")
    print(f"  Crude:     {bias.crude_signal:+.2f} ({bias.crude_change:+.1f}%)")
    print(f"  Nifty:     {bias.nifty_signal:+.2f} ({bias.nifty_change:+.1f}%)")
    print(f"  Rupee:     {bias.rupee_signal:+.2f} ({bias.usdinr_change:+.1f}%)")
