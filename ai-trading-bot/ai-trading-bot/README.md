# AI Trading Bot — NSE/BSE

ML-powered algorithmic trading bot for Indian equity markets.
**Target:** ₹1,00,000 → ₹3,00,000 in 200 trading days (0.55%/day compounded).

---

## Quick Start

```bash
# 1. Clone and enter project
cd ai-trading-bot

# 2. Create conda environment
conda create -n trading python=3.10
conda activate trading

# 3. Install PyTorch with CUDA (RTX 5050 — CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Set up your API keys
cp .env.example .env
# Edit .env and add your Zerodha + Telegram keys

# 6. Start with Week 1 notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## Build Timeline

| Week | Task | Notebook |
|------|------|----------|
| 1–2 | Download data, explore, build LSTM + XGBoost | 01, 02, 03, 04 |
| 3 | Add FinBERT, regime detector, ensemble | 05 |
| 4 | Paper trade live market | 06 |
| 5+ | Go live with Phase 1 risk settings | `main.py` |

---

## Key Files

| File | Purpose |
|------|---------|
| `config/config.yaml` | All parameters — change phase, risk % here |
| `.env` | API keys — never commit |
| `src/data/download.py` | Week 1 starting point |
| `src/risk/drawdown_monitor.py` | Auto-halt on -2%/-5% loss |
| `src/execution/paper_trade.py` | Week 4 paper trading |
| `main.py` | Bot entry point |

---

## Hard Risk Rules

- `-2% daily loss` → bot halts automatically for the day
- `-5% weekly loss` → bot pauses 3 days, mandatory human review
- Max 2 open positions in Phase 1 & 2
- Every trade needs a stop-loss BEFORE entry
- Always keep 20% capital in cash

---

## Phase Risk Settings

| Phase | Days | Risk/Trade | Max Positions |
|-------|------|-----------|----------------|
| 1 | 1–50 | 0.5% | 2 |
| 2 | 51–100 | 1.0% | 2 |
| 3 | 101–150 | 2.0% | 3 |
| 4 | 151–200 | 2.5% | 4 |

To advance phases, edit `risk.phase` in `config/config.yaml`.
