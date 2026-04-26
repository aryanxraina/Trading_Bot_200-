"""
main.py
Entry point for the AI Trading Bot.
Run: python main.py
"""

from src.utils.config_loader import CONFIG
from src.utils.logger import get_logger

logger = get_logger("main")


def main():
    logger.info("=" * 50)
    logger.info("AI Trading Bot — Starting Up")
    logger.info(f"Phase: {CONFIG['risk']['phase']}")
    logger.info(f"Paper Trade Mode: {CONFIG['broker']['paper_trade']}")
    logger.info("=" * 50)

    paper_mode = CONFIG["broker"]["paper_trade"]

    if paper_mode:
        logger.info("Running in PAPER TRADE mode — no real orders will be placed")
        # TODO Week 4: wire up paper_trade.py + signal_generator.py here
    else:
        logger.info("Running in LIVE mode")
        # TODO Week 5+: wire up kite_broker.py + order_manager.py here

    logger.info("Bot initialised. Market loop not yet implemented.")
    logger.info("Start with notebooks/01_data_exploration.ipynb")


if __name__ == "__main__":
    main()
