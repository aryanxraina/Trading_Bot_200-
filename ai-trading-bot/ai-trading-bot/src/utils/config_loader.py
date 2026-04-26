"""
config_loader.py
Loads config.yaml and injects .env variables safely.
"""

import os
import yaml
from dotenv import load_dotenv

load_dotenv()  # Load .env file


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        raw = f.read()

    # Replace ${VAR_NAME} placeholders with env values
    for key, value in os.environ.items():
        raw = raw.replace(f"${{{key}}}", value)

    config = yaml.safe_load(raw)
    return config


# Singleton — import this anywhere
CONFIG = load_config()


if __name__ == "__main__":
    import pprint
    pprint.pprint(CONFIG)
