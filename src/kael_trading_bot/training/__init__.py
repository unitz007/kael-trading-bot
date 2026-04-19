"""ML model training pipeline for forex pair trading.

This package provides:
- :mod:`models` — model registry and factory
- :mod:`splitting` — time-aware train/val/test splitting
- :mod:`evaluation` — classification and trading-oriented metrics
- :mod:`persistence` — model save/load with metadata
- :mod:`logging` — structured training-run logging
- :mod:`pipeline` — end-to-end training pipeline orchestrator
"""

from src.kael_trading_bot.training.models import ModelRegistry, ModelType

__all__ = [
    "ModelRegistry",
    "ModelType",
]
