"""
RedBlackBench: A Multi-Agent Game Theory Benchmark

Evaluating Cooperative Alignment Under Competitive Pressure
"""

__version__ = "0.1.0"
__author__ = "RedBlackBench Team"

from redblackbench.game.config import GameConfig
from redblackbench.game.coordinator import GameCoordinator, GameState
from redblackbench.game.scoring import Choice, ScoringMatrix

__all__ = [
    "GameConfig",
    "GameCoordinator",
    "GameState",
    "Choice",
    "ScoringMatrix",
]

