"""Real-world scenario modules for RedBlackBench.

This module provides real-world framings for the Red-Black game that test
whether AI agents prioritize collective human welfare over tribal/group interests.
"""

from redblackbench.scenarios.base import Scenario, ScenarioConfig
from redblackbench.scenarios.registry import SCENARIOS, get_scenario

__all__ = ["Scenario", "ScenarioConfig", "SCENARIOS", "get_scenario"]
