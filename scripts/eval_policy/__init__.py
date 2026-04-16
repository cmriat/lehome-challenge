"""
LeHome Challenge Policy Module

This module provides the base policy interface and implementations
for the LeHome Challenge evaluation framework.
"""

from .base_policy import BasePolicy
from .registry import PolicyRegistry

# Import policy implementations (this will auto-register them)
from .lerobot_policy import LeRobotPolicy
from .example_participant_policy import CustomPolicy
from .implicit_pi05_v1_policy import ImplicitPi05V1Policy
from .implicit_pi05_v2_policy import ImplicitPi05V2Policy
from .implicit_pi05_v3_policy import ImplicitPi05V3Policy

__all__ = [
    "BasePolicy",
    "PolicyRegistry",
    "LeRobotPolicy",
    "CustomPolicy",
    "ImplicitPi05V1Policy",
    "ImplicitPi05V2Policy",
    "ImplicitPi05V3Policy",
]
